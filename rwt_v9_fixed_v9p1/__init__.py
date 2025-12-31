
bl_info = {
    "name": "Supergroup Weight Transfer",
    "version": (0, 1, 0),
    "blender": (3, 1, 0),
    "location": "View3D > Sidebar > SENT Tab",
    "category": "Object",
}

import sys
import os
import sysconfig
import site

import bpy
from mathutils import Vector
import bmesh
import numpy as np
import math
import importlib
import subprocess
import bpy.utils.previews

from . import thresholds as T
from . import ranges as R

libs_path = os.path.join(os.path.dirname(__file__), "deps")

def _ensure_deps_site_paths() -> None:
    """Ensure our private dependency install locations are on sys.path."""
    candidates = set()

    # Preferred user scheme for the running interpreter
    try:
        scheme = sysconfig.get_preferred_scheme("user")
        candidates.add(sysconfig.get_paths(scheme, vars={"userbase": libs_path})["purelib"])
    except Exception:
        pass

    maj, min_ = sys.version_info.major, sys.version_info.minor
    # Windows-style pip --user under PYTHONUSERBASE
    candidates.add(os.path.join(libs_path, f"Python{maj}{min_}", "site-packages"))
    # Some builds may use Lib/site-packages
    candidates.add(os.path.join(libs_path, "Lib", "site-packages"))
    # POSIX-style layouts
    candidates.add(os.path.join(libs_path, "lib", f"python{maj}.{min_}", "site-packages"))

    for p in sorted(candidates):
        if os.path.isdir(p):
            site.addsitedir(p)

# Add site paths at import-time so importlib can find deps right away.
_ensure_deps_site_paths()
# -------------------------------------------------------------------
# Dependency management
#
# Goal:
# - Check dependencies on add-on load (import-time) and keep UI clean:
#   * If deps are importable -> do NOT show install UI.
#   * If deps are missing -> show install button (manual).
# - Never auto-install during registration.
# - Map import names to pip requirements explicitly.
# -------------------------------------------------------------------

_DEP_SPECS = [
    # (import_name, pip_requirement)
    ("robust_laplacian", "robust-laplacian"),
    ("igl", "libigl==2.5.1"),
    ("scipy", "scipy"),
]

missing_deps: list[str] = []
installed_deps: bool = False
_ALGO_LOADED: bool = False


# Cache for UI-only "band reference" display (avoid heavy recomputation on every redraw).
_BAND_REF_CACHE: dict = {}


def _compute_supergroup_ui_info(context, settings):
    """Compute supergroup band metrics for UI display.

    This is UI-only and must not be required by the core transfer execution.

    Returns a dict with keys:
      - eps_force_raw / eps_norm_raw / face_raw (world)
      - eps_force / eps_norm / face_h_eps (effective, world)
      - bbox_diag, d_nn_median, eps_min, eps_max
      - clamp_ctx (ranges.ClampContext or None)
      - range_mid / range_min / range_face (ranges.RangeResult)
    """
    # util is loaded lazily once deps are installed.
    if 'util' not in globals():
        return None

    src = getattr(settings, 'source_object', None)
    if getattr(settings, 'apply_to_selected', False):
        candidates = [
            o for o in context.selected_objects
            if o is not None and o != src and getattr(o, 'type', None) == 'MESH'
        ]
    else:
        o = context.object
        candidates = [o] if (o is not None and o != src and getattr(o, 'type', None) == 'MESH') else []

    if not candidates:
        return None

    obj = candidates[0]
    depsgraph = context.evaluated_depsgraph_get() if hasattr(context, 'evaluated_depsgraph_get') else None

    # Mirror execute() behavior for evaluated target where safe.
    obj_eval = obj
    if getattr(settings, 'use_deformed_target', False) and depsgraph is not None:
        try:
            eo = obj.evaluated_get(depsgraph)
            if len(eo.data.vertices) == len(obj.data.vertices):
                obj_eval = eo
        except Exception:
            obj_eval = obj

    strong_clean = bool(getattr(settings, 'strong_clean_enable', True))
    rel_enable_setting = bool(getattr(settings, 'supergroup_rel_clamp_enable', False))
    rel_enable = bool(strong_clean and rel_enable_setting)
    clean_faces_setting = bool(getattr(settings, 'clean_degenerate_faces', False))
    face_c = float(getattr(settings, 'face_clean_height', 0.0))

    key = (
        int(obj_eval.as_pointer()),
        int(getattr(obj_eval.data, 'as_pointer', lambda: 0)()),
        int(len(obj_eval.data.vertices)) if getattr(obj_eval, 'data', None) else 0,
        float(getattr(settings, 'supergroup_eps_min', T.SUPERGROUP_EPS_MINSCALE_DEFAULT)),
        float(getattr(settings, 'supergroup_eps_mid', T.SUPERGROUP_EPS_MIDSCALE_DEFAULT)),
        float(getattr(settings, 'supergroup_eps_rel_min', T.SUPERGROUP_EPS_REL_MIN)),
        float(getattr(settings, 'supergroup_eps_rel_max', T.SUPERGROUP_EPS_REL_MAX)),
        bool(strong_clean),
        bool(rel_enable),
        bool(clean_faces_setting),
        float(face_c),
    )
    cached_key = _BAND_REF_CACHE.get('key')
    if cached_key == key:
        return _BAND_REF_CACHE.get('ui_info')

    try:
        V_world, _F, _N = util.get_obj_arrs_world(obj_eval)
        bbox = util.object_scale_world(V_world)
        # Raw (pre-clamp) values driven by UI (direct world-unit eps).
        eps_force_raw = float(getattr(settings, 'supergroup_eps_min', T.SUPERGROUP_EPS_MINSCALE_DEFAULT))
        eps_norm_raw = float(getattr(settings, 'supergroup_eps_mid', T.SUPERGROUP_EPS_MIDSCALE_DEFAULT))

        # Face-clean height raw (pre-clamp)
        face_raw = float(getattr(settings, 'face_clean_height', T.FACE_CLEAN_HEIGHT_DEFAULT))

        
        eps_abs_min = float(getattr(T, 'SUPERGROUP_EPS_ABS_MIN', 1e-12))

        # REL clamp is mesh-density based and varies per-vertex; UI can only estimate from samples.
        d_nn_median = 0.0
        eps_min_ref = float(eps_abs_min)
        eps_max_ref = float('inf')
        clamp_ctx = None
        rel_min = float(getattr(settings, 'supergroup_eps_rel_min', T.SUPERGROUP_EPS_REL_MIN))
        rel_max = float(getattr(settings, 'supergroup_eps_rel_max', T.SUPERGROUP_EPS_REL_MAX))

        if rel_enable:
            clamp_ctx = R.build_clamp_context(
                verts_world=V_world,
                rel_min=rel_min,
                rel_max=rel_max,
                eps_abs_min=eps_abs_min,
                sample_n=500,
            )
            if clamp_ctx.nn_distance_samples.size:
                d_nn_median = float(np.median(clamp_ctx.nn_distance_samples))
                eps_min_ref, eps_max_ref = R.clamp_bounds_percentiles(clamp_ctx)

        # Display-only "effective" (median) values for the current raw inputs.
        eps_force = float(eps_force_raw)
        eps_norm = float(eps_norm_raw)

        eps_min_s = None
        eps_max_s = None
        if rel_enable and clamp_ctx is not None and clamp_ctx.nn_distance_samples.size:
            d = np.asarray(clamp_ctx.nn_distance_samples, dtype=np.float64).reshape(-1)
            eps_min_s = np.maximum(float(eps_abs_min), d * float(rel_min))
            eps_max_s = np.maximum(d * float(rel_max), eps_min_s)

            eff_norm = np.maximum(eps_min_s, np.minimum(float(eps_norm_raw), eps_max_s))
            eps_norm = float(np.percentile(eff_norm, 50))

            eff_force = np.maximum(eps_min_s, np.minimum(float(eps_force_raw), eps_max_s))
            eps_force = float(np.percentile(eff_force, 50))

        # Hierarchy (display): MIN <= MID/5
        if eps_force > (eps_norm / 5.0):
            eps_force = eps_norm / 5.0

        face_h_eps = None
        if strong_clean and clean_faces_setting:
            face_raw_val = float(face_raw)
            if eps_min_s is not None and eps_max_s is not None:
                eff_face = np.maximum(eps_min_s, np.minimum(face_raw_val, eps_max_s))
                face_h_eps = float(np.percentile(eff_face, 50))
            else:
                face_h_eps = max(float(eps_abs_min), face_raw_val)
            if face_h_eps > (eps_norm / 5.0):
                face_h_eps = eps_norm / 5.0

        # UI-only reference ranges (do not modify fill behavior).
        range_mid = R.calculate_midband_reference_range(
            bbox_diag=float(bbox),
            raw_value=float(eps_norm_raw),
            clamp_enabled=bool(rel_enable),
            clamp_context=clamp_ctx,
        )
        range_min = R.calculate_minband_reference_range(
            bbox_diag=float(bbox),
            midband_raw=float(eps_norm_raw),
            raw_value=float(eps_force_raw),
            clamp_enabled=bool(rel_enable),
            clamp_context=clamp_ctx,
            hierarchy_div=5.0,
        )
        range_face = R.calculate_face_clean_reference_range(
            bbox_diag=float(bbox),
            midband_raw=float(eps_norm_raw),
            raw_value=float(face_raw),
            clamp_enabled=bool(rel_enable),
            clamp_context=clamp_ctx,
            hierarchy_div=5.0,
        )

        ui_info = {

            'bbox_diag': float(bbox),
            'd_nn_median': float(d_nn_median),
            'eps_min': float(eps_min_ref),
            'eps_max': float(eps_max_ref),
            'eps_force_raw': float(eps_force_raw),
            'eps_norm_raw': float(eps_norm_raw),
            'face_raw': float(face_raw),
            'eps_force': float(eps_force),
            'eps_norm': float(eps_norm),
            'face_h_eps': (None if face_h_eps is None else float(face_h_eps)),
            'clamp_ctx': clamp_ctx,
            'range_mid': range_mid,
            'range_min': range_min,
            'range_face': range_face,
        }

    except Exception:
        return None

    _BAND_REF_CACHE['key'] = key
    _BAND_REF_CACHE['ui_info'] = ui_info
    return ui_info


def _compute_band_refs_for_ui(context, settings):
    """Compute band reference values (world units) for UI display.

    Returns (eps_force, eps_norm, face_h_eps) where face_h_eps may be None.
    """
    info = _compute_supergroup_ui_info(context, settings)
    if not info:
        return None, None, None
    return info.get('eps_force'), info.get('eps_norm'), info.get('face_h_eps')



def _ui_pick_target_mesh_for_calc(context, settings):
    """Pick the target mesh object used for UI calculations.

    This matches the selection behavior of the main operator as closely as possible,
    but is intentionally lightweight.
    """
    src = getattr(settings, 'source_object', None)
    # Candidate selection: use selected meshes if apply_to_selected_objects, else active object.
    candidates = []
    if getattr(settings, 'apply_to_selected_objects', False):
        for obj in context.selected_objects:
            if obj and obj.type == 'MESH' and obj != src:
                candidates.append(obj)
    else:
        obj = context.object
        if obj and obj.type == 'MESH' and obj != src:
            candidates.append(obj)

    if not candidates:
        return None

    obj = candidates[0]
    if getattr(settings, 'use_deformed_target', False):
        dg = context.evaluated_depsgraph_get()
        try:
            return obj.evaluated_get(dg)
        except Exception:
            return obj
    return obj




def _ui_bbox_diag_world(obj) -> float:
    """Compute world-space bbox diagonal for UI range display (no deps)."""
    if obj is None or getattr(obj, 'type', None) != 'MESH':
        return 0.0
    try:
        # bound_box corners are in local space
        pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        min_v = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
        max_v = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
        return (max_v - min_v).length
    except Exception:
        try:
            d = obj.dimensions
            return float((d.x * d.x + d.y * d.y + d.z * d.z) ** 0.5)
        except Exception:
            return 0.0

def _ui_make_prob_region_token(kind: str, obj, settings) -> str:
    """Create a lightweight signature for UI prob-region results."""
    if obj is None or obj.type != 'MESH':
        return f"{kind}:<none>"
    n_v = 0
    try:
        n_v = len(obj.data.vertices)
    except Exception:
        n_v = 0

    # Use rounded values to reduce noise.
    def r(x, nd=6):
        try:
            return round(float(x), nd)
        except Exception:
            return x

    parts = [kind, obj.name, str(n_v), str(bool(getattr(settings, 'use_deformed_target', False)))]
    if kind == 'mid':
        parts += [
            str(r(getattr(settings, 'supergroup_eps_mid', 0.0))),
            str(r(getattr(settings, 'supergroup_theta_merge_deg', 60.0))),
            str(r(getattr(settings, 'supergroup_mix_alpha', 0.7))),
            str(bool(getattr(settings, 'supergroup_rel_clamp_enable', True))),
            str(int(getattr(settings, 'supergroup_eps_rel_min', 1))),
            str(int(getattr(settings, 'supergroup_eps_rel_max', 1))),
        ]
    elif kind == 'min':
        parts += [
            str(r(getattr(settings, 'supergroup_eps_min', 0.0))),
            str(r(getattr(settings, 'supergroup_eps_mid', 0.0))),  # MIN depends on MID via hierarchy (eps_force <= eps_norm/5).
            str(bool(getattr(settings, 'supergroup_rel_clamp_enable', True))),
            str(int(getattr(settings, 'supergroup_eps_rel_min', 1))),
            str(int(getattr(settings, 'supergroup_eps_rel_max', 1))),
        ]
    elif kind == 'face':
        parts += [str(r(getattr(settings, 'face_clean_height', -6.0)))]
    return "|".join(parts)


def _check_missing_deps() -> list[str]:
    """Return missing dependencies as pip requirement strings."""
    importlib.invalidate_caches()
    missing: list[str] = []
    for import_name, pip_req in _DEP_SPECS:
        if importlib.util.find_spec(import_name) is None:
            missing.append(pip_req)
    return missing


def _load_algo_modules() -> None:
    """Import (and hot-reload) add-on submodules that rely on optional deps."""
    global _ALGO_LOADED
    global util, _rwt_weighttransfer
    global find_matches_closest_surface, inpaint, limit_mask, smooth_weigths

    if _ALGO_LOADED:
        return

    # Hot-reload safety: Blender often reloads only __init__.py.
    if __name__ + '.thresholds' in sys.modules:
        importlib.reload(sys.modules[__name__ + '.thresholds'])
    if __name__ + '.ranges' in sys.modules:
        importlib.reload(sys.modules[__name__ + '.ranges'])
    if __name__ + '.util' in sys.modules:
        importlib.reload(sys.modules[__name__ + '.util'])
    if __name__ + '.weighttransfer' in sys.modules:
        importlib.reload(sys.modules[__name__ + '.weighttransfer'])

    from . import util as _util
    from . import weighttransfer as _wt
    from .weighttransfer import (
        find_matches_closest_surface as _fmcs,
        inpaint as _inpaint,
        limit_mask as _limit_mask,
        smooth_weigths as _smooth_weigths,
    )

    util = _util
    _rwt_weighttransfer = _wt
    find_matches_closest_surface = _fmcs
    inpaint = _inpaint
    limit_mask = _limit_mask
    smooth_weigths = _smooth_weigths

    _ALGO_LOADED = True


def refresh_dependency_state(*, load_modules: bool = True) -> None:
    """Refresh globals `missing_deps` / `installed_deps`, optionally loading modules."""
    importlib.invalidate_caches()
    global missing_deps, installed_deps

    # Make sure the add-on's private site-packages are visible.
    _ensure_deps_site_paths()
    missing_deps = _check_missing_deps()
    installed_deps = (len(missing_deps) == 0)

    # Only load heavy modules when deps are satisfied.
    if installed_deps and load_modules:
        _load_algo_modules()


# Do an initial check at import-time.
refresh_dependency_state(load_modules=True)



def _tag_redraw_all_areas(context) -> None:
    try:
        win = context.window
        if not win:
            return
        screen = win.screen
        if not screen:
            return
        for area in screen.areas:
            area.tag_redraw()
    except Exception:
        pass

"""Add-on operators and UI."""



class InstallDependencies(bpy.types.Operator):
    """Install missing Python dependencies (isolated to add-on folder)."""
    bl_idname = "wm.install_rwt_dependencies"
    bl_label = "Install Dependencies"

    def execute(self, context):
        python_exe = sys.executable
        _ensure_deps_site_paths()
        missing = _check_missing_deps()
        if not missing:
            refresh_dependency_state(load_modules=True)
            self.report({'INFO'}, "All dependencies are already available.")
            _tag_redraw_all_areas(context)
            return {'FINISHED'}
        try:
            # Constrain numpy to Blender's bundled version to avoid ABI conflicts.
            constraints_path = os.path.join(os.path.dirname(__file__), "constraints.txt")
            with open(constraints_path, "w", encoding="utf-8") as f:
                f.write(f"numpy=={np.__version__}")

            # Install into our private userbase under ./deps
            env = os.environ.copy()
            env["PYTHONUSERBASE"] = libs_path

            # Ensure pip exists
            try:
                subprocess.check_call([python_exe, "-m", "pip", "--version"], env=env)
            except Exception:
                subprocess.check_call([python_exe, "-m", "ensurepip", "--upgrade"], env=env)

            # Install missing deps
            subprocess.check_call(
                [python_exe, "-m", "pip", "install", "--user", *missing, "--break-system-packages", "-c", constraints_path],
                env=env,
            )
            _ensure_deps_site_paths()
            refresh_dependency_state(load_modules=True)
            if installed_deps:
                self.report({'INFO'}, "Installation successful! Dependencies are now importable; no restart needed.")
            else:
                self.report({'INFO'}, "Installed, but deps are still not importable in this session. Restart Blender.")
            _tag_redraw_all_areas(context)
            return {'FINISHED'}
        except subprocess.CalledProcessError as e:
            self.report({'ERROR'}, f"Installation failed: {str(e)}")
            return {'CANCELLED'}


class RobustWeightTransfer(bpy.types.Operator):
    """Transfer Skin Weights Robust"""
    bl_idname = "object.skin_weight_transfer"
    bl_label = "Robust Weight Transfer"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        # Only available in Mesh Object/Edit contexts (disabled for Armature Edit/Pose)
        if context.mode not in {'OBJECT', 'EDIT_MESH'}:
            return False

        if missing_deps:
            return False
        
        scene_settings: SceneSettingsGroup = context.scene.robust_weight_transfer_settings
        if not scene_settings.source_object: return False
        if not scene_settings.apply_to_selected and scene_settings.source_object == context.active_object: return False
        if scene_settings.group_selection == 'DEFORM_POSE_BONES':
            armature_mods = [mod for mod in scene_settings.source_object.modifiers if mod.type == "ARMATURE"]
            if len(armature_mods) != 1: return False # no armature modifier or more than one
            if not armature_mods[0].object: return False

            
        objs = lambda x: [obj for obj in x if obj != scene_settings.source_object and isinstance(obj.data, bpy.types.Mesh)]
        if scene_settings.apply_to_selected:
            target_objs = objs(context.selected_objects)
        else:
            if not context.object: return False
            target_objs = objs([context.object])
            
        if len(target_objs) == 0: return False
        if scene_settings.use_deformed_target and any(util.has_modifier(obj, *util.TOPOLOGY_MODS) for obj in target_objs): return False
        
        if not scene_settings.apply_to_selected:
            obj = target_objs[0]
            object_settings: ObjectSettingsGroup = obj.robust_weight_transfer_settings

        # Lazy import: avoid NameError when libigl is available but not imported at module scope.
        try:
            igl = importlib.import_module("igl")
        except Exception as e:
            self.report({'ERROR'}, f"libigl (igl) is not importable: {e}")
            return {'CANCELLED'}

            mask = object_settings.vertex_group
            if len(mask) > 0 and mask not in obj.vertex_groups: return False
            inpaint = object_settings.inpaint_group
            if len(inpaint) > 0 and  inpaint not in obj.vertex_groups: return False
        return True


    def execute(self, context: bpy.types.Context):
        scene_settings: SceneSettingsGroup = context.scene.robust_weight_transfer_settings
        
        source_obj: bpy.types.Object = scene_settings.source_object
        if source_obj.type != 'MESH':
            self.report({'ERROR'}, f'Source object {source_obj.name} is not a mesh')
            return {'CANCELLED'}
    
        if scene_settings.apply_to_selected:
            target_objs = [obj for obj in context.selected_objects if obj != source_obj and isinstance(obj.data, bpy.types.Mesh)]
        else:
            target_objs = [context.object]
        
        depsgraph = context.evaluated_depsgraph_get()
        source_obj_raw = source_obj
        source_obj_geom = source_obj
        if scene_settings.use_deformed_source:
            try:
                _eval_src = source_obj_raw.evaluated_get(depsgraph)
                if len(_eval_src.data.vertices) != len(source_obj_raw.data.vertices):
                    self.report({'WARNING'},
                               f"Source '{source_obj_raw.name}': evaluated mesh changes topology (verts {len(source_obj_raw.data.vertices)} -> {len(_eval_src.data.vertices)}). "
                               f"Falling back to original mesh for computation.")
                else:
                    source_obj_geom = _eval_src
            except Exception:
                source_obj_geom = source_obj_raw

        weights_all = []  # (num_objs, (num_vertices, num_weights))
        source_verts, source_triangles, source_normals = util.get_obj_arrs_world(source_obj_geom)
        deform_only = scene_settings.group_selection == 'DEFORM_POSE_BONES'
        is_deform = [util.is_vertex_group_deform_bone(source_obj_raw, g.name) for g in source_obj_raw.vertex_groups]
        source_weights = util.get_groups_arr(source_obj_raw, is_deform if deform_only else None)  # (num_verts, num_groups)

        # ---- Precompute source supergroups & representative weights (process once per execute) ----
        # Supergroup construction thresholds are derived from object scale:
        #   eps_force = max(1e-12, ||obj|| * 1e-7)
        #   eps_norm  = max(1e-12, ||obj|| * 1e-3) with normal angle < 45°
        
# Source is always used as the original triangle mesh (no supergroups on source).
# Lazy import: avoid NameError when libigl is available but not imported at module scope.
        try:
            igl = importlib.import_module("igl")
        except Exception as e:
            self.report({'ERROR'}, f"libigl (igl) is not importable: {e}. Use the 'Install Dependencies' button in the add-on panel.")
            return {'CANCELLED'}

        # Topology safety: when using evaluated (deformed) meshes, we require the vertex count to match
        # the original mesh to safely write weights back to obj.data vertex groups.
        for obj in target_objs:
            object_settings: ObjectSettingsGroup = obj.robust_weight_transfer_settings

            _geom_obj = obj
            if scene_settings.use_deformed_target:
                try:
                    _eval_obj = obj.evaluated_get(depsgraph)
                    if len(_eval_obj.data.vertices) != len(obj.data.vertices):
                        self.report({'WARNING'},
                                   f"Target '{obj.name}': evaluated mesh changes topology (verts {len(obj.data.vertices)} -> {len(_eval_obj.data.vertices)}). "
                                   f"Falling back to original mesh for computation.")
                    else:
                        _geom_obj = _eval_obj
                except Exception:
                    _geom_obj = obj

            verts, triangles, normals = util.get_obj_arrs_world(_geom_obj)

            # --- Target supergroups (structure layer) ---
            # Build target supergroups on the evaluated target mesh (world space).
            # Supergroup representative position/normal = mean of member vertices.
            try:
                from scipy.spatial import cKDTree
                from scipy import sparse
                from scipy.sparse import linalg as spla
            except Exception as e:
                self.report({'ERROR'}, f"scipy is not importable: {e}. Use the 'Install Dependencies' button in the add-on panel.")
                return {'CANCELLED'}

            
            bbox = util.object_scale_world(verts)  # evaluated bbox diagonal length (world)
            strong_clean = bool(getattr(scene_settings, 'strong_clean_enable', True))
            rel_enable_setting = bool(getattr(scene_settings, 'supergroup_rel_clamp_enable', False))
            rel_enable = bool(strong_clean and rel_enable_setting)

            # Raw thresholds (world units): UI values are direct eps (no log / bbox scaling).
            eps_force_raw = float(getattr(scene_settings, 'supergroup_eps_min', T.SUPERGROUP_EPS_MINSCALE_DEFAULT))
            eps_norm_raw  = float(getattr(scene_settings, 'supergroup_eps_mid', T.SUPERGROUP_EPS_MIDSCALE_DEFAULT))

            eps_abs_min = float(getattr(T, 'SUPERGROUP_EPS_ABS_MIN', 1e-12))

            n_v = int(verts.shape[0])
            d_nn_v = None
            eps_min_v = None
            eps_max_v = None
            if rel_enable and n_v > 1:
                rel_min = float(getattr(scene_settings, 'supergroup_eps_rel_min', T.SUPERGROUP_EPS_REL_MIN))
                rel_max = float(getattr(scene_settings, 'supergroup_eps_rel_max', T.SUPERGROUP_EPS_REL_MAX))

                # Per-vertex local NN scale (world). This drives per-vertex clamp.
                kdt_v = cKDTree(verts)
                dists, _ = kdt_v.query(verts, k=2)  # self + nearest
                d_nn_v = dists[:, 1] if (dists.ndim == 2 and dists.shape[1] > 1) else np.zeros((n_v,), dtype=np.float64)

                eps_min_v = np.maximum(eps_abs_min, d_nn_v * rel_min)
                eps_max_v = np.maximum(eps_min_v, d_nn_v * rel_max)

                eps_force = np.maximum(eps_min_v, np.minimum(eps_force_raw, eps_max_v))
                eps_norm  = np.maximum(eps_min_v, np.minimum(eps_norm_raw,  eps_max_v))

                # Hierarchy: MIN <= MID/5 (per-vertex)
                eps_force = np.minimum(eps_force, eps_norm / 5.0)
            else:
                # No REL clamp: keep raw values (only abs_min + hierarchy)
                eps_force = max(eps_abs_min, float(eps_force_raw))
                eps_norm  = max(eps_abs_min, float(eps_norm_raw))
                if eps_force > (eps_norm / 5.0):
                    eps_force = eps_norm / 5.0

            theta_merge_deg = float(getattr(scene_settings, 'supergroup_theta_merge_deg', T.SUPERGROUP_THETA_MERGE_DEG))

            if not bool(getattr(scene_settings, 'supergroup_master_enable', True)):
                # Fallback: disable merging; treat each vertex as its own supergroup.
                n_v = int(verts.shape[0])
                target_sgid = np.arange(n_v, dtype=np.int64)
                target_comps = [[i] for i in range(n_v)]
            else:
                target_sgid, target_comps = util.build_supergroups_from_world(
                    verts, normals,
                    eps_force=eps_force, eps_norm=eps_norm, angle_deg=theta_merge_deg
                )
            n_sg = len(target_comps)

            # Representative points/normals
            rep_pos = np.zeros((n_sg, 3), dtype=np.float64)
            rep_nrm = np.zeros((n_sg, 3), dtype=np.float64)
            for sid, comp in enumerate(target_comps):
                if not comp:
                    continue
                idx = np.asarray(comp, dtype=np.int64)
                rp = verts[idx].mean(axis=0)
                rn = normals[idx].mean(axis=0)
                nn = float(np.linalg.norm(rn))
                if nn > 0.0:
                    rn = rn / nn
                else:
                    rn = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                rep_pos[sid] = rp
                rep_nrm[sid] = rn

            # --- Optional strong cleaning: degenerate-face prefilter threshold (world height) ---
            effective_clean_faces = bool(strong_clean and getattr(scene_settings, 'clean_degenerate_faces', False))
            face_height_eps = None
            if effective_clean_faces:
                face_raw = float(getattr(scene_settings, 'face_clean_height', T.FACE_CLEAN_HEIGHT_DEFAULT))
                face_height_eps = max(eps_abs_min, face_raw)

                # Optional: apply a global clamp window estimate when REL clamp is enabled.
                if rel_enable and eps_min_v is not None and eps_max_v is not None:
                    eps_min_g = float(np.percentile(eps_min_v, 5.0))
                    eps_max_g = float(np.percentile(eps_max_v, 95.0))
                    face_height_eps = max(eps_min_g, min(face_height_eps, eps_max_g))

                # Hierarchy cap (keep consistent with MIN <= MID/5 reference).
                face_height_eps = min(face_height_eps, float(eps_norm_raw) / 5.0)

            # --- Triangle match (supergroup representative points P) ---
            matched_sg, rep_weights, sqrD, deg_angles, pass_dist, pass_norm, face_I = find_matches_closest_surface(
                source_verts,
                source_triangles,
                source_normals,
                rep_pos,
                rep_nrm,
                source_weights,
                scene_settings.max_distance**2,
                math.degrees(scene_settings.max_normal_angle_difference) + getattr(scene_settings, 'normal_angle_offset_deg', 0.0),
                scene_settings.flip_vertex_normal,
                clean_degenerate_faces=effective_clean_faces,
                face_height_eps=face_height_eps,
                return_diagnostics=True,
            )

            # Match gate for anchoring/island rules: distance AND normal must pass.
            gate = np.logical_and(pass_dist, pass_norm).astype(bool, copy=False)
            matched_sg = gate.copy()

            # --- Mixed adjacency graph on supergroups (topology + spatial) ---
            # α=1: topology only; α=0: spatial only; 0<α<1: mixed.
            alpha = float(getattr(scene_settings, 'supergroup_mix_alpha', T.SUPERGROUP_MIX_ALPHA))
            alpha = max(0.0, min(1.0, alpha))

            # Topology edges (contracted from triangle mesh).
            # NOTE: we also use topo-edges to define "topology islands" for reject/promote.
            topo_edges = set()
            if alpha > 0.0 or scene_settings.supergroup_enable:
                for (a0, b0, c0) in triangles:
                    sa = int(target_sgid[int(a0)])
                    sb = int(target_sgid[int(b0)])
                    sc = int(target_sgid[int(c0)])
                    if sa != sb: topo_edges.add((sa, sb) if sa < sb else (sb, sa))
                    if sb != sc: topo_edges.add((sb, sc) if sb < sc else (sc, sb))
                    if sc != sa: topo_edges.add((sc, sa) if sc < sa else (sa, sc))

            # Optional island rules (reject/promote) based on pass ratio inside each TOPOLOGY island.
            if scene_settings.supergroup_enable and (scene_settings.supergroup_reject_weak_islands or scene_settings.supergroup_promote_strong_islands):
                # Build adjacency list on supergroups using topology edges only.
                adj_sg = [[] for _ in range(n_sg)]
                for (i, j) in topo_edges:
                    adj_sg[i].append(j)
                    adj_sg[j].append(i)
                comps = util.connected_components(adj_sg)
                min_ratio = float(scene_settings.supergroup_min_ratio)
                for comp in comps:
                    if not comp:
                        continue
                    idx = np.asarray(comp, dtype=np.int64)
                    ratio = float(gate[idx].mean())
                    if scene_settings.supergroup_reject_weak_islands and ratio < min_ratio:
                        matched_sg[idx] = False
                    if scene_settings.supergroup_promote_strong_islands and ratio >= min_ratio:
                        matched_sg[idx] = True

            # Spatial edges between representative points (no UI radius).
            spatial_edges = set()
            if alpha < 1.0 and n_sg > 1:
                cos_th = math.cos(math.radians(theta_merge_deg))
                kdt_sg = cKDTree(rep_pos)
                k = int(getattr(T, 'SUPERGROUP_SPATIAL_K', 16))
                k = max(1, min(k, n_sg - 1))
                # Query k nearest neighbors for each rep point.
                dists, idxs = kdt_sg.query(rep_pos, k=k + 1)
                if np.ndim(idxs) == 1:
                    idxs = idxs[:, None]
                for i in range(n_sg):
                    ni = rep_nrm[i]
                    for j in idxs[i, 1:]:
                        j = int(j)
                        if j <= i:
                            continue
                        nj = rep_nrm[j]
                        c = float(ni[0]*nj[0] + ni[1]*nj[1] + ni[2]*nj[2])
                        if c >= cos_th:
                            spatial_edges.add((i, j))

            # Build symmetric weighted adjacency W
            rows = []
            cols = []
            data = []

            def _add_edge(i, j, w):
                rows.append(i); cols.append(j); data.append(w)
                rows.append(j); cols.append(i); data.append(w)

            w_topo = float(alpha)
            w_spat = float(1.0 - alpha)
            if w_topo > 0.0:
                for (i, j) in topo_edges:
                    _add_edge(i, j, w_topo)
            if w_spat > 0.0:
                for (i, j) in spatial_edges:
                    _add_edge(i, j, w_spat)

            if len(rows) == 0:
                # Degenerate: no edges. Fall back to broadcasting representative weights directly.
                sg_weights = rep_weights.astype(np.float32, copy=False)
            else:
                W = sparse.coo_matrix((data, (rows, cols)), shape=(n_sg, n_sg), dtype=np.float64).tocsr()
                # Collapse duplicates by summation
                W.sum_duplicates()
                # Laplacian
                deg = np.asarray(W.sum(axis=1)).reshape(-1)
                L = sparse.diags(deg, 0, shape=(n_sg, n_sg), dtype=np.float64) - W

                # Inpaint (stable Laplacian with soft anchors instead of hard Dirichlet)
                lam = float(getattr(scene_settings, 'supergroup_anchor_lambda', 1000.0))
                m = matched_sg.astype(np.float64)
                M = sparse.diags(m, 0, shape=(n_sg, n_sg), dtype=np.float64)
                A = (L + lam * M).tocsc()
                # tiny diagonal jitter for numerical safety
                A = A + sparse.identity(n_sg, dtype=np.float64, format='csc') * eps_abs_min

                try:
                    lu = spla.splu(A)
                except Exception:
                    # fallback to generic solver
                    lu = None

                X0 = rep_weights.astype(np.float64, copy=False)
                B = (lam * (m[:, None] * X0))

                X = np.zeros_like(X0, dtype=np.float64)
                if lu is not None:
                    for k in range(X0.shape[1]):
                        X[:, k] = lu.solve(B[:, k])
                else:
                    for k in range(X0.shape[1]):
                        X[:, k] = spla.spsolve(A, B[:, k])

                # Smooth on the same mixed graph
                if scene_settings.smoothing_enable:
                    beta = float(scene_settings.smoothing_factor)
                    beta = max(0.0, min(1.0, beta))
                    iters = int(scene_settings.smoothing_repeat)
                    if iters > 0 and beta > 0.0:
                        # Row-normalized diffusion
                        inv_deg = 1.0 / np.maximum(deg, eps_abs_min)
                        Dinv = sparse.diags(inv_deg, 0, shape=(n_sg, n_sg), dtype=np.float64)
                        P = (Dinv @ W).tocsr()
                        for _ in range(iters):
                            X = (1.0 - beta) * X + beta * (P @ X)

                sg_weights = X.astype(np.float32, copy=False)

            # Broadcast supergroup weights back to vertices
            weights = sg_weights[target_sgid].astype(np.float32, copy=False)
            matched_verts = matched_sg[target_sgid]

            base_weights = weights.copy()
            adj_mat = util.get_mesh_adjacency_matrix_sparse(obj.data, include_self=True)
            if scene_settings.smoothing_enable and getattr(scene_settings, 'inpaint_enable', T.INPAINT_ENABLE):
                adj_list = util.get_mesh_adjacency_list(obj.data)
                weights = smooth_weigths(verts, weights, matched_verts, adj_mat, adj_list, scene_settings.smoothing_repeat, scene_settings.smoothing_factor, T.MAX_SMOOTH_SPATIAL_DISTANCE)
            
            # Inpaint mask semantics: blend only; does not affect matching/propagation.
            if getattr(scene_settings, 'inpaint_enable', T.INPAINT_ENABLE):
                eps = T.EPS_MASK_FOR_INPAINT
                if util.is_group_valid(obj.vertex_groups, object_settings.inpaint_group):
                    inpaint_mask = util.get_group_arr(obj, object_settings.inpaint_group).astype(np.float32, copy=False)
                    if object_settings.inpaint_group_invert:
                        inpaint_mask = 1.0 - inpaint_mask
                    inpaint_mask = np.clip(inpaint_mask, 0.0, 1.0)
                else:
                    inpaint_mask = np.ones(len(verts), dtype=np.float32)
                # Treat tiny paint-smear values as 0 to avoid mask contamination
                inpaint_mask[inpaint_mask < eps] = 0.0
                weights = base_weights * (1.0 - inpaint_mask[:, None]) + weights * (inpaint_mask[:, None])

            if scene_settings.enforce_four_bone_limit:
                weights[weights <= T.MIN_MATCH_CONFIDENCE] = 0
                mask = limit_mask(weights, adj_mat, limit_num=scene_settings.num_limit_groups)
                weights = (1 - mask) * weights
                weights[weights <= T.MIN_MATCH_CONFIDENCE] = 0
            
            weights_all.append(weights)

            # No per-object mapping cache is maintained. Selection helpers recompute
            # diagnostics on demand to avoid stale-cache bugs.
        ratio = float(getattr(scene_settings, 'new_weight_ratio', T.TRANSFER_NEW_WEIGHT_RATIO))
        ratio = max(0.0, min(1.0, ratio))
        if ratio <= 0.0:
            self.report({'INFO'}, 'New Weight Ratio is 0.0: discarded new results (no weights were written).')
            return {'FINISHED'}

        for obj, weights in zip(target_objs, weights_all):
            target_obj = obj
            source_vertex_groups = source_obj.vertex_groups
            weight_counts = np.count_nonzero(weights, axis=0)
            for group, w_count in zip(source_vertex_groups, weight_counts):
                if w_count > 0:
                    if group.name not in obj.vertex_groups:
                        obj.vertex_groups.new(name=group.name)
            
            is_deform = [util.is_vertex_group_deform_bone(source_obj, g.name) for g in source_vertex_groups]
            
            # Selection scope replaces transfer mask:
            # - Edit Mode: operate only on selected vertices (if none selected -> all vertices)
            # - Object Mode: operate on all vertices
            scope_indices = None
            if obj.mode == 'EDIT':
                scope_indices = util.get_selected_vert_indices(obj)
                if len(scope_indices) == 0:
                    self.report({'WARNING'}, 'Edit Mode requires at least one selected vertex')
                    return {'CANCELLED'}

            for i, w in enumerate(weights.T):
                w_count = weight_counts[i]
                if w_count == 0: continue
                
                source_group = source_vertex_groups[i]
                target_group = obj.vertex_groups[source_group.name]
                
                if target_group.lock_weight:
                    continue
                if deform_only and not is_deform[i]:
                    continue
                # Write weights (BMesh in Edit Mode, vertex group ops otherwise)
                # Safety: never write back to a non-target object
                assert obj is target_obj
                
                # Blend with existing target weights (ratio = NEW contribution; 1.0 = overwrite; 0.0 = keep existing)
                w_to_write = w
                if obj is not source_obj:
                    try:
                        old_w = util.get_group_arr(obj, target_group.name)
                    except Exception:
                        old_w = None
                    if old_w is not None and old_w.shape[0] == w.shape[0]:
                        w_to_write = ratio * w + (1.0 - ratio) * old_w

                util.set_group_weights(obj, target_group.name, w_to_write, threshold=T.MIN_MATCH_CONFIDENCE, indices=scope_indices)   
        if scene_settings.apply_to_selected:
            self.report({'INFO'}, f'Weights transfered from {source_obj.name} to selected objects')
        else:
            self.report({'INFO'}, f'Weights transfered from {source_obj.name} to {context.object.name}')
        return {'FINISHED'}
    

class SelectRejectedByDistance(bpy.types.Operator):
    """Select vertices that fail the distance gate (cached)"""
    bl_idname = "object.rwt_select_rejected_by_distance"
    bl_label = "Rejected (Distance)"
    bl_description = "Select/deselect vertices that did not find a closest-surface match within Max Distance (recomputes mapping once; does not write weights)"
    bl_options = {'REGISTER', 'UNDO'}

    deselect: bpy.props.BoolProperty(name="Deselect", default=False)

    @classmethod
    def poll(cls, context):
        if context.mode != 'EDIT_MESH':
            return False

        if missing_deps:
            return False
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            return False
        settings: SceneSettingsGroup = context.scene.robust_weight_transfer_settings
        if not settings.source_object or settings.source_object == obj:
            return False
        if settings.use_deformed_target and util.has_modifier(obj, *util.TOPOLOGY_MODS):
            return False
        return True

    def execute(self, context):
        settings: SceneSettingsGroup = context.scene.robust_weight_transfer_settings
        obj = context.active_object
        object_settings: ObjectSettingsGroup = obj.robust_weight_transfer_settings

        # Lazy import: avoid NameError when libigl is available but not imported at module scope.
        try:
            igl = importlib.import_module("igl")
        except Exception as e:
            self.report({'ERROR'}, f"libigl (igl) is not importable: {e}")
            return {'CANCELLED'}


        depsgraph = context.evaluated_depsgraph_get()
        src = settings.source_object

        source_obj = src.evaluated_get(depsgraph) if settings.use_deformed_source else src
        target_obj = obj.evaluated_get(depsgraph) if settings.use_deformed_target else obj

        source_verts, source_triangles, source_normals = util.get_obj_arrs_world(source_obj)
        verts, triangles, normals = util.get_obj_arrs_world(target_obj)

        deform_only = settings.group_selection == 'DEFORM_POSE_BONES'
        is_deform = [g.name in src.vertex_groups for g in src.vertex_groups]  # placeholder; kept for compat
        source_weights = util.get_groups_arr(source_obj, is_deform if deform_only else None)

        eps_force_ref, _eps_norm_ref, face_h_eps = _compute_band_refs_for_ui(context, settings)

        strong_clean = bool(getattr(settings, 'strong_clean_enable', True))
        effective_clean_faces = bool(strong_clean and getattr(settings, 'clean_degenerate_faces', False))

        matched, _w2, sqrD, deg_angles, pass_dist, pass_norm, _face_I = find_matches_closest_surface(
            source_verts,
            source_triangles,
            source_normals,
            verts,
            normals,
            source_weights,
            settings.max_distance ** 2,
            math.degrees(settings.max_normal_angle_difference),
            settings.flip_vertex_normal,
            clean_degenerate_faces=effective_clean_faces,
            face_height_eps=(face_h_eps if effective_clean_faces else None),
            return_diagnostics=True,
        )

        # Island-level semantics (former "Rejected Loose Parts"): select entire connected
        # components that have ZERO vertices passing the distance gate.
        num_conn, conn, _num_vertices = igl.connected_components(igl.adjacency_matrix(triangles))
        conns = [np.where(conn == i)[0] for i in range(num_conn)]
        pass_per_island = [int(np.count_nonzero(pass_dist[c])) for c in conns]
        zero_pass_islands = [i for i, c in enumerate(pass_per_island) if c == 0]

        rejected_by_distance = np.zeros(verts.shape[0], dtype=bool)
        for i in zero_pass_islands:
            rejected_by_distance[conns[i]] = True

        util.select_mask_vertices(obj, rejected_by_distance, action=('deselect' if self.deselect else 'select'))
        self.report({'INFO'}, f"{'Deselected' if self.deselect else 'Selected'} {int(np.count_nonzero(rejected_by_distance))} vertices (distance: zero-pass islands).")
        return {'FINISHED'}


class SelectRejectedByNormal(bpy.types.Operator):
    """Select vertices that pass distance but fail normal gate (cached)"""
    bl_idname = "object.rwt_select_rejected_by_normal"
    bl_label = "Rejected (Normal)"
    bl_description = "Select/deselect vertices that passed Max Distance but were rejected by Max Normal Difference (recomputes mapping once; does not write weights)"
    bl_options = {'REGISTER', 'UNDO'}

    deselect: bpy.props.BoolProperty(name="Deselect", default=False)

    @classmethod
    def poll(cls, context):
        if context.mode != 'EDIT_MESH':
            return False

        if missing_deps:
            return False
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            return False
        settings: SceneSettingsGroup = context.scene.robust_weight_transfer_settings
        if not settings.source_object or settings.source_object == obj:
            return False
        if settings.use_deformed_target and util.has_modifier(obj, *util.TOPOLOGY_MODS):
            return False
        return True

    def execute(self, context):
        settings: SceneSettingsGroup = context.scene.robust_weight_transfer_settings
        obj = context.active_object
        object_settings: ObjectSettingsGroup = obj.robust_weight_transfer_settings

        # Lazy import: avoid NameError when libigl is available but not imported at module scope.
        try:
            igl = importlib.import_module("igl")
        except Exception as e:
            self.report({'ERROR'}, f"libigl (igl) is not importable: {e}")
            return {'CANCELLED'}


        depsgraph = context.evaluated_depsgraph_get()
        src = settings.source_object

        source_obj = src.evaluated_get(depsgraph) if settings.use_deformed_source else src
        target_obj = obj.evaluated_get(depsgraph) if settings.use_deformed_target else obj

        source_verts, source_triangles, source_normals = util.get_obj_arrs_world(source_obj)
        verts, triangles, normals = util.get_obj_arrs_world(target_obj)

        deform_only = settings.group_selection == 'DEFORM_POSE_BONES'
        is_deform = [g.name in src.vertex_groups for g in src.vertex_groups]  # placeholder; kept for compat
        source_weights = util.get_groups_arr(source_obj, is_deform if deform_only else None)

        eps_force_ref, _eps_norm_ref, face_h_eps = _compute_band_refs_for_ui(context, settings)

        strong_clean = bool(getattr(settings, 'strong_clean_enable', True))
        effective_clean_faces = bool(strong_clean and getattr(settings, 'clean_degenerate_faces', False))

        matched, _w2, sqrD, deg_angles, pass_dist, pass_norm, _face_I = find_matches_closest_surface(
            source_verts,
            source_triangles,
            source_normals,
            verts,
            normals,
            source_weights,
            settings.max_distance ** 2,
            math.degrees(settings.max_normal_angle_difference),
            settings.flip_vertex_normal,
            clean_degenerate_faces=effective_clean_faces,
            face_height_eps=(face_h_eps if effective_clean_faces else None),
            return_diagnostics=True,
        )

        rejected = np.logical_and(pass_dist, np.logical_not(pass_norm))

        # UI spec: "Rejected by Normal" selects verts that passed distance but failed
        # normal, excluding successful matches PLUS one topological edge ring around them.
        success_mask = matched.astype(bool)
        expanded = util.expand_mask_rings(obj.data, success_mask, 1)
        rejected = np.logical_and(rejected, np.logical_not(expanded))

        util.select_mask_vertices(obj, rejected, action=('deselect' if self.deselect else 'select'))
        self.report({'INFO'}, f"{'Deselected' if self.deselect else 'Selected'} {int(np.count_nonzero(rejected))} vertices (normal gate).")
        return {'FINISHED'}

    
class Inpaint(bpy.types.Operator):
    """Inpaint"""
    bl_idname = "object.rwt_inpaint"
    bl_label = "Inpaint"
    bl_description = "Inpaint active object using the inpaint mask"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        # Only available in Mesh Object/Edit contexts (disabled for Armature Edit/Pose)
        if context.mode not in {'OBJECT', 'EDIT_MESH'}:
            return False

        if missing_deps:
            return False
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            return False

        # When an object is deleted or the context is in a transient state,
        # `active_object` can be None; also a non-mesh can be active.
        # Guard before accessing custom properties.
        if not hasattr(obj, "robust_weight_transfer_settings"):
            return False

        scene_settings: SceneSettingsGroup = context.scene.robust_weight_transfer_settings
        object_settings: ObjectSettingsGroup = obj.robust_weight_transfer_settings

        # Lazy import: avoid NameError when libigl is available but not imported at module scope.
        try:
            igl = importlib.import_module("igl")
        except Exception as e:
            self.report({'ERROR'}, f"libigl (igl) is not importable: {e}")
            return {'CANCELLED'}


        if len(object_settings.inpaint_group) == 0 or object_settings.inpaint_group not in obj.vertex_groups:
            return False

        if scene_settings.use_deformed_target and util.has_modifier(obj, *util.TOPOLOGY_MODS):
            return False
        return True

    def execute(self, context):
        scene_settings: SceneSettingsGroup = context.scene.robust_weight_transfer_settings
        obj = context.active_object
        object_settings: ObjectSettingsGroup = obj.robust_weight_transfer_settings

        # Lazy import: avoid NameError when libigl is available but not imported at module scope.
        try:
            igl = importlib.import_module("igl")
        except Exception as e:
            self.report({'ERROR'}, f"libigl (igl) is not importable: {e}")
            return {'CANCELLED'}

        

        scope_indices = None
        if obj.mode == 'EDIT':
            scope_indices = util.get_selected_vert_indices(obj)
            if len(scope_indices) == 0:
                self.report({'WARNING'}, 'Edit Mode requires at least one selected vertex')
                return {'CANCELLED'}
        depsgraph = context.evaluated_depsgraph_get()
        verts, triangles, normals = util.get_obj_arrs_world(obj.evaluated_get(depsgraph) if scene_settings.use_deformed_target else obj)
        is_deform = [util.is_vertex_group_deform_bone(obj, g.name) for g in obj.vertex_groups]
        weights = util.get_groups_arr(obj, is_deform)

        # Inpaint mask semantics:
        # - If no mask group is selected, treat as mask=1 everywhere (full inpaint influence).
        # - If a mask group is selected, blend per-vertex: final=(1-mask)*base + mask*inpainted.
        # - Values < EPS_MASK_FOR_INPAINT are treated as 0 to ignore tiny paint-smear noise.
        eps = T.EPS_MASK_FOR_INPAINT
        if util.is_group_valid(obj.vertex_groups, object_settings.inpaint_group):
            inpaint_mask = util.get_group_arr(obj, object_settings.inpaint_group).astype(np.float32, copy=False)
            if object_settings.inpaint_group_invert:
                inpaint_mask = 1.0 - inpaint_mask
            inpaint_mask = np.clip(inpaint_mask, 0.0, 1.0)
            inpaint_mask[inpaint_mask < eps] = 0.0
        else:
            inpaint_mask = np.ones(len(verts), dtype=np.float32)
        
        if np.max(inpaint_mask) < eps:
            self.report({'INFO'}, 'Inpaint mask is 0 everywhere; nothing to apply.')
            return {'FINISHED'}
        
        base_weights = weights.copy()
        # Fixed constraints are vertices that already have some weights (independent of mask).
        matched_verts = np.any(np.abs(base_weights) > 1e-12, axis=1)
        if not np.any(matched_verts):
            self.report({'ERROR'}, 'No existing weights found to serve as inpaint constraints.')
            return {'CANCELLED'}
        
        result, inpainted = inpaint(verts, triangles, base_weights, matched_verts, scene_settings.inpaint_mode == 'POINT')
        weights = base_weights * (1.0 - inpaint_mask[:, None]) + inpainted * (inpaint_mask[:, None])
        if not result:
            self.report({'ERROR'}, f'Failed weight inpainting on {obj.name}: This usually happens on disconnected/loose parts that do not find any match on the source mesh. Use the Vertex Mapping > Rejected (Distance) selection tool to locate them.')
            return {'CANCELLED'}

        for i, w in enumerate(weights.T):
            group = obj.vertex_groups[i]
            if group.lock_weight: continue
            
            if not is_deform[i]: continue
            
            w_count = np.count_nonzero(w)
            if w_count == 0: continue

            util.set_group_weights(obj, group.name, w, threshold=T.MIN_MATCH_CONFIDENCE, indices=scope_indices)  
        self.report({'INFO'}, f'Weights inpainted.')
        return {'FINISHED'}
        


class CalcMidProbRegion(bpy.types.Operator):
    bl_idname = "object.rwt_calc_mid_prob_region"
    bl_label = "Calculate prob region (MID)"
    bl_description = "Compute and show MID band reference information for the current target mesh"
    bl_options = {'REGISTER'}

    def execute(self, context):
        settings = context.scene.robust_weight_transfer_settings
        obj = _ui_pick_target_mesh_for_calc(context, settings)
        info = _compute_supergroup_ui_info(context, settings)
        if not info or obj is None:
            settings.ui_mid_valid = False
            self.report({'WARNING'}, "No valid target mesh for MID prob-region calculation.")
            return {'CANCELLED'}

        token = _ui_make_prob_region_token('mid', obj, settings)
        settings.ui_mid_token = token
        settings.ui_mid_valid = True

        settings.ui_mid_raw = float(info.get('eps_norm_raw') or 0.0)
        settings.ui_mid_effective = float(info.get('eps_norm') or 0.0)

        rm = info.get('range_mid')
        if rm is not None:
            settings.ui_mid_allowed_min = float(rm.min_allowed)
            settings.ui_mid_allowed_max = float(rm.max_allowed)
            settings.ui_mid_note = str(rm.note or "")
            if rm.clamp_estimate_min is not None and rm.clamp_estimate_max is not None:
                settings.ui_mid_has_clamp = True
                settings.ui_mid_clamp_p05 = float(rm.clamp_estimate_min)
                settings.ui_mid_clamp_p95 = float(rm.clamp_estimate_max)
            else:
                settings.ui_mid_has_clamp = False
        else:
            settings.ui_mid_allowed_min = 0.0
            settings.ui_mid_allowed_max = 0.0
            settings.ui_mid_has_clamp = False
            settings.ui_mid_note = ""

        # Also cache REL clamp context if present
        d_nn = float(info.get('d_nn_median') or 0.0)
        settings.ui_rel_dnn_median = d_nn
        settings.ui_rel_eps_min = float(info.get('eps_min') or 0.0)
        eps_max = info.get('eps_max')
        settings.ui_rel_eps_max = float(eps_max) if isinstance(eps_max, (float, int)) and math.isfinite(float(eps_max)) else 0.0
        settings.ui_rel_valid = True

        return {'FINISHED'}


class CalcMinProbRegion(bpy.types.Operator):
    bl_idname = "object.rwt_calc_min_prob_region"
    bl_label = "Calculate prob region (MIN)"
    bl_description = "Compute and show MIN band reference information for the current target mesh"
    bl_options = {'REGISTER'}

    def execute(self, context):
        settings = context.scene.robust_weight_transfer_settings
        obj = _ui_pick_target_mesh_for_calc(context, settings)
        info = _compute_supergroup_ui_info(context, settings)
        if not info or obj is None:
            settings.ui_min_valid = False
            self.report({'WARNING'}, "No valid target mesh for MIN prob-region calculation.")
            return {'CANCELLED'}

        token = _ui_make_prob_region_token('min', obj, settings)
        settings.ui_min_token = token
        settings.ui_min_valid = True

        settings.ui_min_raw = float(info.get('eps_force_raw') or 0.0)
        settings.ui_min_effective = float(info.get('eps_force') or 0.0)

        rm = info.get('range_min')
        if rm is not None:
            settings.ui_min_allowed_min = float(rm.min_allowed)
            settings.ui_min_allowed_max = float(rm.max_allowed)
            settings.ui_min_note = str(rm.note or "")
            if rm.clamp_estimate_min is not None and rm.clamp_estimate_max is not None:
                settings.ui_min_has_clamp = True
                settings.ui_min_clamp_p05 = float(rm.clamp_estimate_min)
                settings.ui_min_clamp_p95 = float(rm.clamp_estimate_max)
            else:
                settings.ui_min_has_clamp = False
        else:
            settings.ui_min_allowed_min = 0.0
            settings.ui_min_allowed_max = 0.0
            settings.ui_min_has_clamp = False
            settings.ui_min_note = ""

        # REL clamp cache
        d_nn = float(info.get('d_nn_median') or 0.0)
        settings.ui_rel_dnn_median = d_nn
        settings.ui_rel_eps_min = float(info.get('eps_min') or 0.0)
        eps_max = info.get('eps_max')
        settings.ui_rel_eps_max = float(eps_max) if isinstance(eps_max, (float, int)) and math.isfinite(float(eps_max)) else 0.0
        settings.ui_rel_valid = True

        return {'FINISHED'}


class CalcFaceProbRegion(bpy.types.Operator):
    bl_idname = "object.rwt_calc_face_prob_region"
    bl_label = "Calculate prob region (Face Cleaning)"
    bl_description = "Compute and show Face Cleaning reference information for the current target mesh"
    bl_options = {'REGISTER'}

    def execute(self, context):
        settings = context.scene.robust_weight_transfer_settings
        obj = _ui_pick_target_mesh_for_calc(context, settings)
        info = _compute_supergroup_ui_info(context, settings)
        if not info or obj is None:
            settings.ui_face_valid = False
            self.report({'WARNING'}, "No valid target mesh for Face prob-region calculation.")
            return {'CANCELLED'}

        token = _ui_make_prob_region_token('face', obj, settings)
        settings.ui_face_token = token
        settings.ui_face_valid = True

        settings.ui_face_raw = float(info.get('face_raw') or 0.0)
        settings.ui_face_ref = float(info.get('face_h_eps') or 0.0)

        rm = info.get('range_face')
        if rm is not None:
            settings.ui_face_allowed_min = float(rm.min_allowed)
            settings.ui_face_allowed_max = float(rm.max_allowed)
            settings.ui_face_note = str(rm.note or "")
            if rm.clamp_estimate_min is not None and rm.clamp_estimate_max is not None:
                settings.ui_face_has_clamp = True
                settings.ui_face_clamp_p05 = float(rm.clamp_estimate_min)
                settings.ui_face_clamp_p95 = float(rm.clamp_estimate_max)
            else:
                settings.ui_face_has_clamp = False
        else:
            settings.ui_face_allowed_min = 0.0
            settings.ui_face_allowed_max = 0.0
            settings.ui_face_has_clamp = False
            settings.ui_face_note = ""

        return {'FINISHED'}


class ObjectSettingsGroup(bpy.types.PropertyGroup):
    vertex_group: bpy.props.StringProperty(name='Mask Vertex Group')
    vertex_group_invert: bpy.props.BoolProperty(name='Invert')
    inpaint_group: bpy.props.StringProperty(name='Inpaint Vertex Group')
    inpaint_group_invert: bpy.props.BoolProperty(name='Invert Inpaint')
    
def update_enforce_four_bone_limit(self, context):
    """Ensure the correct group selection and enforce constraints."""
    if self.enforce_four_bone_limit:
        self.group_selection = 'DEFORM_POSE_BONES'
    
class SceneSettingsGroup(bpy.types.PropertyGroup):
    source_object: bpy.props.PointerProperty(name='Source', type=bpy.types.Object, poll=lambda self, obj: obj.type == 'MESH')
    shape_key_mix: bpy.props.BoolProperty(name='Use Shape Key Mix', description='Uses the Shape of the Shape Key Mix to transfer the weights', default=True)
    max_distance: bpy.props.FloatProperty(
        name='Max Distance',
        description='Maximum allowed distance between source and destination vertex',
        default=T.MAX_MATCH_DISTANCE,
        min=0,
        unit='LENGTH',
        subtype='DISTANCE')
    max_normal_angle_difference: bpy.props.FloatProperty(
        name='Max Normal Difference',
        description='Maximum allowed vertex normal difference between source and destination vertex',
        default=math.radians(T.MAX_NORMAL_ANGLE_DEG),
        min=0,
        max=math.pi,
        precision=3,
        step=100,
        unit='ROTATION',
        subtype='ANGLE')

    normal_angle_offset_deg: bpy.props.FloatProperty(
        name='Normal Angle Offset (deg)',
        description='Offset added to the max-normal-angle threshold (in degrees). Positive = more permissive.',
        default=T.NORMAL_ANGLE_OFFSET_DEG,
        soft_min=-30.0,
        soft_max=30.0,
        step=10,
        precision=2,
        unit='NONE')

    flip_vertex_normal: bpy.props.BoolProperty(
        name='Flip Vertex Normal',
        description='Allow vertex normal flipped at 180° between source and destination vertex',
        default=True)


    # --- Supergroup matching (connected-component rule) ---

    supergroup_master_enable: bpy.props.BoolProperty(
    name="Enable Supergroups",
    description="Master switch for the supergroup (representative-point) pipeline. If disabled, each vertex acts as its own supergroup (debug/fallback).",
    default=True)

    supergroup_enable: bpy.props.BoolProperty(
    name="Enable Island Rules",
    description="Optionally promote/reject whole topology islands based on the supergroup match pass ratio.",
    default=False)

    supergroup_reject_weak_islands: bpy.props.BoolProperty(
    name="Reject Weak Islands",
    description="If a topology island's pass ratio is below the threshold, treat the whole island as UNMATCHED (it will be inpainted from neighbors).",
    default=True)

    supergroup_promote_strong_islands: bpy.props.BoolProperty(
    name="Promote Strong Islands",
    description="If a topology island's pass ratio meets the threshold, treat the whole island as MATCHED (it will be anchored).",
    default=False)

    supergroup_min_ratio: bpy.props.FloatProperty(
    name="Min Pass Ratio",
    description="Minimum fraction of supergroups in a topology island that must pass the (distance+normal) match gate.",
    default=T.SUPERGROUP_REJECT_RATIO,
    min=0.0,
    max=1.0,
    step=10)

# --- Supergroup propagation graph (topology + spatial) ---
    supergroup_mix_alpha: bpy.props.FloatProperty(
    name="Topo↔Spatial Mix (α)",
    description="α=1: topology only. α=0: spatial only. 0<α<1: mixed adjacency (W = α·W_topo + (1-α)·W_spatial).",
    default=T.SUPERGROUP_MIX_ALPHA,
    min=0.0,
    max=1.0,
    step=10,
    precision=3)

    supergroup_theta_merge_deg: bpy.props.FloatProperty(
    name="MID Merge Angle θ (deg)",
    description="Normal-angle threshold (degrees) for the MID merge band (eps_norm): dist<=eps_norm AND angle<=θ.",
    default=60.0,
    min=0.0,
    max=180.0,
    step=10,
    precision=1)

    # Strong cleaning gate (default ON): controls whether advanced cleaning features
    # (rel clamp, strong point clean, face-clean prefilter) are active and visible.
    strong_clean_enable: bpy.props.BoolProperty(
    name="Strong Clean",
    description="Enable advanced/strong cleaning for supergroup matching (rel clamp + optional face pre-clean). Turn OFF to use weak cleaning only.",
    default=True)
    supergroup_eps_min: bpy.props.FloatProperty(
        name="minscale",
        description="MIN band threshold in world units (direct eps). This is the force-merge distance threshold before REL clamp and MID/k hierarchy.",
        default=T.SUPERGROUP_EPS_MINSCALE_DEFAULT,
        min=0.0,
        max=1.0e6,
        soft_min=0.0,
        soft_max=1.0,
        step=1,
        precision=6)
    supergroup_eps_mid: bpy.props.FloatProperty(
        name="midscale",
        description="MID band threshold in world units (direct eps). This is the distance threshold for distance+angle merge before REL clamp.",
        default=T.SUPERGROUP_EPS_MIDSCALE_DEFAULT,
        min=0.0,
        max=1.0e6,
        soft_min=0.0,
        soft_max=10.0,
        step=1,
        precision=6)

    # Relative clamp switch: ON by default.
    supergroup_rel_clamp_enable: bpy.props.BoolProperty(
    name="Enable Rel Clamp (Advanced)",
    description="Clamp eps to a mesh-density window using d_nn: eps_min=max(eps_abs_min, d_nn*REL_MIN), eps_max=d_nn*REL_MAX.",
    default=True)

    # UI-only: keep REL_MIN/REL_MAX hidden unless the user explicitly expands them.
    supergroup_rel_clamp_show: bpy.props.BoolProperty(
    name="修改 clamp",
    description="Show REL_MIN/REL_MAX controls.",
    default=False)

    # Degenerate-face cleaning for surface projection (optional; OFF by default).
    clean_degenerate_faces: bpy.props.BoolProperty(
    name="Clean Degenerate Faces",
    description="(Strong Clean only) Pre-filter near-degenerate source triangles (low height) before barycentric projection to reduce NaN/Inf barycentrics on broken topology.",
    default=False)
    face_clean_height: bpy.props.FloatProperty(
        name="face clean",
        description="(Strong Clean only) Face-clean height threshold in world units (direct eps). This is the raw height threshold before REL clamp.",
        default=T.FACE_CLEAN_HEIGHT_DEFAULT,
        min=0.0,
        max=1.0e6,
        soft_min=0.0,
        soft_max=1.0,
        step=1,
        precision=6)

    supergroup_eps_rel_min: bpy.props.FloatProperty(
    name="REL_MIN (× d_nn)",
    description="Lower clamp for eps in multiples of d_nn: eps_min = max(1e-12, REL_MIN * d_nn).",
    default=T.SUPERGROUP_EPS_REL_MIN,
    min=0.0,
    max=1.0,
    soft_min=0.0,
    soft_max=1.0,
    step=1,
    precision=4)

    supergroup_eps_rel_max: bpy.props.IntProperty(
    name="REL_MAX (× d_nn)",
    description="Upper clamp for eps in multiples of d_nn: eps_max = REL_MAX * d_nn.",
    default=int(T.SUPERGROUP_EPS_REL_MAX),
    min=1,
    soft_min=1,
    soft_max=64)

    supergroup_anchor_lambda: bpy.props.FloatProperty(
    name="Inpaint Anchor λ",
    description="Soft-anchor strength for matched supergroups during Laplacian inpaint (higher = closer to matched values).",
    default=1000.0,
    min=0.0,
    step=100,
    precision=1)

    # --- UI: Prob region calculations (press "Calculate prob region") ---
    ui_rel_valid: bpy.props.BoolProperty(name="UI REL valid", default=False, options={'SKIP_SAVE'})
    ui_rel_dnn_median: bpy.props.FloatProperty(name="UI d_nn median", default=0.0, options={'SKIP_SAVE'})
    ui_rel_eps_min: bpy.props.FloatProperty(name="UI eps_min", default=0.0, options={'SKIP_SAVE'})
    ui_rel_eps_max: bpy.props.FloatProperty(name="UI eps_max", default=0.0, options={'SKIP_SAVE'})

    ui_mid_valid: bpy.props.BoolProperty(name="UI MID valid", default=False, options={'SKIP_SAVE'})
    ui_mid_token: bpy.props.StringProperty(name="UI MID token", default="", options={'SKIP_SAVE'})
    ui_mid_raw: bpy.props.FloatProperty(name="UI MID raw", default=0.0, options={'SKIP_SAVE'})
    ui_mid_effective: bpy.props.FloatProperty(name="UI MID effective", default=0.0, options={'SKIP_SAVE'})
    ui_mid_allowed_min: bpy.props.FloatProperty(name="UI MID allowed min", default=0.0, options={'SKIP_SAVE'})
    ui_mid_allowed_max: bpy.props.FloatProperty(name="UI MID allowed max", default=0.0, options={'SKIP_SAVE'})
    ui_mid_has_clamp: bpy.props.BoolProperty(name="UI MID has clamp", default=False, options={'SKIP_SAVE'})
    ui_mid_clamp_p05: bpy.props.FloatProperty(name="UI MID clamp p05", default=0.0, options={'SKIP_SAVE'})
    ui_mid_clamp_p95: bpy.props.FloatProperty(name="UI MID clamp p95", default=0.0, options={'SKIP_SAVE'})
    ui_mid_note: bpy.props.StringProperty(name="UI MID note", default="", options={'SKIP_SAVE'})

    ui_min_valid: bpy.props.BoolProperty(name="UI MIN valid", default=False, options={'SKIP_SAVE'})
    ui_min_token: bpy.props.StringProperty(name="UI MIN token", default="", options={'SKIP_SAVE'})
    ui_min_raw: bpy.props.FloatProperty(name="UI MIN raw", default=0.0, options={'SKIP_SAVE'})
    ui_min_effective: bpy.props.FloatProperty(name="UI MIN effective", default=0.0, options={'SKIP_SAVE'})
    ui_min_allowed_min: bpy.props.FloatProperty(name="UI MIN allowed min", default=0.0, options={'SKIP_SAVE'})
    ui_min_allowed_max: bpy.props.FloatProperty(name="UI MIN allowed max", default=0.0, options={'SKIP_SAVE'})
    ui_min_has_clamp: bpy.props.BoolProperty(name="UI MIN has clamp", default=False, options={'SKIP_SAVE'})
    ui_min_clamp_p05: bpy.props.FloatProperty(name="UI MIN clamp p05", default=0.0, options={'SKIP_SAVE'})
    ui_min_clamp_p95: bpy.props.FloatProperty(name="UI MIN clamp p95", default=0.0, options={'SKIP_SAVE'})
    ui_min_note: bpy.props.StringProperty(name="UI MIN note", default="", options={'SKIP_SAVE'})

    ui_face_valid: bpy.props.BoolProperty(name="UI FACE valid", default=False, options={'SKIP_SAVE'})
    ui_face_token: bpy.props.StringProperty(name="UI FACE token", default="", options={'SKIP_SAVE'})
    ui_face_raw: bpy.props.FloatProperty(name="UI FACE raw", default=0.0, options={'SKIP_SAVE'})
    ui_face_ref: bpy.props.FloatProperty(name="UI FACE ref", default=0.0, options={'SKIP_SAVE'})
    ui_face_allowed_min: bpy.props.FloatProperty(name="UI FACE allowed min", default=0.0, options={'SKIP_SAVE'})
    ui_face_allowed_max: bpy.props.FloatProperty(name="UI FACE allowed max", default=0.0, options={'SKIP_SAVE'})
    ui_face_has_clamp: bpy.props.BoolProperty(name="UI FACE has clamp", default=False, options={'SKIP_SAVE'})
    ui_face_clamp_p05: bpy.props.FloatProperty(name="UI FACE clamp p05", default=0.0, options={'SKIP_SAVE'})
    ui_face_clamp_p95: bpy.props.FloatProperty(name="UI FACE clamp p95", default=0.0, options={'SKIP_SAVE'})
    ui_face_note: bpy.props.StringProperty(name="UI FACE note", default="", options={'SKIP_SAVE'})

    smoothing_factor: bpy.props.FloatProperty(
        name='Smoothing factor',
        description='Smoothing factor used in the smoothing pass.',
        default=T.SMOOTHING_STRENGTH,
        min=0,
        max=1,
        step=10)
    smoothing_repeat: bpy.props.IntProperty(
        name='Smoothing repeat',
        description='Amount of iterations of smoothing used in the smoothing pass',
        default=T.SMOOTHING_ITERATIONS,
        min=0)
    apply_to_selected: bpy.props.BoolProperty(
        name='Apply to all Selected Objects',
        description='Weight transfers the from the source object to all selected objects')
    use_modifier: bpy.props.BoolProperty(name='Use Modifier', description='Uses the Shape resulting from the source objects modifier stack', default=True)
    use_deformed_source: bpy.props.BoolProperty(name='Use Deformed Source', description='Uses the Shape resulting from the source object\'s modifier stack and shape keys', default=True)
    use_deformed_target: bpy.props.BoolProperty(name='Use Deformed Target', description='Uses the Shape resulting from the target object\'s modifier stack and shape keys', default=True)
    enforce_four_bone_limit: bpy.props.BoolProperty(
        name='Limit Groups per Vertex',
        description='Limit a vertex to being influenced to a specific amount of groups. This is useful when a mesh will be exported to game engines like Unity, that normally only support 4 bones per vertex',
        default=True,
        update=update_enforce_four_bone_limit)
    group_selection: bpy.props.EnumProperty(
        name='Group Type',
        description='Select what subset of Vertex Group\'s should be transferred',
        items=[
            ('ALL_GROUPS', 'All Groups', 'Transfer all groups'),
            ('DEFORM_POSE_BONES', 'Deform Pose Bones', 'Only transfer deform pose bones, used by the Armature')
        ],
        default='DEFORM_POSE_BONES')
    new_weight_ratio: bpy.props.FloatProperty(
        name='New Weight Ratio',
        description='Blend ratio between newly computed weights and existing target weights. 1.0 = write new weights only; 0.0 = keep existing weights (discard new).',
        default=T.TRANSFER_NEW_WEIGHT_RATIO,
        min=0.0,
        max=1.0)

    dilation_repeat: bpy.props.IntProperty(
        name='Dilation repeat',
        description='Amount of iterations used to smooth the weight remove mask, that is used to limit the bone influence per vertex to 4',
        default=4,
        min=0)
    inpaint_enable: bpy.props.BoolProperty(

        name='Enable Inpaint',

        description='Run inpaint stage (unmatched vertices are filled) before optional smoothing',

        default=T.INPAINT_ENABLE)


    inpaint_mode: bpy.props.EnumProperty(
        name='Mode',
        description='Choose the Inpaint Mode',
        items=[
            ('POINT', 'Point', 'Object is remeshed internally. Weights can "flow" outside a mesh/loose part and more robust' ),
            ('SURFACE', 'Surface', 'Mesh is used as is. Weights "flow" only inside a mesh/loose part. More likely to fail compared to "Point"')
        ],
        default='POINT')
    smoothing_enable: bpy.props.BoolProperty(
        name='Enable Smoothing',
        description='Smooths weights in the area where weights got inpainted',
        default=False)
    smooth_limit_debug: bpy.props.BoolProperty(
        name='Limited vertices to Vertex Group',
        description='Visualize the vertices that got limited by writing to the "Limited" vertex group',
        default=False)
    num_limit_groups: bpy.props.IntProperty(
        name="Max groups per vertex",
        description="Amount of groups a vertex should be limited to. For VRChat/Unity keep it at 4.",
        min=1,
        default=4
    )


class RobustWeightTransferPanel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Robust Weight Transfer"
    bl_idname = "OBJECT_PT_robust_weight_transfer_panel"
    bl_space_type = 'VIEW_3D'   # Defines the space type where the panel is located
    bl_region_type = 'UI'       # Specifies that the panel is drawn in the UI region
    bl_category = 'SENT'      # The name of the tab the panel will be in
    bl_options = set()

    def draw(self, context): 
        layout = self.layout

        refresh_dependency_state(load_modules=False)

        if installed_deps and not _ALGO_LOADED:
            _load_algo_modules()

        if missing_deps:
            box = layout.box()
            col = box.column()
            if installed_deps:
                col.label(text="Dependencies installed!", icon='INFO')
                col.label(text="Restart Blender!", icon='ERROR')
                return
            
            col.label(text="Blender will be unreactive while installing")
            col.operator("wm.install_rwt_dependencies", icon='IMPORT')
            col.label(text="This might take a few minutes", icon='INFO')
            return

        active_obj = context.object
        if not context.object:
            layout.label(text='No active object selected.')
            return
        
        props = active_obj.robust_weight_transfer_settings
        settings = context.scene.robust_weight_transfer_settings
        
        # Object field for source
        row = layout.row(align=True)
        row.prop(settings, "source_object")
        row.prop(settings, "use_deformed_source",toggle=True, text="", icon='SHAPEKEY_DATA')
        row.prop(settings, "use_deformed_source",toggle=True, text="", icon='MODIFIER')
        
        # Selection scope (replaces transfer mask)
        box = layout.box()
        col = box.column(align=True)
        col.label(text='Selection Scope')
        col.label(text='Edit Mode: only selected vertices are modified (must select >= 1)')
        col.label(text='Object Mode: all vertices are modified')
        
        objs = lambda x: [obj for obj in x if obj != settings.source_object and isinstance(obj.data, bpy.types.Mesh)]
        if settings.apply_to_selected:
            target_objs = objs(context.selected_objects)
        else:
            target_objs = objs([context.object])
        if (len(target_objs) > 0
                and settings.use_deformed_target
                and any(util.has_modifier(obj, *util.TOPOLOGY_MODS) for obj in target_objs)):
            objs_str = ', '.join(obj.name for obj in target_objs)
            col = layout.column(align=True)
            col.label(text=f'Error: {objs_str}', icon='ERROR')
            col.label(text='  Topology altering Modifier!', icon='SHAPEKEY_DATA')
            col.label(text='  Deactivate Use Deformed Target or apply/delete modifier.', icon='MODIFIER')
            
        source_obj = settings.source_object
        if source_obj:
            armature_mods = [mod for mod in source_obj.modifiers if mod.type == "ARMATURE"]
            if len(armature_mods) == 0:
                col = layout.column(align=True)
                col.label(text=f'Subset is set to Deform Pose Bones,', icon='ERROR')
                col.label(text=f'but {source_obj.name} has no Armature Modifier')
            elif len(armature_mods) == 1:
                if not armature_mods[0].object:
                    col = layout.column(align=True)
                    col.label(text=f'Subset is set to Deform Pose Bones,', icon='ERROR')
                    col.label(text=f'but {source_obj.name} has an empty Armature Modifier object')
            else:
                col = layout.column(align=True)
                col.label(text=f'Subset is set to Deform Pose Bones,', icon='ERROR')
                col.label(text=f'but {source_obj.name} has multiple Armature Modifiers')
            
        row = layout.row(align=True)
        row.prop(settings, 'apply_to_selected', text='', icon='RESTRICT_SELECT_OFF')
        row.operator("object.skin_weight_transfer", text="Transfer Weights")
        row.prop(settings, "use_deformed_target",toggle=True, text="", icon='SHAPEKEY_DATA')
        row.prop(settings, "use_deformed_target",toggle=True, text="", icon='MODIFIER')
        
        layout.separator(factor=1)
        
        col = layout.column()
        col.label(text='Inpaint Mask')
        row = col.row(align=True)
        row.prop_search(props, "inpaint_group", active_obj, 'vertex_groups', text='')
        row.prop(props , "inpaint_group_invert",text="", toggle=True, icon='ARROW_LEFTRIGHT')
        row.enabled = not settings.apply_to_selected

        
class SettingsPanel(bpy.types.Panel):
    bl_idname = 'OBJECT_PT_robust_weight_transfer_settings_panel'
    bl_label = 'Settings'
    bl_space_type = 'VIEW_3D'   # Defines the space type where the panel is located
    bl_region_type = 'UI'       # Specifies that the panel is drawn in the UI region
    bl_category = 'SENT'      # The name of the tab the panel will be in
    bl_parent_id = 'OBJECT_PT_robust_weight_transfer_panel'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.robust_weight_transfer_settings
        layout.operator('object.rbt_reset_scene_settings', icon='LOOP_BACK', text='Reset to Defaults')
        layout.prop(settings, 'inpaint_mode')
        layout.prop(settings, 'new_weight_ratio')
        row = layout.row()
        row.enabled = not settings.enforce_four_bone_limit
        row.prop(settings, 'group_selection', text='Subset')


class VertexMappingPanel(bpy.types.Panel):
    bl_label = "Vertex Mapping"
    bl_idname = "OBJECT_PT_vertex_mapping"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SENT'
    bl_parent_id = 'OBJECT_PT_robust_weight_transfer_settings_panel'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.robust_weight_transfer_settings
        col = layout.column(align=True)
        col.prop(settings, "max_distance")
        row = col.row(align=True)
        row.operator("object.rwt_select_rejected_by_distance", text="Select Rejected (Islands)", icon='RESTRICT_SELECT_OFF').deselect = False
        row.operator("object.rwt_select_rejected_by_distance", text="Deselect", icon='RESTRICT_SELECT_ON').deselect = True

        col.separator(factor=0.5)
        col.prop(settings, "max_normal_angle_difference")
        row = col.row(align=True)
        row.operator("object.rwt_select_rejected_by_normal", text="Select Rejected", icon='RESTRICT_SELECT_OFF').deselect = False
        row.operator("object.rwt_select_rejected_by_normal", text="Deselect", icon='RESTRICT_SELECT_ON').deselect = True

        col.separator(factor=0.5)
        col.prop(settings, "flip_vertex_normal", text='Allow Flipped Vertex Normals')


        # --- Supergroup thresholds (representative-point pipeline) ---
        box = layout.box()
        box.label(text="Supergroups")
        col2 = box.column(align=True)

        obj_ui = _ui_pick_target_mesh_for_calc(context, settings)
        bbox_diag_ui = _ui_bbox_diag_world(obj_ui) if obj_ui else 0.0

        # Master switch (supergroup pipeline)
        col2.prop(settings, "supergroup_master_enable")
        if settings.supergroup_master_enable:
            # MID band (core supergroup merge rule)
            mid_box = col2.box()
            mid_box.label(text="MID band (distance + angle)")
            mid_box.prop(settings, "supergroup_eps_mid")
            if bbox_diag_ui > 0.0:
                lo = bbox_diag_ui * float(getattr(R, 'DOMAIN_RATIO_MIN', 1e-8))
                hi = bbox_diag_ui * float(getattr(R, 'DOMAIN_RATIO_MAX', 1e-1))
                mid_box.label(text=f"Fill range: [{lo:.6g}, {hi:.6g}]")
            else:
                mid_box.label(text="Fill range: (select a mesh)", icon='INFO')
            mid_box.prop(settings, "supergroup_theta_merge_deg")

            mid_box.separator(factor=0.25)
            mid_box.prop(settings, "supergroup_mix_alpha")
            if settings.supergroup_mix_alpha < 0.1:
                warn2 = mid_box.row()
                warn2.alert = True
                warn2.label(text="Small α can be unstable when topology matters.", icon='ERROR')

            row_btn = mid_box.row(align=True)
            row_btn.operator("object.rwt_calc_mid_prob_region", text="Calculate prob region", icon='FILE_REFRESH')

            # On-demand reference display (only after Calculate)
            mid_token_cur = _ui_make_prob_region_token('mid', obj_ui, settings)
            if settings.ui_mid_valid and settings.ui_mid_token == mid_token_cur:
                mid_box.separator(factor=0.25)
                mid_box.label(text=f"MIDBAND raw (world): {settings.ui_mid_raw:.6g}")
                mid_box.label(text=f"MIDBAND effective (world): {settings.ui_mid_effective:.6g}")
                mid_box.label(text=f"Allowed raw range: [{settings.ui_mid_allowed_min:.6g}, {settings.ui_mid_allowed_max:.6g}]")
                if settings.ui_mid_has_clamp:
                    mid_box.label(text=f"Clamp P05–P95: [{settings.ui_mid_clamp_p05:.6g}, {settings.ui_mid_clamp_p95:.6g}]")
                if settings.ui_mid_note:
                    mid_box.label(text=settings.ui_mid_note, icon='INFO')

        # Strong cleaning gate (default ON)
        col2.separator(factor=0.25)
        col2.prop(settings, "strong_clean_enable")

        if settings.strong_clean_enable:
            strong = col2.box()
            strong.label(text="Strong Clean")

            # MIN band (distance-only force-merge)
            min_box = strong.box()
            min_box.label(text="MIN band (distance-only)")
            min_box.prop(settings, "supergroup_eps_min")
            if bbox_diag_ui > 0.0:
                lo = bbox_diag_ui * float(getattr(R, 'DOMAIN_RATIO_MIN', 1e-8))
                hi = bbox_diag_ui * float(getattr(R, 'DOMAIN_RATIO_MAX', 1e-1))
                min_box.label(text=f"Fill range: [{lo:.6g}, {hi:.6g}]")
            else:
                min_box.label(text="Fill range: (select a mesh)", icon='INFO')

            row_btn = min_box.row(align=True)
            row_btn.operator("object.rwt_calc_min_prob_region", text="Calculate prob region", icon='FILE_REFRESH')

            min_token_cur = _ui_make_prob_region_token('min', obj_ui, settings)
            if settings.ui_min_valid and settings.ui_min_token == min_token_cur:
                min_box.separator(factor=0.25)
                min_box.label(text=f"MINBAND raw (world): {settings.ui_min_raw:.6g}")
                min_box.label(text=f"MINBAND effective (world): {settings.ui_min_effective:.6g}")
                min_box.label(text=f"Allowed raw range: [{settings.ui_min_allowed_min:.6g}, {settings.ui_min_allowed_max:.6g}]")
                if settings.ui_min_has_clamp:
                    min_box.label(text=f"Clamp P05–P95: [{settings.ui_min_clamp_p05:.6g}, {settings.ui_min_clamp_p95:.6g}]")
                if settings.ui_min_note:
                    min_box.label(text=settings.ui_min_note, icon='INFO')

            strong.separator(factor=0.25)

            # Face cleaning
            strong.prop(settings, "clean_degenerate_faces")
            if settings.clean_degenerate_faces:
                face_box = strong.box()
                face_box.label(text="Face Cleaning")
                face_box.prop(settings, "face_clean_height")
                if bbox_diag_ui > 0.0:
                    lo = bbox_diag_ui * float(getattr(R, 'DOMAIN_RATIO_MIN', 1e-8))
                    hi = bbox_diag_ui * float(getattr(R, 'DOMAIN_RATIO_MAX', 1e-1))
                    face_box.label(text=f"Fill range: [{lo:.6g}, {hi:.6g}]")
                else:
                    face_box.label(text="Fill range: (select a mesh)", icon='INFO')

                row_btn = face_box.row(align=True)
                row_btn.operator("object.rwt_calc_face_prob_region", text="Calculate prob region", icon='FILE_REFRESH')

                face_token_cur = _ui_make_prob_region_token('face', obj_ui, settings)
                if settings.ui_face_valid and settings.ui_face_token == face_token_cur:
                    face_box.separator(factor=0.25)
                    face_box.label(text=f"Face clean raw (world): {settings.ui_face_raw:.6g}")
                    face_box.label(text=f"Face clean ref (h_eps): {settings.ui_face_ref:.6g}")
                    face_box.label(text=f"Allowed raw range: [{settings.ui_face_allowed_min:.6g}, {settings.ui_face_allowed_max:.6g}]")
                    if settings.ui_face_has_clamp:
                        face_box.label(text=f"Clamp P05–P95: [{settings.ui_face_clamp_p05:.6g}, {settings.ui_face_clamp_p95:.6g}]")
                    if settings.ui_face_note:
                        face_box.label(text=settings.ui_face_note, icon='INFO')

            strong.separator(factor=0.25)

            # --- Relative clamp controls ---
            strong.prop(settings, "supergroup_rel_clamp_enable")

            if settings.supergroup_rel_clamp_enable:
                strong.prop(settings, "supergroup_rel_clamp_show")

                if settings.supergroup_rel_clamp_show:
                    adv = strong.box()
                    adv.label(text="REL clamp (advanced)")
                    row = adv.row(align=True)
                    row.prop(settings, "supergroup_eps_rel_min")
                    row.prop(settings, "supergroup_eps_rel_max")
                    adv.label(text="eps_min = REL_MIN * d_nn, eps_max = REL_MAX * d_nn", icon='INFO')

                    # Do NOT auto-compute; show only if a prob-region calc has cached REL values.
                    if settings.ui_rel_valid:
                        adv.separator(factor=0.25)
                        adv.label(text=f"d_nn median (world): {settings.ui_rel_dnn_median:.6g}")
                        adv.label(text=f"REL window: [{settings.ui_rel_eps_min:.6g}, {settings.ui_rel_eps_max:.6g}]")

            strong.separator(factor=0.25)

            # Island rules (NOT a master switch)
            isl_box = strong.box()
            isl_box.label(text="Island Rules")
            isl_box.prop(settings, "supergroup_enable", text="Enable Island Rules")
            isl_col = isl_box.column(align=True)
            isl_col.enabled = bool(settings.supergroup_enable)
            isl_col.prop(settings, "supergroup_reject_weak_islands")
            isl_col.prop(settings, "supergroup_promote_strong_islands")
            isl_col.prop(settings, "supergroup_min_ratio")

            strong.separator(factor=0.25)

            col2.separator(factor=0.25)


class SmoothingPanel(bpy.types.Panel):
    bl_label = "Smoothing"
    bl_idname = "OBJECT_PT_smoothing"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SENT'
    bl_parent_id = 'OBJECT_PT_robust_weight_transfer_settings_panel'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.robust_weight_transfer_settings
        layout.enabled = settings.smoothing_enable
        layout.prop(settings, 'smoothing_repeat')
        layout.prop(settings, 'smoothing_factor')
        
    def draw_header(self, context: bpy.types.Context):
        settings = context.scene.robust_weight_transfer_settings
        col = self.layout.column(align=True)
        col.prop(settings, 'smoothing_enable', text='')

class LimitGroupsPanel(bpy.types.Panel):
    bl_label = "Limit Vertex Groups"
    bl_idname = "OBJECT_PT_limit_groups"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SENT'
    bl_parent_id = 'OBJECT_PT_robust_weight_transfer_settings_panel'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.robust_weight_transfer_settings
        layout.enabled = settings.enforce_four_bone_limit
        layout.prop(settings, 'num_limit_groups')
        
    def draw_header(self, context: bpy.types.Context):
        settings = context.scene.robust_weight_transfer_settings
        col = self.layout.column(align=True)
        col.prop(settings, 'enforce_four_bone_limit', text='')


class ResetSceneSettings(bpy.types.Operator):
    """Reset all settings to their default values"""
    bl_idname = "object.rbt_reset_scene_settings"
    bl_label = "Reset Robust Weight Transfer to Default Settings"

    def execute(self, context):
        settings = context.scene.robust_weight_transfer_settings
        for prop_name, prop in settings.bl_rna.properties.items():
            if prop.is_readonly or prop_name in {'rna_type', 'name'}:
                continue
            if hasattr(prop, 'default'):
                setattr(settings, prop_name, prop.default)
            else:
                setattr(settings, prop_name, None)
        return {'FINISHED'}




# (Cache management operators removed)


def register():
    # Evaluate dependency state early so UI/Operators behave consistently.
    refresh_dependency_state(load_modules=False)

    for cls in (
        RobustWeightTransfer,
        RobustWeightTransferPanel,
        InstallDependencies,
        SettingsPanel,
        VertexMappingPanel,
        LimitGroupsPanel,
        SmoothingPanel,
        ObjectSettingsGroup,
        SceneSettingsGroup,
        SelectRejectedByDistance,
        SelectRejectedByNormal,
        ResetSceneSettings,
        Inpaint,
        CalcMidProbRegion,
        CalcMinProbRegion,
        CalcFaceProbRegion,
    ):
        bpy.utils.register_class(cls)

    bpy.types.Object.robust_weight_transfer_settings = bpy.props.PointerProperty(type=ObjectSettingsGroup)
    bpy.types.Scene.robust_weight_transfer_settings = bpy.props.PointerProperty(type=SceneSettingsGroup)

    # If deps are already available, bind algorithm functions now.
    if installed_deps and not _ALGO_LOADED:
        _load_algo_modules()


def unregister():
    # Clean up PointerProperties safely.
    if hasattr(bpy.types.Object, "robust_weight_transfer_settings"):
        del bpy.types.Object.robust_weight_transfer_settings
    if hasattr(bpy.types.Scene, "robust_weight_transfer_settings"):
        del bpy.types.Scene.robust_weight_transfer_settings

    for cls in (
        Inpaint,
        CalcMidProbRegion,
        CalcMinProbRegion,
        CalcFaceProbRegion,
        ResetSceneSettings,
        SelectRejectedByNormal,
        SelectRejectedByDistance,
        SceneSettingsGroup,
        ObjectSettingsGroup,
        SmoothingPanel,
        LimitGroupsPanel,
        VertexMappingPanel,
        SettingsPanel,
        InstallDependencies,
        RobustWeightTransferPanel,
        RobustWeightTransfer,
    ):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass