# This file is part of Robust Weight Transfer for Blender.
#
# Copyright (C) 2025 sentfromspacevr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Attribution: Developed by sentfromspacevr
import numpy as np
import scipy as sp
import bpy
import bmesh

# ------------------------------------------------------------
# Runtime mapping cache (NOT saved into .blend)
# ------------------------------------------------------------

# Keyed by (target_obj_ptr, source_obj_ptr, params_hash)
_RWT_MAPPING_CACHE: dict[tuple[int, int, int], dict] = {}


def _hash_params(*items) -> int:
    """Stable-ish hash for cache keys (process lifetime only)."""
    return hash(tuple(items))


def cache_key(target_obj: bpy.types.Object, source_obj: bpy.types.Object, params_hash: int) -> tuple[int, int, int]:
    return (int(target_obj.as_pointer()), int(source_obj.as_pointer()), int(params_hash))


def cache_store(target_obj: bpy.types.Object, source_obj: bpy.types.Object, params_hash: int, payload: dict):
    _RWT_MAPPING_CACHE[cache_key(target_obj, source_obj, params_hash)] = payload


def cache_get(target_obj: bpy.types.Object, source_obj: bpy.types.Object, params_hash: int) -> dict | None:
    return _RWT_MAPPING_CACHE.get(cache_key(target_obj, source_obj, params_hash))


def cache_clear(target_obj: bpy.types.Object | None = None):
    """Clear runtime cache.

    - If target_obj is None: clear everything.
    - Else: clear entries for that active target.
    """
    if target_obj is None:
        _RWT_MAPPING_CACHE.clear()
        return
    tgt_ptr = int(target_obj.as_pointer())
    for k in [k for k in _RWT_MAPPING_CACHE.keys() if k[0] == tgt_ptr]:
        _RWT_MAPPING_CACHE.pop(k, None)


def params_hash_for_mapping(
    *,
    max_distance: float,
    max_normal_angle: float,
    flip_vertex_normal: bool,
    use_deformed_source: bool,
    use_deformed_target: bool,
    inpaint_group: str,
    inpaint_threshold: float,
    inpaint_invert: bool,
    apply_to_selected: bool,
    exclude_success_ring: int,
) -> int:
    return _hash_params(
        float(max_distance),
        float(max_normal_angle),
        bool(flip_vertex_normal),
        bool(use_deformed_source),
        bool(use_deformed_target),
        str(inpaint_group),
        float(inpaint_threshold),
        bool(inpaint_invert),
        bool(apply_to_selected),
        int(exclude_success_ring),
    )


def select_mask_vertices(obj: bpy.types.Object, mask: np.ndarray, *, action: str = 'select'):
    """Select/deselect vertices by boolean mask.

    action:
      - 'select': select vertices where mask is True (leave others unchanged)
      - 'deselect': deselect vertices where mask is True (leave others unchanged)
    """
    if obj is None or obj.type != 'MESH':
        return
    mesh = obj.data
    if not isinstance(mesh, bpy.types.Mesh):
        return

    if obj.mode == 'EDIT':
        bm = bmesh.from_edit_mesh(mesh)
        bm.verts.ensure_lookup_table()
        if action == 'select':
            for i, m in enumerate(mask):
                if m:
                    bm.verts[i].select_set(True)
        else:
            for i, m in enumerate(mask):
                if m:
                    bm.verts[i].select_set(False)
        bm.select_flush(True)
        bm.select_flush(False)
        bmesh.update_edit_mesh(mesh, destructive=False)
        return

    # Object mode
    if action == 'select':
        for i, m in enumerate(mask):
            if m:
                mesh.vertices[i].select = True
    else:
        for i, m in enumerate(mask):
            if m:
                mesh.vertices[i].select = False
    mesh.update()


def expand_mask_rings(mesh: bpy.types.Mesh, seed_mask: np.ndarray, rings: int) -> np.ndarray:
    """Topologically expand a boolean vertex mask by N edge rings."""
    if rings <= 0:
        return seed_mask.copy()
    if seed_mask.dtype != bool:
        seed_mask = seed_mask.astype(bool)
    adj = get_mesh_adjacency_list(mesh)
    out = seed_mask.copy()
    frontier = set(np.where(seed_mask)[0].tolist())
    for _ in range(rings):
        new_frontier = set()
        for v in frontier:
            for n in adj[v]:
                if not out[n]:
                    out[n] = True
                    new_frontier.add(n)
        if not new_frontier:
            break
        frontier = new_frontier
    return out



# ------------------------------------------------------------
# Edit Mode helpers (BMesh deform layer)
# ------------------------------------------------------------

def is_edit_mesh(obj: bpy.types.Object) -> bool:
    return bool(obj and obj.type == 'MESH' and obj.mode == 'EDIT' and isinstance(obj.data, bpy.types.Mesh))

def get_edit_bmesh(obj: bpy.types.Object) -> bmesh.types.BMesh | None:
    if not is_edit_mesh(obj):
        return None
    bm = bmesh.from_edit_mesh(obj.data)
    # Ensure deform layer exists
    bm.verts.layers.deform.verify()
    return bm

def get_selected_vert_indices(obj: bpy.types.Object) -> list[int]:
    bm = get_edit_bmesh(obj)
    if bm is None:
        return []
    return [v.index for v in bm.verts if v.select]

def set_group_weights(obj: bpy.types.Object, group_name: str, weights_vec: np.ndarray, threshold: float = 0.0, indices: list[int] | None = None):
    """Set weights for a single vertex group.

    - In Edit Mode: uses BMesh deform layer (works immediately in Edit Mode).
    - In Object Mode: uses vertex_group.add/remove.
    - If indices is provided, only those vertex indices are modified; others are untouched.
    """
    if group_name not in obj.vertex_groups:
        obj.vertex_groups.new(name=group_name)

    vg = obj.vertex_groups[group_name]
    if vg.lock_weight:
        return

    mesh = obj.data
    if not isinstance(mesh, bpy.types.Mesh):
        return

    n = len(mesh.vertices)
    if weights_vec.shape[0] != n:
        raise ValueError(f"weights_vec length ({weights_vec.shape[0]}) does not match vertex count ({n})")

    if indices is None:
        indices = list(range(n))

    if is_edit_mesh(obj):
        bm = get_edit_bmesh(obj)
        dvert = bm.verts.layers.deform.active
        gi = vg.index
        for vi in indices:
            v = bm.verts[vi]
            dv = v[dvert]
            w = float(weights_vec[vi])
            if w >= threshold:
                dv[gi] = w
            else:
                if gi in dv:
                    del dv[gi]
        bmesh.update_edit_mesh(mesh, loop_triangles=False, destructive=False)
        return

    # Object mode (or any non-edit mesh mode)
    add_ids = [vi for vi in indices if float(weights_vec[vi]) >= threshold]
    for vi in add_ids:
        vg.add([vi], float(weights_vec[vi]), 'REPLACE')

    # Remove only within the indices scope
    rem_ids = [vi for vi in indices if float(weights_vec[vi]) < threshold]
    if rem_ids:
        vg.remove(rem_ids)


def get_obj_arrs_world(object: bpy.types.Object):
    mesh: bpy.types.Mesh = object.data
    mesh.calc_loop_triangles()

    vertices = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    indices = np.empty((len(mesh.loop_triangles), 3), dtype=np.int64)
    normals = np.empty((len(mesh.vertices), 3), dtype=np.float32)

    mesh.vertices.foreach_get("co", vertices.reshape(-1))
    mesh.loop_triangles.foreach_get("vertices", indices.reshape(-1))
    # mesh.vertex_normals.foreach_get('vector', normals.reshape(-1))
    mesh.vertices.foreach_get('normal', normals.reshape(-1))
    
    world_matrix = np.array(object.matrix_world)
    ones = np.ones((vertices.shape[0], 1))
    
    vertices_4d = np.hstack((vertices, ones))
    world_vertices_4d = (world_matrix @ vertices_4d.T).T
    world_vertices_4d = np.ascontiguousarray(world_vertices_4d, dtype=np.float32)
    world_vertices = world_vertices_4d[:,:3] / world_vertices_4d[:, 3][:, np.newaxis]
    
    world_normals = (np.linalg.inv(world_matrix[:3, :3]).T @ normals.T).T
    world_normals = np.ascontiguousarray(world_normals, dtype=np.float32)
    
    return world_vertices, indices, world_normals


def get_group_arr(obj: bpy.types.Object, group_name):
    mesh: bpy.types.Mesh = obj.data
    if not isinstance(mesh, bpy.types.Mesh): 
        return
    if group_name not in obj.vertex_groups:
        return np.zeros(len(mesh.vertices), dtype=np.float32)

    group_index = obj.vertex_groups[group_name].index
    arr = np.zeros(len(mesh.vertices), dtype=np.float32)

    if is_edit_mesh(obj):
        bm = get_edit_bmesh(obj)
        dvert = bm.verts.layers.deform.active
        for v in bm.verts:
            dv = v[dvert]
            if group_index in dv:
                arr[v.index] = float(dv[group_index])
        return arr

    for i, v in enumerate(mesh.vertices):
        for g in v.groups:
            if g.group == group_index:
                arr[i] = g.weight
                break
    return arr
def get_groups_arr(obj: bpy.types.Object, include_groups: list[bool]=None):
    mesh: bpy.types.Mesh = obj.data
    if not isinstance(mesh, bpy.types.Mesh): 
        return

    num_groups = len(obj.vertex_groups)
    arr = np.zeros((len(mesh.vertices), num_groups), dtype=np.float32)

    if is_edit_mesh(obj):
        bm = get_edit_bmesh(obj)
        dvert = bm.verts.layers.deform.active
        for v in bm.verts:
            dv = v[dvert]
            row = arr[v.index]
            for gi, w in dv.items():
                if gi >= num_groups:
                    continue
                if include_groups is None or include_groups[gi]:
                    row[gi] = float(w)
        return arr

    for i, v in enumerate(mesh.vertices):
        current_vertex = arr[i]
        for g in v.groups:
            if g.group >= arr.shape[1]:
                print(f"WARNING: group index {g.group} out of bounds ({arr.shape[1]} vertex groups) for vertex {i}")
                continue
            if include_groups is not None:
                if include_groups[g.group]:
                    current_vertex[g.group] = g.weight
            else:
                current_vertex[g.group] = g.weight
    return arr
def draw_debug_vertex_colors(obj, matched):
    mesh: bpy.types.Mesh = obj.data
    if not isinstance(mesh, bpy.types.Mesh): return

    if "RBT Matched" in mesh.vertex_colors:
        color_layer = mesh.vertex_colors["RBT Matched"]
    else:
        color_layer = mesh.vertex_colors.new(name="RBT Matched")
    if not color_layer: return False
    color_layer.active = True
    loop_ind = np.zeros(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get('vertex_index', loop_ind)
    loop_matched = matched[loop_ind]
    color_data = np.ones((len(mesh.loops), 4), dtype=np.float32)
    color_data[~loop_matched] = [234/255, 0, 255/255, 1.0]
    color_layer.data.foreach_set("color", color_data.reshape(-1))
    mesh.update()
    mesh.vertex_colors.active = color_layer
    return True
    
    
TOPOLOGY_MODS = {
    'ARRAY',
    'BEVEL',
    'BOOLEAN',
    'BUILD',
    'DECIMATE',
    'EDGE_SPLIT',
    'MASK',
    'MIRROR',
    'MULTIRES',
    'REMESH',
    'SCREW',
    'SKIN',
    'SOLIDIFY',
    'SUBSURF',
    'TRIANGULATE',
    'WELD',
    'WIREFRAME'
}


def has_modifier(obj: bpy.types.Object, *mod_types):
    if obj and obj.type == 'MESH' and obj.modifiers:
        return any(mod.type in mod_types for mod in obj.modifiers)
    return False


# TODO: source object required to have armature modifier
# throw exceptions
def is_vertex_group_deform_bone(obj, group_name):
    armature_mod = None
    for mod in obj.modifiers:
        if mod.type == 'ARMATURE':
            armature_mod = mod
            break

    if not armature_mod or not armature_mod.object or armature_mod.object.type != 'ARMATURE':
        return False

    armature_obj = armature_mod.object
    bone = armature_obj.data.bones.get(group_name)

    if bone and bone.use_deform:
        return True

    return False


def get_mesh_adjacency_matrix_sparse(mesh: bpy.types.Mesh, include_self=False):
    edge_data = np.empty((len(mesh.edges), 2), dtype=int)
    mesh.edges.foreach_get("vertices", edge_data.reshape(-1))
    num_verts = len(mesh.vertices)
    rows = np.hstack([edge_data[:, 0], edge_data[:, 1]])
    cols = np.hstack([edge_data[:, 1], edge_data[:, 0]])
    data = np.ones(len(rows), dtype=int)  # Corresponding data entries for the CSR matrix

    # Create a symmetric adjacency matrix (since each edge is undirected)
    adjacency_matrix = sp.sparse.csr_array((data, (rows, cols)), shape=(num_verts, num_verts))
    if include_self:
        adjacency_matrix.setdiag(1)
    return adjacency_matrix
    
    
def get_mesh_adjacency_list(mesh: bpy.types.Mesh):
    edge_data = np.empty((len(mesh.edges), 2), dtype=int)
    mesh.edges.foreach_get("vertices", edge_data.reshape(-1))
    num_verts = len(mesh.vertices)
    adj_list = [[] for _ in range(num_verts)]
    for edge in edge_data:
        adj_list[edge[0]].append(edge[1])
        adj_list[edge[1]].append(edge[0])
    return adj_list


def write_weights(obj, weights, names, threshold=0, scope_indices=None):
    """Write weights matrix to vertex groups.

    If scope_indices is provided, only those vertices are modified (useful for Edit Mode selection scope).
    """
    if scope_indices is not None and len(scope_indices) == 0:
        return

    groups = obj.vertex_groups
    mesh = obj.data
    if not isinstance(mesh, bpy.types.Mesh):
        return

    n = len(mesh.vertices)
    if weights.shape[0] != n:
        raise ValueError(f"weights rows ({weights.shape[0]}) != vertex count ({n})")

    for name, w in zip(names, weights.T):
        # Ensure group exists
        if name not in groups:
            obj.vertex_groups.new(name=name)

        vg = obj.vertex_groups[name]
        if vg.lock_weight:
            continue

        set_group_weights(obj, name, w, threshold=threshold, indices=scope_indices)
