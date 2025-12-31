
import numpy as np
import scipy as sp
import bpy
import bmesh
import math
from . import thresholds as T
# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------

def is_group_valid(vertex_groups, group_name: str) -> bool:
    """Return True if group_name is a non-empty name and exists in vertex_groups."""
    if not group_name:
        return False
    try:
        return vertex_groups.get(group_name) is not None
    except Exception:
        # Fallback for unexpected vertex_groups implementations
        return any(getattr(g, "name", None) == group_name for g in vertex_groups)


# ------------------------------------------------------------
# Runtime mapping cache intentionally disabled.
# (Removed to avoid stale-cache bugs and to simplify plugin behavior.)

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

def set_group_weights(obj: bpy.types.Object, group_name: str, weights_vec: np.ndarray, threshold: float = T.MIN_MATCH_CONFIDENCE, indices: list[int] | None = None):
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



def get_obj_arrs_world(obj: bpy.types.Object):
    """Return (V_world, F_tris, N_world) for an object.

    - In Object Mode: uses mesh.loop_triangles for triangles and mesh vertex normals.
    - In Edit Mode: uses BMesh (current edit state) and triangulates faces with a simple fan.
      This keeps the main execution chain bmesh-driven.
    """
    mesh: bpy.types.Mesh = obj.data

    # Collect local-space verts, normals, and triangles
    if is_edit_mesh(obj):
        bm = get_edit_bmesh(obj)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        bm.normal_update()

        n_verts = len(bm.verts)
        V = np.empty((n_verts, 3), dtype=np.float32)
        N = np.empty((n_verts, 3), dtype=np.float32)
        for i, v in enumerate(bm.verts):
            V[i] = v.co[:]
            N[i] = v.normal[:]

        # Triangulate faces (fan). Works well for weight transfer diagnostics.
        tris = []
        for f in bm.faces:
            vs = [v.index for v in f.verts]
            if len(vs) < 3:
                continue
            if len(vs) == 3:
                tris.append(vs)
            else:
                v0 = vs[0]
                for k in range(1, len(vs) - 1):
                    tris.append([v0, vs[k], vs[k+1]])
        F = np.asarray(tris, dtype=np.int64)
    else:
        mesh.calc_loop_triangles()
        V = np.empty((len(mesh.vertices), 3), dtype=np.float32)
        F = np.empty((len(mesh.loop_triangles), 3), dtype=np.int64)
        N = np.empty((len(mesh.vertices), 3), dtype=np.float32)
        mesh.vertices.foreach_get("co", V.reshape(-1))
        mesh.loop_triangles.foreach_get("vertices", F.reshape(-1))
        mesh.vertices.foreach_get("normal", N.reshape(-1))

    # Local -> world
    world_matrix = np.array(obj.matrix_world, dtype=np.float32)
    ones = np.ones((V.shape[0], 1), dtype=np.float32)
    V4 = np.hstack((V, ones))
    WV4 = (world_matrix @ V4.T).T
    W = WV4[:, :3] / WV4[:, 3][:, None]

    # Normals: inverse-transpose of the 3x3
    M3 = world_matrix[:3, :3]
    invT = np.linalg.inv(M3).T
    WN = (invT @ N.T).T
    # normalize
    nrm = np.linalg.norm(WN, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    WN = WN / nrm

    return np.ascontiguousarray(W, dtype=np.float32), np.ascontiguousarray(F, dtype=np.int64), np.ascontiguousarray(WN, dtype=np.float32)

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


def connected_components(adj_list, active_mask=None):
    """Return connected components (list of lists of vertex indices).

    If active_mask is provided (bool array-like), only vertices with mask True
    are considered part of components; edges crossing into masked-out vertices
    are ignored.
    """
    n = len(adj_list)
    if active_mask is not None:
        active_mask = np.asarray(active_mask, dtype=bool)
        if active_mask.shape[0] != n:
            raise ValueError("active_mask length must equal adjacency size")
    visited = np.zeros(n, dtype=bool)
    comps = []
    for i in range(n):
        if visited[i]:
            continue
        if active_mask is not None and not active_mask[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for nb in adj_list[v]:
                if visited[nb]:
                    continue
                if active_mask is not None and not active_mask[nb]:
                    continue
                visited[nb] = True
                stack.append(nb)
        comps.append(comp)
    return comps


def apply_supergroup_matching(matched, pass_dist, pass_norm, adj_list, *, basis="DIST_AND_NORM", mode="REJECT_ISLANDS", min_ratio=0.5):
    """Adjust matched mask using connected-component ("supergroup") logic."""
    matched = np.asarray(matched, dtype=bool)
    pass_dist = np.asarray(pass_dist, dtype=bool)
    pass_norm = np.asarray(pass_norm, dtype=bool)
    n = matched.shape[0]
    if pass_dist.shape[0] != n or pass_norm.shape[0] != n:
        raise ValueError("matched/pass_dist/pass_norm lengths must match")

    if basis == "DIST":
        gate = pass_dist
    else:
        gate = pass_dist & pass_norm

    comps = connected_components(adj_list)
    out = matched.copy()
    modified = np.zeros(n, dtype=bool)

    for comp in comps:
        if not comp:
            continue
        comp_idx = np.asarray(comp, dtype=int)
        ratio = float(gate[comp_idx].mean())
        if mode == "REJECT_ISLANDS":
            if ratio < min_ratio:
                out[comp_idx] = False
                modified[comp_idx] = True
        elif mode == "PROMOTE_ISLANDS":
            if ratio >= min_ratio:
                out[comp_idx] = True
                modified[comp_idx] = True
        else:
            raise ValueError("Unknown mode")

    return out, modified




# ---------------------------------------------------------------------------
# Supergroup construction (spatial + normal-gated union)
# ---------------------------------------------------------------------------

class _UF:
    __slots__ = ("p","sz")
    def __init__(self, n:int):
        self.p=list(range(n))
        self.sz=[1]*n
    def find(self,a:int)->int:
        p=self.p
        while p[a]!=a:
            p[a]=p[p[a]]
            a=p[a]
        return a
    def union(self,a:int,b:int):
        ra=self.find(a); rb=self.find(b)
        if ra==rb: return
        if self.sz[ra] < self.sz[rb]:
            ra,rb=rb,ra
        self.p[rb]=ra
        self.sz[ra]+=self.sz[rb]

def _cell_key(xyz: np.ndarray, cell: float):
    return (int(math.floor(xyz[0]/cell)), int(math.floor(xyz[1]/cell)), int(math.floor(xyz[2]/cell)))

def _neighbor_keys(ck):
    x,y,z=ck
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            for dz in (-1,0,1):
                yield (x+dx,y+dy,z+dz)

def build_supergroups_from_world(
    V_world: np.ndarray,
    N_world: np.ndarray,
    *,
    eps_force,
    eps_norm,
    angle_deg: float
):
    """Build supergroups using distance thresholds and normal-gated union.

    Supports either scalar thresholds or per-vertex thresholds (shape=(N,)).

    Rules (for a pair i,j):
    - dist <= min(eps_force[i], eps_force[j]): always union
    - dist <= min(eps_norm[i],  eps_norm[j]): union if angle(Ni, Nj) < angle_deg
    - otherwise: no union

    Notes:
    - For varying thresholds, the pairwise threshold uses `min` so dense regions
      are not forced to merge by a neighbor in a sparse region.
    """
    V = np.asarray(V_world, dtype=np.float64)
    N = np.asarray(N_world, dtype=np.float64)
    n = int(V.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int32), []

    # Determine whether thresholds are scalar or per-vertex.
    eps_force_is_scalar = np.isscalar(eps_force)
    eps_norm_is_scalar = np.isscalar(eps_norm)

    if eps_force_is_scalar:
        eps_force_v = float(eps_force)
    else:
        eps_force_v = np.asarray(eps_force, dtype=np.float64).reshape(-1)
        if eps_force_v.shape[0] != n:
            raise ValueError(f"eps_force length ({eps_force_v.shape[0]}) != vertex count ({n})")

    if eps_norm_is_scalar:
        eps_norm_v = float(eps_norm)
    else:
        eps_norm_v = np.asarray(eps_norm, dtype=np.float64).reshape(-1)
        if eps_norm_v.shape[0] != n:
            raise ValueError(f"eps_norm length ({eps_norm_v.shape[0]}) != vertex count ({n})")

    # Cell size for spatial hashing: needs to be at least as large as the largest eps_norm.
    if eps_norm_is_scalar:
        cell = max(float(eps_norm_v), 1e-12)
    else:
        cell = max(float(np.max(eps_norm_v)) if eps_norm_v.size else 1e-12, 1e-12)

    cmap = {}
    for i in range(n):
        ck = _cell_key(V[i], cell)
        cmap.setdefault(ck, []).append(i)

    uf = _UF(n)

    # Precompute squared thresholds.
    if eps_force_is_scalar:
        eps_force2 = float(eps_force_v) * float(eps_force_v)
    else:
        eps_force2 = eps_force_v * eps_force_v

    if eps_norm_is_scalar:
        eps_norm2 = float(eps_norm_v) * float(eps_norm_v)
    else:
        eps_norm2 = eps_norm_v * eps_norm_v

    cos_th = math.cos(math.radians(float(angle_deg)))

    for ck, ids in cmap.items():
        for nk in _neighbor_keys(ck):
            nbr = cmap.get(nk)
            if not nbr:
                continue
            for i in ids:
                vi = V[i]
                ni = N[i]
                for j in nbr:
                    if j <= i:
                        continue
                    d = vi - V[j]
                    dd = float(d[0]*d[0] + d[1]*d[1] + d[2]*d[2])

                    if eps_force_is_scalar:
                        force2 = eps_force2
                    else:
                        # pairwise threshold: min(eps_i, eps_j)
                        force2 = float(min(eps_force2[i], eps_force2[j]))

                    if dd <= force2:
                        uf.union(i, j)
                        continue

                    if eps_norm_is_scalar:
                        norm2 = eps_norm2
                    else:
                        norm2 = float(min(eps_norm2[i], eps_norm2[j]))

                    if dd <= norm2:
                        nj = N[j]
                        c = float(ni[0]*nj[0] + ni[1]*nj[1] + ni[2]*nj[2])
                        if c >= cos_th:
                            uf.union(i, j)

    root_to_id = {}
    sgid = np.empty((n,), dtype=np.int32)
    comps = []
    for i in range(n):
        r = uf.find(i)
        sid = root_to_id.get(r)
        if sid is None:
            sid = len(comps)
            root_to_id[r] = sid
            comps.append([])
        sgid[i] = sid
        comps[sid].append(i)
    return sgid, comps


def object_scale_world(V_world: np.ndarray) -> float:
    if V_world.shape[0] == 0:
        return 1.0
    mn = V_world.min(axis=0)
    mx = V_world.max(axis=0)
    diag = mx - mn
    s = float(np.linalg.norm(diag))
    return s if s > 0 else 1.0

def compute_source_supergroup_rep_weights(source_weights: np.ndarray, source_sgid: np.ndarray, *, submesh_mask: np.ndarray | None = None, mask_threshold: float = T.EPS_MASK_FOR_INPAINT):
    """Compute representative weight vector for each source supergroup.

    If submesh_mask is provided, and a supergroup contains mask vertices (mask > threshold),
    the representative is computed from only those vertices; otherwise from all vertices in the supergroup.
    """
    W = np.asarray(source_weights, dtype=np.float64)
    sgid = np.asarray(source_sgid, dtype=np.int32)
    n_sg = int(sgid.max()) + 1 if sgid.size else 0
    rep = np.zeros((n_sg, W.shape[1]), dtype=np.float64)

    if n_sg == 0:
        return rep.astype(np.float32)

    if submesh_mask is not None:
        m = np.asarray(submesh_mask, dtype=np.float64).reshape(-1)
    else:
        m = None

    for sid in range(n_sg):
        idx = np.where(sgid == sid)[0]
        if idx.size == 0:
            continue
        use_idx = idx
        if m is not None:
            m_idx = idx[m[idx] > mask_threshold]
            if m_idx.size > 0:
                use_idx = m_idx
        rep[sid] = W[use_idx].mean(axis=0)
    return rep.astype(np.float32)

def face_to_supergroup(F_tris: np.ndarray, sgid: np.ndarray) -> np.ndarray:
    F = np.asarray(F_tris, dtype=np.int64)
    sg = np.asarray(sgid, dtype=np.int32)
    out = np.empty((F.shape[0],), dtype=np.int32)
    for fi, (a,b,c) in enumerate(F):
        sa, sb, sc = int(sg[a]), int(sg[b]), int(sg[c])
        # mode of 3
        if sa == sb or sa == sc:
            out[fi] = sa
        elif sb == sc:
            out[fi] = sb
        else:
            out[fi] = sa
    return out


def write_weights(obj, weights, names, threshold=T.MIN_MATCH_CONFIDENCE, scope_indices=None):
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