

import igl
import numpy as np
import scipy as sp
import robust_laplacian
from scipy.spatial import cKDTree

from . import thresholds as T


def find_closest_point_on_surface(P, V, F):
    """
    Given a number of points find their closest points on the surface of the V,F mesh

    Args:
        P: #P by 3, where every row is a point coordinate
        V: #V by 3 mesh vertices
        F: #F by 3 mesh triangles indices
    Returns:
        sqrD #P smallest squared distances
        I #P primitive indices corresponding to smallest distances
        C #P by 3 closest points
        B #P by 3 of the barycentric coordinates of the closest point
    """
    
    # libigl's Python bindings are strict about dtypes/layout.
    # Blender-side arrays are frequently float32, while some igl routines
    # return float64. Downstream calls (e.g. barycentric_coordinates_tri)
    # require consistent dtype across all inputs.

    # Force float64 + C-contiguous for all point/vertex arrays.
    P = np.ascontiguousarray(np.asarray(P, dtype=np.float64), dtype=np.float64)
    V = np.ascontiguousarray(np.asarray(V, dtype=np.float64), dtype=np.float64)
    # Faces: keep integer and C-contiguous (libigl accepts int32/int64).
    F = np.ascontiguousarray(np.asarray(F, dtype=np.int64), dtype=np.int64)

    sqrD, I, C = igl.point_mesh_squared_distance(P, V, F)

    F_closest = F[I,:]
    V1 = V[F_closest[:,0],:]
    V2 = V[F_closest[:,1],:]
    V3 = V[F_closest[:,2],:]

    # Ensure matching dtype for barycentric computation.
    C = np.ascontiguousarray(np.asarray(C, dtype=np.float64), dtype=np.float64)
    V1 = np.ascontiguousarray(np.asarray(V1, dtype=np.float64), dtype=np.float64)
    V2 = np.ascontiguousarray(np.asarray(V2, dtype=np.float64), dtype=np.float64)
    V3 = np.ascontiguousarray(np.asarray(V3, dtype=np.float64), dtype=np.float64)

    B = igl.barycentric_coordinates_tri(C, V1, V2, V3)

    # Safety fuse: barycentric_coordinates_tri can produce NaN/Inf for degenerate
    # or near-degenerate triangles (or numerical noise). Replace invalid barycentrics
    # with a stable nearest-vertex barycentric (one-hot) to prevent downstream melt.
    if not np.all(np.isfinite(B)):
        bad = ~np.isfinite(B).all(axis=1)
        if np.any(bad):
            # pick closest vertex among triangle corners for each bad row
            C_bad = C[bad]
            V1_bad = V1[bad]
            V2_bad = V2[bad]
            V3_bad = V3[bad]
            d1 = np.sum((C_bad - V1_bad) ** 2, axis=1)
            d2 = np.sum((C_bad - V2_bad) ** 2, axis=1)
            d3 = np.sum((C_bad - V3_bad) ** 2, axis=1)
            choice = np.stack([d1, d2, d3], axis=1).argmin(axis=1)
            B_fix = np.zeros((C_bad.shape[0], 3), dtype=np.float64)
            B_fix[np.arange(C_bad.shape[0]), choice] = 1.0
            B[bad] = B_fix

            # Also snap closest point C to that vertex so sqrD is consistent.
            C_snap = np.where(choice[:, None] == 0, V1_bad,
                              np.where(choice[:, None] == 1, V2_bad, V3_bad))
            C[bad] = C_snap
            sqrD[bad] = np.sum((P[bad] - C_snap) ** 2, axis=1)

    return sqrD,I,C,B


def _filter_degenerate_triangles_by_height(V: np.ndarray, F: np.ndarray, h_eps: float) -> np.ndarray:
    """Filter triangles whose geometric height is below h_eps.

    Height is computed as h = ||(v2-v1) x (v3-v1)|| / Lmax, where Lmax is the
    max edge length in the triangle. This catches the common case where one edge
    is fine but the third point is nearly collinear ("flat" triangle).

    V: (#V,3) float64
    F: (#F,3) int
    h_eps: world-unit threshold (same unit as V)
    """
    if F is None or len(F) == 0:
        return F
    if h_eps <= 0.0:
        return F

    V = np.asarray(V)
    F = np.asarray(F)

    v1 = V[F[:, 0]]
    v2 = V[F[:, 1]]
    v3 = V[F[:, 2]]
    e1 = v2 - v1
    e2 = v3 - v1
    A2 = np.linalg.norm(np.cross(e1, e2), axis=1)

    l12 = np.linalg.norm(v2 - v1, axis=1)
    l23 = np.linalg.norm(v3 - v2, axis=1)
    l31 = np.linalg.norm(v1 - v3, axis=1)
    Lmax = np.maximum(l12, np.maximum(l23, l31))

    # Avoid division by zero for fully collapsed triangles.
    Lmax = np.maximum(Lmax, 1e-30)
    h = A2 / Lmax
    keep = np.isfinite(h) & (h >= float(h_eps))
    return F[keep]

def interpolate_attribute_from_bary(A,B,I,F):
    """
    Interpolate per-vertex attributes A via barycentric coordinates B of the F[I,:] vertices

    Args:
        A: #V by N per-vertex attributes
        B  #B by 3 array of the barycentric coordinates of some points
        I  #B primitive indices containing the closest point
        F: #F by 3 mesh triangle indices
    Returns:
        A_out #B interpolated attributes
    """
    F_closest = F[I,:]
    a1 = A[F_closest[:,0],:]
    a2 = A[F_closest[:,1],:]
    a3 = A[F_closest[:,2],:]

    b1 = B[:,0]
    b2 = B[:,1]
    b3 = B[:,2]

    b1 = b1.reshape(-1,1)
    b2 = b2.reshape(-1,1)
    b3 = b3.reshape(-1,1)
    
    A_out = a1*b1 + a2*b2 + a3*b3

    return A_out


def normalize_vec(v):
    return v/np.linalg.norm(v)


def find_matches_closest_surface(
    source_verts,
    source_triangles,
    source_normals,
    target_verts,
    target_normals,
    source_weights,
    dDISTANCE_THRESHOLD_SQRD,
    dANGLE_THRESHOLD_DEGREES,
    flip_vertex_normal,
    *,
    clean_degenerate_faces: bool = False,
    face_height_eps: float | None = None,
    return_diagnostics: bool = False,
):
    """
    For each vertex on the target mesh find a match on the source mesh.

    Args:
        V1: #V1 by 3 source mesh vertices
        F1: #F1 by 3 source mesh triangles indices
        N1: #V1 by 3 source mesh normals
        
        V2: #V2 by 3 target mesh vertices
        F2: #F2 by 3 target mesh triangles indices
        N2: #V2 by 3 target mesh normals
        
        W1: #V1 by num_bones source mesh skin weights

        dDISTANCE_THRESHOLD_SQRD: scalar distance threshold
        dANGLE_THRESHOLD_DEGREES: scalar normal threshold

    Returns:
        Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh
        W2: #V2 by num_bones, where W2[i,:] are skinning weights copied directly from source using closest point method
    """
    # Optional: pre-clean degenerate triangles (numerical stability).
    # We do NOT compare raw triangle area to a distance-scale threshold; instead we
    # filter by the triangle's minimum geometric "height" (world units).
    #   h = ||(v2-v1) x (v3-v1)|| / Lmax
    # where Lmax is the max edge length. This catches the common case of "flat" tris.
    F_use = source_triangles
    if clean_degenerate_faces and (face_height_eps is not None):
        try:
            h_eps = float(face_height_eps)
        except Exception:
            h_eps = 0.0
        F_use = _filter_degenerate_triangles_by_height(source_verts, source_triangles, h_eps)

    # If the filtered surface is empty, fall back to nearest-vertex mapping.
    if F_use is None or len(F_use) == 0:
        kdt = cKDTree(np.asarray(source_verts, dtype=np.float64))
        dist, idx = kdt.query(np.asarray(target_verts, dtype=np.float64), k=1)
        idx = np.asarray(idx, dtype=np.int64)
        sqrD = (np.asarray(dist, dtype=np.float64) ** 2)
        W2 = source_weights[idx]
        N1_match_interpolated = source_normals[idx]
        I = idx  # for diagnostics; not a face index in this fallback
    else:
        sqrD,I,C,B = find_closest_point_on_surface(target_verts,source_verts,F_use)

        # for each closest point on the source, interpolate its per-vertex attributes(skin weights and normals)
        # using the barycentric coordinates
        W2 = interpolate_attribute_from_bary(source_weights,B,I,F_use)
        N1_match_interpolated = interpolate_attribute_from_bary(source_normals,B,I,F_use)
    
    # (W2, N1_match_interpolated) have been computed above for either surface or fallback mode.
    
    norm_N1 = np.linalg.norm(N1_match_interpolated, axis=1, keepdims=True)
    norm_N2 = np.linalg.norm(target_normals, axis=1, keepdims=True)
    normalized_N1 = N1_match_interpolated / norm_N1
    normalized_N2 = target_normals / norm_N2

    dot_product = np.einsum('ij,ij->i', normalized_N1, normalized_N2)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure the dot product is in the valid range for arccos
    rad_angles = np.arccos(dot_product)
    deg_angles = np.degrees(rad_angles)
    is_distance_threshold = sqrD <= dDISTANCE_THRESHOLD_SQRD
    angle_thresholds = np.full(deg_angles.shape, dANGLE_THRESHOLD_DEGREES)

    is_deg_threshold = deg_angles <= angle_thresholds
    if flip_vertex_normal:
        deg_angles_mirror = 180 - deg_angles
        is_deg_threshold = np.logical_or(is_deg_threshold, deg_angles_mirror <= angle_thresholds)

    Matched = np.logical_and(is_distance_threshold, is_deg_threshold)

    if not return_diagnostics:
        return Matched, W2

    # Diagnostics for UI/cache:
    # - sqrD: squared distance to closest point on source surface
    # - deg_angles: angle between interpolated source normal at closest point and target normal
    # - is_distance_threshold: passed distance gate
    # - is_deg_threshold: passed normal gate
    return Matched, W2, sqrD, deg_angles, is_distance_threshold, is_deg_threshold, I


def find_matches_supergroup_knn_point_interface(
    source_verts,
    source_normals,
    source_weights,
    target_verts,
    target_normals,
    dDISTANCE_THRESHOLD_SQRD,
    dANGLE_THRESHOLD_DEGREES,
    flip_vertex_normal,
    *,
    k: int = 3,
    return_diagnostics: bool = False,
):
    """Interface stub for the upcoming supergroup-based *point* matcher.

    This is intentionally NOT wired into the operator yet.

    It mirrors the diagnostics contract of `find_matches_closest_surface`, but
    replaces "closest point on triangle" with vertex-to-vertex / kNN.

    Notes:
    - Parameters like `clean_degenerate_faces` / `face_height_eps` do not apply here.
    - Weight transfer uses inverse-distance weighting over k nearest source vertices.
    - Normal for the match is the same inverse-distance blend of source vertex normals.
    """
    V1 = np.ascontiguousarray(np.asarray(source_verts, dtype=np.float64), dtype=np.float64)
    N1 = np.ascontiguousarray(np.asarray(source_normals, dtype=np.float64), dtype=np.float64)
    W1 = np.ascontiguousarray(np.asarray(source_weights, dtype=np.float64), dtype=np.float64)
    V2 = np.ascontiguousarray(np.asarray(target_verts, dtype=np.float64), dtype=np.float64)
    N2 = np.ascontiguousarray(np.asarray(target_normals, dtype=np.float64), dtype=np.float64)

    k = int(max(1, k))
    kdt = cKDTree(V1)
    dist, idx = kdt.query(V2, k=k)
    dist = np.asarray(dist, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64)
    if k == 1:
        dist = dist.reshape(-1, 1)
        idx = idx.reshape(-1, 1)

    # Inverse-distance weights (with hard snap for exact matches).
    w = 1.0 / np.maximum(dist, 1e-30)
    exact = dist[:, 0] <= 1e-15
    if np.any(exact):
        w[exact, :] = 0.0
        w[exact, 0] = 1.0
    wsum = np.sum(w, axis=1, keepdims=True)
    w = w / np.maximum(wsum, 1e-30)

    # Blend attributes.
    W2 = np.sum(W1[idx] * w[..., None], axis=1)
    N1m = np.sum(N1[idx] * w[..., None], axis=1)
    sqrD = np.sum((dist[:, 0]) ** 2, axis=0) if dist.ndim == 1 else (dist[:, 0] ** 2)

    norm_N1 = np.linalg.norm(N1m, axis=1, keepdims=True)
    norm_N2 = np.linalg.norm(N2, axis=1, keepdims=True)
    normalized_N1 = N1m / np.maximum(norm_N1, 1e-30)
    normalized_N2 = N2 / np.maximum(norm_N2, 1e-30)

    dot_product = np.einsum('ij,ij->i', normalized_N1, normalized_N2)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    rad_angles = np.arccos(dot_product)
    deg_angles = np.degrees(rad_angles)
    is_distance_threshold = sqrD <= dDISTANCE_THRESHOLD_SQRD
    angle_thresholds = np.full(deg_angles.shape, dANGLE_THRESHOLD_DEGREES)
    is_deg_threshold = deg_angles <= angle_thresholds
    if flip_vertex_normal:
        deg_angles_mirror = 180 - deg_angles
        is_deg_threshold = np.logical_or(is_deg_threshold, deg_angles_mirror <= angle_thresholds)

    Matched = np.logical_and(is_distance_threshold, is_deg_threshold)

    if not return_diagnostics:
        return Matched, W2
    # In this stub, we return the index of the closest source vertex as "I".
    return Matched, W2, sqrD, deg_angles, is_distance_threshold, is_deg_threshold, idx[:, 0]


def inpaint(V2, F2, W2, Matched, point_cloud):
    """
    Inpaint weights for all the vertices on the target mesh for which  we didnt 
    find a good match on the source (i.e. Matched[i] == False).

    Args:
        V2: #V2 by 3 target mesh vertices
        F2: #F2 by 3 target mesh triangles indices
        W2: #V2 by num_bones, where W2[i,:] are skinning weights copied directly from source using closest point method
        Matched: #V2 array of bools, where Matched[i] is True if we found a good match for vertex i on the source mesh

    Returns:
        W_inpainted: #V2 by num_bones, final skinning weights where we inpainted weights for all vertices i where Matched[i] == False
    """
    
    if point_cloud:
        L, M = robust_laplacian.point_cloud_laplacian(V2)
    else:
        L, M = robust_laplacian.mesh_laplacian(V2, F2)
    L = -L # igl and robust_laplacian have different laplacian conventions
    
    diag = M.diagonal()
    # Guard against zeros in the mass matrix diagonal (degenerate/isolated vertices).
    diag = np.asarray(diag).ravel()
    diag = np.maximum(diag, 1e-20)
    Minv = sp.sparse.diags(1.0 / diag)

    Q2 = -L + L*Minv*L
    Q2 = Q2.astype(np.float64)

    Aeq = sp.sparse.csc_matrix((0, 0), dtype=np.float64)
    Beq = np.array([], dtype=np.float64)
    B = np.zeros(shape = (L.shape[0], W2.shape[1]), dtype=np.float64)

    b = np.array(range(0, int(V2.shape[0])), dtype=np.int64)
    b = b[Matched]
    bc = W2[Matched,:].astype(np.float64)
    result, W_inpainted = igl.min_quad_with_fixed(Q2, B, b, bc, Aeq, Beq, True)
    W_inpainted = W_inpainted.astype(np.float32)
    # when W2 shape = (num_verts, 1), it gets flattened to (num_verts, )
    # reshape it back to initial shape, limit_mask expects 2d array
    if result:
        W_inpainted = W_inpainted.reshape(W2.shape)
    return result, W_inpainted # TODO: Add results
    
    
def limit_mask(weights, adjacency_matrix, dilation_repeat=5, limit_num=4):
    if weights.shape[1] <= limit_num: return np.zeros_like(weights)
    
    count = np.count_nonzero(weights, axis=1)
    to_limit = count > limit_num
    k = weights.shape[1] - limit_num
    weights_inds = np.argpartition(weights, kth=k, axis=1)[:, :k]
    row_indices = np.arange(weights.shape[0])[:, None]
    erode_mask = np.zeros_like(weights, dtype=bool)
    erode_mask[row_indices, weights_inds] = True
    erode_mask = np.logical_and(erode_mask, to_limit[:, np.newaxis])
    erode_mask = sp.sparse.csr_array(erode_mask).astype(np.float32)
    adj_mat = adjacency_matrix
    degrees = adj_mat.sum(axis=1)
    smooth_mat = (1/degrees[:, np.newaxis]) * adj_mat
    for _ in range(dilation_repeat):
        avg_weights = smooth_mat @ erode_mask
        erode_mask = erode_mask.maximum(avg_weights)
    
    return erode_mask.toarray()


def smooth_weigths(
    verts,
    weights,
    matched,
    adjacency_matrix,
    adjacency_list,
    num_smooth_iter_steps,
    smooth_alpha,
    _distance_threshold_unused=None,
):
    """Smooth weights around unmatched vertices.

    Semantics (v9):
    - Propagation scope is defined by BOTH:
        * topology hops (MAX_SMOOTH_TOPOLOGY_DEPTH)
        * spatial radius (MAX_SMOOTH_SPATIAL_DISTANCE)
    - Safety limits:
        * MAX_NEIGHBOR_SEARCH_DEPTH
        * MAX_VERTICES_PER_NEIGHBORHOOD
    - The unmatched seed vertices themselves MUST be included in the scope.
    """
    V = np.asarray(verts, dtype=np.float64)
    W = np.asarray(weights, dtype=np.float32)
    matched = np.asarray(matched, dtype=bool)
    n = V.shape[0]
    if n == 0:
        return W

    not_matched = ~matched
    seed_ids = np.nonzero(not_matched)[0]

    scope = np.zeros((n,), dtype=bool)

    max_depth = int(T.MAX_SMOOTH_TOPOLOGY_DEPTH)
    max_spatial = float(T.MAX_SMOOTH_SPATIAL_DISTANCE)
    max_search_depth = int(T.MAX_NEIGHBOR_SEARCH_DEPTH)
    max_vertices = int(T.MAX_VERTICES_PER_NEIGHBORHOOD)

    # Build scope via bounded BFS from each unmatched seed
    for seed in seed_ids:
        seed = int(seed)
        if seed < 0 or seed >= n:
            continue

        scope[seed] = True  # ensure seed included (fixes previous logic gap)
        visited = {seed}
        queue = [(seed, 0)]
        expanded = 0

        while queue:
            vid, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            if depth >= max_search_depth:
                continue

            if vid >= len(adjacency_list):
                continue
            neigh = adjacency_list[vid]
            for nn in neigh:
                nn = int(nn)
                if nn < 0 or nn >= n:
                    continue
                if nn in visited:
                    continue

                # spatial gate: distance to SEED (not incremental)
                if max_spatial > 0.0:
                    if np.linalg.norm(V[seed, :] - V[nn, :]) > max_spatial:
                        continue

                visited.add(nn)
                scope[nn] = True
                queue.append((nn, depth + 1))
                expanded += 1
                if expanded >= max_vertices:
                    queue.clear()
                    break

    # Laplacian smoothing restricted to scope
    adj_mat = adjacency_matrix.astype(np.float32)
    degrees = np.asarray(adj_mat.sum(axis=1)).reshape(-1)
    degrees = np.maximum(degrees, 1e-12)  # numerical-stability safeguard (excluded from v9 semantics table)

    smooth_mat = sp.sparse.diags(1.0 / degrees) @ adj_mat

    iters = int(num_smooth_iter_steps)
    iters = max(0, min(iters, int(T.MAX_SMOOTH_REPEAT)))
    alpha = float(smooth_alpha)

    weights_smoothed = sp.sparse.csr_array(W)
    for _ in range(iters):
        weights_smoothed = (1.0 - alpha) * weights_smoothed + alpha * (smooth_mat @ weights_smoothed)
        # keep outside-scope weights fixed
        weights_smoothed[~scope] = W[~scope]

    return np.asarray(weights_smoothed.todense(), dtype=np.float32)