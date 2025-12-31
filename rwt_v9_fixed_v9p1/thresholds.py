"""Semantic parameter/threshold registry (v9).

Notes:
- Values here are DEFAULTS only. UI-exposed properties may override these.
- This module intentionally does NOT contain numerical-stability epsilons (1e-12, 1e-20),
  nor debug/visualization/performance-only safeguards.
"""

# -----------------------------
# Class 1 — UI-adjustable (user semantics)
# -----------------------------
MAX_MATCH_DISTANCE: float = 0.05
MAX_NORMAL_ANGLE_DEG: float = 30.0
NORMAL_ANGLE_OFFSET_DEG: float = 0.0


# -----------------------------
# Class 1b — Supergroup (UI-adjustable semantics)
# -----------------------------
# Supergroup merge thresholds are built in WORLD units from mesh sampling density (d_nn) and user scales.
# eps_force_raw = d_nn * (10**A) * SUPERGROUP_EPS_FORCE_BASE
# eps_norm_raw  = d_nn * (10**B) * SUPERGROUP_EPS_NORM_BASE
# # then clamped: eps = clamp(eps_raw, max(SUPERGROUP_EPS_ABS_MIN, d_nn*SUPERGROUP_EPS_REL_MIN), d_nn*SUPERGROUP_EPS_REL_MAX)
SUPERGROUP_EPS_FORCE_BASE: float = 1e-7
SUPERGROUP_EPS_NORM_BASE: float = 1e-3

# Face-cleaning height threshold base (world):
# h_eps_raw = bbox_diag * (10**c) * FACE_CLEAN_HEIGHT_BASE
# Then (optionally) clamped by the same REL_MIN/REL_MAX window when Strong Clean is enabled.
FACE_CLEAN_HEIGHT_BASE: float = 1e-7

# UI defaults (world units). These are object-scale-dependent in practice; the UI shows fill ranges per-object.
SUPERGROUP_EPS_MINSCALE_DEFAULT: float = SUPERGROUP_EPS_FORCE_BASE
SUPERGROUP_EPS_MIDSCALE_DEFAULT: float = SUPERGROUP_EPS_NORM_BASE
FACE_CLEAN_HEIGHT_DEFAULT: float = FACE_CLEAN_HEIGHT_BASE


SUPERGROUP_EPS_ABS_MIN: float = 1e-12
# Relative clamp window (in multiples of d_nn).
# Default lower bound tightened per UI request.
SUPERGROUP_EPS_REL_MIN: float = 0.02
SUPERGROUP_EPS_REL_MAX: float = 8.0

SUPERGROUP_LOG10_A_DEFAULT: float = 0.0  # A = lg(a) where a is force-merge scale
SUPERGROUP_LOG10_B_DEFAULT: float = 0.0  # B = lg(b) where b is normal-merge scale

SUPERGROUP_MIX_ALPHA: float = 0.7
SUPERGROUP_SPATIAL_RADIUS_DEFAULT: float = 0.0  # 0 -> auto from bbox (optional)

# Spatial adjacency between supergroup representative points (internal; not exposed in UI).
# We intentionally avoid a user-facing "spatial radius" parameter.
SUPERGROUP_SPATIAL_K: int = 16

SUPERGROUP_THETA_MERGE_DEG: float = 60.0

SMOOTHING_STRENGTH: float = 0.2
SMOOTHING_ITERATIONS: int = 4

TRANSFER_NEW_WEIGHT_RATIO: float = 0.9

INPAINT_ENABLE: bool = True

# -----------------------------
# Class 2 — Topology / propagation semantics
# -----------------------------
# 2.1 Neighborhood / propagation (core)
MAX_SMOOTH_TOPOLOGY_DEPTH: int = 3
MAX_SMOOTH_SPATIAL_DISTANCE: float = 0.05
MAX_NEIGHBOR_SEARCH_DEPTH: int = 12
MAX_VERTICES_PER_NEIGHBORHOOD: int = 500

# 2.2 Inpaint / blending semantics
EPS_MASK_FOR_INPAINT: float = 1e-6
INPAINT_FALLOFF_MODE: str = "LINEAR"  # "LINEAR" | "SMOOTHSTEP" | "BINARY"
INPAINT_ALLOW_UNMATCHED: bool = True

# 2.3 Match reliability / reject logic
SUPERGROUP_REJECT_RATIO: float = 0.6
MIN_MATCH_CONFIDENCE: float = 1e-4
ALLOW_FLIPPED_NORMAL: bool = False

# -----------------------------
# Class 3 — Discrete/angle controls
# -----------------------------
MAX_GROUP_SIZE: int = 1000
MIN_GROUP_SIZE: int = 1

MAX_SMOOTH_REPEAT: int = 25
NORMAL_SMOOTH_REPEAT: int = 0
ANGLE_SMOOTH_THRESHOLD_DEG: float = 45.0


def clamp_int(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def apply_inpaint_falloff(mask):
    """Optional remapping of mask before final blend.

    Currently not wired into the main pipeline by default; provided for
    centralized semantics when you choose to enable different modes.
    """
    import numpy as _np
    m = _np.asarray(mask, dtype=_np.float32).copy()
    if INPAINT_FALLOFF_MODE == "LINEAR":
        return _np.clip(m, 0.0, 1.0)
    if INPAINT_FALLOFF_MODE == "SMOOTHSTEP":
        m = _np.clip(m, 0.0, 1.0)
        return m * m * (3.0 - 2.0 * m)
    if INPAINT_FALLOFF_MODE == "BINARY":
        return (m >= 0.5).astype(_np.float32)
    return _np.clip(m, 0.0, 1.0)
