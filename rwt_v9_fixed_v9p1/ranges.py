"""UI-only reference range + clamp estimates for supergroup parameters.

This module must not be required by the execution path.

Key idea:
- User inputs (MINBAND/MIDBAND/Face cleaning) are raw world-unit thresholds.
- If REL clamp is enabled, the *effective* threshold varies per-vertex based on
  local NN distance. UI can only *estimate* this variability from samples.
- "Calculate prob region" computes a reference [min,max] interval that reflects
  multiple constraints, but it never mutates fill behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import math
import numpy as np


@dataclass
class RangeResult:
    min_allowed: float
    max_allowed: float
    clamp_estimate_min: Optional[float] = None
    clamp_estimate_max: Optional[float] = None
    note: Optional[str] = None


@dataclass
class ClampContext:
    rel_min: float
    rel_max: float
    eps_abs_min: float = 0.0
    nn_distance_samples: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))


def build_clamp_context(
    *,
    verts_world: np.ndarray,
    rel_min: float,
    rel_max: float,
    eps_abs_min: float,
    sample_n: int = 500,
) -> ClampContext:
    """Sample local nearest-neighbor distances on the target mesh (UI-only)."""
    V = np.asarray(verts_world, dtype=np.float64)
    n_v = int(V.shape[0])

    ctx = ClampContext(rel_min=float(rel_min), rel_max=float(rel_max), eps_abs_min=float(eps_abs_min))
    if n_v <= 1:
        ctx.nn_distance_samples = np.asarray([], dtype=np.float64)
        return ctx

    m = int(max(2, min(n_v, int(sample_n))))
    if m < n_v:
        idx = np.linspace(0, n_v - 1, m, dtype=np.int64)
        V_s = V[idx]
    else:
        V_s = V

    try:
        from scipy.spatial import cKDTree  # type: ignore

        kdt = cKDTree(V_s)
        dists, _ = kdt.query(V_s, k=2)
        nn = dists[:, 1] if dists.ndim == 2 and dists.shape[1] >= 2 else np.asarray([], dtype=np.float64)
        ctx.nn_distance_samples = np.asarray(nn, dtype=np.float64)
        return ctx
    except Exception:
        pass

    diff = V_s[:, None, :] - V_s[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.min(d2, axis=1))
    ctx.nn_distance_samples = np.asarray(nn, dtype=np.float64)
    return ctx


def _clamp_bounds_from_samples(ctx: ClampContext) -> Tuple[np.ndarray, np.ndarray]:
    d = np.asarray(ctx.nn_distance_samples, dtype=np.float64).reshape(-1)
    if d.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    eps_min = np.maximum(float(ctx.eps_abs_min), d * float(ctx.rel_min))
    eps_max = d * float(ctx.rel_max)
    eps_max = np.maximum(eps_max, eps_min)  # avoid inverted windows on degenerate samples
    return eps_min, eps_max


def clamp_bounds_percentiles(
    ctx: ClampContext,
    p_lo: float = 5.0,
    p_hi: float = 95.0,
) -> Tuple[float, float]:
    """Return a typical [eps_min, eps_max] window from samples."""
    eps_min, eps_max = _clamp_bounds_from_samples(ctx)
    if eps_min.size == 0:
        return float(ctx.eps_abs_min), float('inf')
    lo = float(np.percentile(eps_min, p_lo))
    hi = float(np.percentile(eps_max, p_hi))
    if hi < lo:
        hi = lo
    return lo, hi


def estimate_clamp_effect_range(
    *,
    raw_value: float,
    clamp_context: ClampContext,
    percentile_lo: float = 5.0,
    percentile_hi: float = 95.0,
) -> Tuple[float, float]:
    """Estimate p05/p95 of per-sample effective values for a given raw input."""
    eps_min, eps_max = _clamp_bounds_from_samples(clamp_context)
    if eps_min.size == 0:
        return float(raw_value), float(raw_value)
    eff = np.maximum(eps_min, np.minimum(float(raw_value), eps_max))
    lo = float(np.percentile(eff, percentile_lo))
    hi = float(np.percentile(eff, percentile_hi))
    if hi < lo:
        hi = lo
    return lo, hi


# Default raw-input reference domains (as a fraction of bbox diagonal).
# - Force/Face: 1e-8 .. 1e-1 (matches 10**[-8,-1] × bbox when base=1e-7)
# - Normal:     1e-4 .. 1    (matches 10**[-4,0] × bbox when base=1e-3)
DOMAIN_RATIO_MIN_FORCE_FACE: float = 1e-8
DOMAIN_RATIO_MAX_FORCE_FACE: float = 1e-1
DOMAIN_RATIO_MIN_MID: float = 1e-4
DOMAIN_RATIO_MAX_MID: float = 1.0


def _intersect(lo: float, hi: float, lo2: float, hi2: float) -> Tuple[float, float]:
    lo3 = max(float(lo), float(lo2))
    hi3 = min(float(hi), float(hi2))
    if hi3 < lo3:
        hi3 = lo3
    return float(lo3), float(hi3)


def calculate_midband_reference_range(
    *,
    bbox_diag: float,
    raw_value: Optional[float] = None,
    clamp_enabled: bool,
    clamp_context: Optional[ClampContext],
    domain_ratio_min: float = DOMAIN_RATIO_MIN_MID,
    domain_ratio_max: float = DOMAIN_RATIO_MAX_MID,
) -> RangeResult:
    base_lo = float(bbox_diag) * float(domain_ratio_min)
    base_hi = float(bbox_diag) * float(domain_ratio_max)
    lo, hi = base_lo, base_hi

    note = f"Base domain: [{domain_ratio_min:g}, {domain_ratio_max:g}] × bbox"
    if clamp_enabled and clamp_context is not None and clamp_context.nn_distance_samples.size:
        c_lo, c_hi = clamp_bounds_percentiles(clamp_context)
        lo, hi = _intersect(lo, hi, c_lo, c_hi)
        note += "; REL clamp window (sampled) applied."
    res = RangeResult(min_allowed=lo, max_allowed=hi, note=note)
    if clamp_enabled and raw_value is not None and clamp_context is not None and clamp_context.nn_distance_samples.size:
        e_lo, e_hi = estimate_clamp_effect_range(raw_value=float(raw_value), clamp_context=clamp_context)
        res.clamp_estimate_min = e_lo
        res.clamp_estimate_max = e_hi
    return res


def calculate_minband_reference_range(
    *,
    bbox_diag: float,
    midband_raw: float,
    raw_value: Optional[float] = None,
    clamp_enabled: bool,
    clamp_context: Optional[ClampContext],
    domain_ratio_min: float = DOMAIN_RATIO_MIN_FORCE_FACE,
    domain_ratio_max: float = DOMAIN_RATIO_MAX_FORCE_FACE,
    hierarchy_div: float = 5.0,
) -> RangeResult:
    base_lo = float(bbox_diag) * float(domain_ratio_min)
    base_hi = float(bbox_diag) * float(domain_ratio_max)

    cap = float(midband_raw) / float(hierarchy_div) if float(hierarchy_div) > 0 else float(midband_raw)
    lo, hi = base_lo, min(base_hi, cap)

    note = f"Base domain: [{domain_ratio_min:g}, {domain_ratio_max:g}] × bbox; cap <= MID/{hierarchy_div:g}"
    if clamp_enabled and clamp_context is not None and clamp_context.nn_distance_samples.size:
        c_lo, c_hi = clamp_bounds_percentiles(clamp_context)
        lo, hi = _intersect(lo, hi, c_lo, c_hi)
        note += "; REL clamp window (sampled) applied."
    res = RangeResult(min_allowed=lo, max_allowed=hi, note=note)
    if clamp_enabled and raw_value is not None and clamp_context is not None and clamp_context.nn_distance_samples.size:
        e_lo, e_hi = estimate_clamp_effect_range(raw_value=float(raw_value), clamp_context=clamp_context)
        res.clamp_estimate_min = e_lo
        res.clamp_estimate_max = e_hi
    return res


def calculate_face_clean_reference_range(
    *,
    bbox_diag: float,
    midband_raw: float,
    raw_value: Optional[float] = None,
    clamp_enabled: bool,
    clamp_context: Optional[ClampContext],
    domain_ratio_min: float = DOMAIN_RATIO_MIN_FORCE_FACE,
    domain_ratio_max: float = DOMAIN_RATIO_MAX_FORCE_FACE,
    hierarchy_div: float = 5.0,
) -> RangeResult:
    base_lo = float(bbox_diag) * float(domain_ratio_min)
    base_hi = float(bbox_diag) * float(domain_ratio_max)

    cap = float(midband_raw) / float(hierarchy_div) if float(hierarchy_div) > 0 else float(midband_raw)
    lo, hi = base_lo, min(base_hi, cap)

    note = f"Base domain: [{domain_ratio_min:g}, {domain_ratio_max:g}] × bbox; cap <= MID/{hierarchy_div:g}"
    if clamp_enabled and clamp_context is not None and clamp_context.nn_distance_samples.size:
        c_lo, c_hi = clamp_bounds_percentiles(clamp_context)
        lo, hi = _intersect(lo, hi, c_lo, c_hi)
        note += "; REL clamp window (sampled) applied."
    res = RangeResult(min_allowed=lo, max_allowed=hi, note=note)
    if clamp_enabled and raw_value is not None and clamp_context is not None and clamp_context.nn_distance_samples.size:
        e_lo, e_hi = estimate_clamp_effect_range(raw_value=float(raw_value), clamp_context=clamp_context)
        res.clamp_estimate_min = e_lo
        res.clamp_estimate_max = e_hi
    return res
