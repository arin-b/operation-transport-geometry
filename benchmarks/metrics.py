from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


PRIMARY_CLAIMS = [
    "harmless_operational_collapse",
    "harmful_operational_separation",
    "admissibility_respected",
    "false_collapse_avoided",
    "dangerous_unmatched_exposed",
    "system_level_localization",
]

SECONDARY_CLAIMS = [
    "numerical_feasibility",
    "risk_estimation_quality",
    "sample_stability_proxy",
]


def gini_like(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0 or np.allclose(x.sum(), 0):
        return 0.0
    x = np.sort(np.maximum(x, 0.0))
    n = x.size
    return float((2 * np.sum(np.arange(1, n + 1) * x) / (n * np.sum(x))) - (n + 1) / n)


def normalize_cost(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    positive = M[np.isfinite(M) & (M > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return M / scale


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _positive_scale(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 1e-12)]
    if arr.size == 0:
        return 1.0
    scale = float(np.median(arr))
    return scale if scale > 1e-12 else 1.0


def legacy_composite_score(metrics: dict[str, float | int | bool]) -> float:
    """Old scalar benchmark score retained only for backward compatibility.

    Reports no longer rank methods primarily by this number; graph-level benchmark
    reports claim-specific scores over the full domain/node tensor.
    """
    risk_mae = _clip01(1.0 - float(metrics.get("risk_estimation_mae", 1.0)) / 0.25)
    collapse = _clip01(float(metrics.get("harmless_shift_collapse", 0.0)))
    detect = _clip01(float(metrics.get("harmful_shift_detection", 0.0)))
    feasible = np.mean([
        _clip01(float(metrics.get("allowed_pair_fraction", 0.0)) / 0.25),
        _clip01(float(metrics.get("row_feasible_fraction", 0.0))),
        _clip01(float(metrics.get("col_feasible_fraction", 0.0))),
    ])
    penalties = np.mean([
        _clip01(1.0 - float(metrics.get("false_collapse_mass", 0.0)) / 0.15),
        _clip01(1.0 - float(metrics.get("transport_forbidden_mass", 0.0)) / 0.01),
        _clip01(1.0 - float(metrics.get("transport_row_l1_error", 0.0)) / 0.05),
        _clip01(1.0 - float(metrics.get("transport_col_l1_error", 0.0)) / 0.05),
    ])
    score = 100.0 * (0.25 * risk_mae + 0.20 * collapse + 0.20 * detect + 0.15 * feasible + 0.20 * penalties)
    return float(score)


# Backward-compatible name for old callers.
composite_score = legacy_composite_score


def pair_key(a: str, b: str) -> str:
    return f"{a}::{b}"


def _matrix_get(M: np.ndarray, domain_order: list[str], a: str, b: str) -> float | None:
    idx = {d: i for i, d in enumerate(domain_order)}
    if a not in idx or b not in idx:
        return None
    return float(M[idx[a], idx[b]])


def _mean_existing(vals: list[float | None], default: float = 0.0) -> float:
    xs = [float(v) for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(xs)) if xs else default


def graph_claim_scores(
    *,
    D_matrix: np.ndarray,
    dangerous_matrix: np.ndarray,
    forbidden_matrix: np.ndarray,
    false_collapse_matrix: np.ndarray,
    row_error_matrix: np.ndarray,
    col_error_matrix: np.ndarray,
    risk_error_matrix: np.ndarray,
    localization_hits: dict[str, bool],
    domain_order: list[str],
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Claim-specific benchmark scores over the whole domain discrepancy matrix.

    This deliberately avoids treating the benchmark as a list of isolated pairs.
    Pairwise OT is only the numerical primitive used to populate full matrices.
    """
    cfg = config or {}
    claims_cfg = cfg.get("benchmark", {}).get("claims", {})
    harmless_pairs = [tuple(p) for p in claims_cfg.get("harmless_pairs", [["clear", "viewpoint_shift"]])]
    harmful_pairs = [tuple(p) for p in claims_cfg.get("harmful_pairs", [["clear", "glare"], ["clear", "occlusion"], ["viewpoint_shift", "occlusion"]])]
    dangerous_target = str(claims_cfg.get("dangerous_target_domain", "occlusion"))

    off_diag = D_matrix[~np.eye(D_matrix.shape[0], dtype=bool)] if D_matrix.size else np.asarray([1.0])
    scale = _positive_scale(off_diag)
    harmless_vals = [_matrix_get(D_matrix, domain_order, a, b) for a, b in harmless_pairs]
    harmful_vals = [_matrix_get(D_matrix, domain_order, a, b) for a, b in harmful_pairs]
    harmless_mean = _mean_existing(harmless_vals, default=0.0)
    harmful_mean = _mean_existing(harmful_vals, default=0.0)

    harmless_collapse = _clip01(1.0 - harmless_mean / (scale + 1e-12))
    harmful_sep = _clip01((harmful_mean - harmless_mean) / (scale + 1e-12))

    danger_vals = []
    for i, a in enumerate(domain_order):
        for j, b in enumerate(domain_order):
            if i != j and (a == dangerous_target or b == dangerous_target):
                danger_vals.append(float(dangerous_matrix[i, j]))
    danger_raw = float(np.max(danger_vals)) if danger_vals else 0.0
    danger_threshold = float(claims_cfg.get("dangerous_unmatched_target", 0.15))
    dangerous_exposed = _clip01(danger_raw / max(danger_threshold, 1e-12))

    forbidden = float(np.nanmax(forbidden_matrix)) if forbidden_matrix.size else 0.0
    forbidden_tol = float(claims_cfg.get("forbidden_mass_tolerance", 1e-4))
    admissibility = _clip01(1.0 - forbidden / max(forbidden_tol, 1e-12))

    false_collapse = float(np.nanmax(false_collapse_matrix)) if false_collapse_matrix.size else 0.0
    false_collapse_tol = float(claims_cfg.get("false_collapse_tolerance", 0.15))
    false_collapse_avoided = _clip01(1.0 - false_collapse / max(false_collapse_tol, 1e-12))

    if localization_hits:
        localization = float(np.mean([1.0 if v else 0.0 for v in localization_hits.values()]))
    else:
        localization = 0.0

    row_err = float(np.nanmean(row_error_matrix)) if row_error_matrix.size else 1.0
    col_err = float(np.nanmean(col_error_matrix)) if col_error_matrix.size else 1.0
    marginal_tol = float(claims_cfg.get("marginal_error_tolerance", 0.05))
    feasibility = _clip01(1.0 - max(row_err, col_err) / max(marginal_tol, 1e-12))

    risk_err = float(np.nanmean(risk_error_matrix)) if risk_error_matrix.size else 1.0
    risk_tol = float(claims_cfg.get("risk_mae_tolerance", 0.25))
    risk_quality = _clip01(1.0 - risk_err / max(risk_tol, 1e-12))

    # A simple matrix-contrast proxy: the full domain geometry should have nonzero
    # spread, but not be dominated by arbitrary numerical blow-up.
    spread = float(np.nanstd(off_diag)) if off_diag.size else 0.0
    stability_proxy = _clip01(1.0 / (1.0 + spread / (scale + 1e-12)))

    scores = {
        "harmless_operational_collapse": harmless_collapse,
        "harmful_operational_separation": harmful_sep,
        "admissibility_respected": admissibility,
        "false_collapse_avoided": false_collapse_avoided,
        "dangerous_unmatched_exposed": dangerous_exposed,
        "system_level_localization": localization,
        "numerical_feasibility": feasibility,
        "risk_estimation_quality": risk_quality,
        "sample_stability_proxy": stability_proxy,
        "harmless_mean_D_op": float(harmless_mean),
        "harmful_mean_D_op": float(harmful_mean),
        "dangerous_unmatched_raw": danger_raw,
        "max_forbidden_mass": forbidden,
        "max_false_collapse_mass": false_collapse,
        "mean_row_marginal_error": row_err,
        "mean_col_marginal_error": col_err,
        "mean_risk_estimation_mae": risk_err,
    }
    scores["primary_claim_average_diagnostic"] = float(np.mean([scores[k] for k in PRIMARY_CLAIMS]))
    scores["secondary_average_diagnostic"] = float(np.mean([scores[k] for k in SECONDARY_CLAIMS]))
    return scores
