from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist

from otg.core.schema import NodeBatch, InvarianceOutput, CostOutput
from otg.core.adaptive import adaptive_node_pair_weights


def _normalize(M: np.ndarray) -> np.ndarray:
    positive = M[M > 0]
    scale = np.median(positive) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return M / scale


def _adaptive_operational_factors(node: NodeBatch, op: np.ndarray, nuisance: np.ndarray) -> tuple[float, float, float]:
    # Legacy local-adaptive mode retained for compatibility. It is intentionally
    # not the preferred OTG mode because it can boost operational penalties under
    # large nuisance drift. Use domain_pair_adaptive for proposal-aligned scoring.
    op_positive = op[op > 0]
    nuisance_positive = nuisance[nuisance > 0]
    op_scale = float(np.median(op_positive)) if op_positive.size else 1.0
    nuisance_scale = float(np.median(nuisance_positive)) if nuisance_positive.size else 0.0
    nuisance_ratio = nuisance_scale / max(op_scale, 1e-12)
    risk_shift = abs(float(np.mean(node.used_risk_b) - np.mean(node.used_risk_a)))
    op_boost = 1.0 + min(2.0, nuisance_ratio + 0.75 * risk_shift)
    risk_boost = 1.0 + min(2.5, 2.5 * risk_shift + 0.5 * nuisance_ratio)
    geometry_downweight = 1.0 / (1.0 + nuisance_ratio)
    return op_boost, risk_boost, geometry_downweight


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


def _mean_l2_gap(A: np.ndarray, B: np.ndarray) -> float:
    A = np.asarray(A, dtype=float).reshape(len(A), -1)
    B = np.asarray(B, dtype=float).reshape(len(B), -1)
    return float(np.linalg.norm(np.mean(A, axis=0) - np.mean(B, axis=0)))


def _domain_pair_adaptive_weights(node: NodeBatch, base: dict[str, float], cfg: dict) -> dict[str, float]:
    """Compute pair-specific OTG cost weights from terminally induced shifts.

    This implements the intended low-level rule: if two deployment domains differ
    mainly by nuisance realization but have near-identical terminal behavior and
    node-wise risk, the operational/risk/invariance terms are relaxed. If the
    terminal/risk/failure signals move, the cost reverts toward the conservative
    OTG weights. No labels such as "viewpoint" or "glare" are used here; the
    gate is induced by the deployed system's terminal behavior.
    """
    acfg = cfg.get("cost", {}).get("adaptive", {})
    risk_shift = abs(float(np.mean(node.used_risk_b) - np.mean(node.used_risk_a)))
    true_risk_shift = abs(float(np.mean(node.true_risk_b) - np.mean(node.true_risk_a)))
    terminal_shift = _mean_l2_gap(node.terminal_a, node.terminal_b)
    failure_shift = abs(float(np.mean(node.failure_b.astype(float)) - np.mean(node.failure_a.astype(float))))
    op_shift = _mean_l2_gap(node.operational_coords_a, node.operational_coords_b)
    nuisance_shift = _mean_l2_gap(node.nuisance_coords_a, node.nuisance_coords_b)

    risk_threshold = float(acfg.get("risk_threshold", 0.060))
    risk_scale = float(acfg.get("risk_scale", 0.040))
    terminal_threshold = float(acfg.get("terminal_threshold", 0.100))
    terminal_scale = float(acfg.get("terminal_scale", 0.060))
    failure_threshold = float(acfg.get("failure_threshold", 0.050))
    failure_scale = float(acfg.get("failure_scale", 0.040))
    gate_bias = float(acfg.get("gate_bias", -0.50))

    gate_weights = acfg.get("gate_weights", {})
    wr = float(gate_weights.get("risk", 1.60))
    wt = float(gate_weights.get("terminal", 1.00))
    wf = float(gate_weights.get("failure", 0.80))

    risk_term = (risk_shift - risk_threshold) / max(risk_scale, 1e-12)
    terminal_term = (terminal_shift - terminal_threshold) / max(terminal_scale, 1e-12)
    failure_term = (failure_shift - failure_threshold) / max(failure_scale, 1e-12)
    raw_gate = gate_bias + wr * risk_term + wt * terminal_term + wf * failure_term
    gate = _sigmoid(raw_gate)

    # Large nuisance-only shifts should relax costs further only when terminal/risk
    # evidence says the shift is harmless. This protects clear-vs-viewpoint while
    # leaving glare/occlusion activated.
    nuisance_ratio = nuisance_shift / max(op_shift + terminal_shift + risk_shift, 1e-12)
    harmless_nuisance = (1.0 - gate) * _sigmoid(float(acfg.get("nuisance_gate_slope", 0.75)) * (nuisance_ratio - float(acfg.get("nuisance_ratio_threshold", 1.0))))

    min_weights = acfg.get("min_weights", {})
    mins = {
        "geometry": float(min_weights.get("geometry", 0.35)),
        "terminal": float(min_weights.get("terminal", 0.10)),
        "risk": float(min_weights.get("risk", 0.20)),
        "invariance": float(min_weights.get("invariance", 0.15)),
        "operational_coordinate": float(min_weights.get("operational_coordinate", 0.25)),
        "nuisance": float(min_weights.get("nuisance", 0.02)),
    }
    eff = {}
    for key, b in base.items():
        lo = min(float(b), float(mins.get(key, b)))
        eff[key] = lo + gate * (float(b) - lo)

    # Extra quotient-like relaxation for nuisance-only geometry; this is kept
    # continuous and data-dependent, not domain-name-specific.
    quotient_strength = float(acfg.get("quotient_nuisance_strength", 0.35))
    relaxation = 1.0 - quotient_strength * harmless_nuisance
    eff["geometry"] *= relaxation
    eff["nuisance"] *= relaxation

    eff.update({
        "adaptive_gate": gate,
        "adaptive_raw_gate": float(raw_gate),
        "adaptive_risk_shift": risk_shift,
        "adaptive_true_risk_shift": true_risk_shift,
        "adaptive_terminal_shift": terminal_shift,
        "adaptive_failure_shift": failure_shift,
        "adaptive_operational_shift": op_shift,
        "adaptive_nuisance_shift": nuisance_shift,
        "adaptive_nuisance_ratio": float(nuisance_ratio),
        "adaptive_harmless_nuisance": float(harmless_nuisance),
    })
    return eff


def build_cost(node: NodeBatch, inv: InvarianceOutput, cfg: dict) -> CostOutput:
    mode = cfg.get("cost", {}).get("mode", "full")
    weights = cfg.get("cost", {}).get("weights", {})
    w = {
        "geometry": float(weights.get("geometry", 1.0)),
        "terminal": float(weights.get("terminal", weights.get("output", 1.0))),
        "risk": float(weights.get("risk", 2.0)),
        "invariance": float(weights.get("invariance", 3.0)),
        "operational_coordinate": float(weights.get("operational_coordinate", 1.5)),
        "nuisance": float(weights.get("nuisance", 0.15)),
    }

    geometry = _normalize(cdist(node.z_a, node.z_b))
    terminal = _normalize(cdist(node.terminal_a.reshape(len(node.terminal_a), -1), node.terminal_b.reshape(len(node.terminal_b), -1)))
    risk = _normalize(np.abs(node.used_risk_a[:, None] - node.used_risk_b[None, :]))
    op = _normalize(cdist(node.operational_coords_a, node.operational_coords_b))
    nuisance = _normalize(cdist(node.nuisance_coords_a, node.nuisance_coords_b))
    invariance = _normalize(inv.penalty + 0.25 * inv.soft_score)

    if mode == "geometry":
        total = w["geometry"] * geometry
    elif mode == "geometry_risk":
        total = w["geometry"] * geometry + w["risk"] * risk
    elif mode == "geometry_output_risk":
        total = w["geometry"] * geometry + w["terminal"] * terminal + w["risk"] * risk
    elif mode == "operational_only":
        total = w["operational_coordinate"] * op + w["terminal"] * terminal + w["risk"] * risk
    elif mode == "adaptive_operational":
        op_boost, risk_boost, geometry_downweight = _adaptive_operational_factors(node, op, nuisance)
        total = (
            (w["geometry"] * geometry_downweight) * geometry
            + w["terminal"] * terminal
            + (w["risk"] * risk_boost) * risk
            + (w["operational_coordinate"] * op_boost) * op
            + (w["nuisance"] * geometry_downweight) * nuisance
        )
        w.update({"legacy_op_boost": op_boost, "legacy_risk_boost": risk_boost, "legacy_geometry_downweight": geometry_downweight})
    elif mode in {"domain_pair_adaptive", "adaptive_node_pair"}:
        # Final OTG low-level profile: node-sensitive/domain-pair adaptive
        # operational geometry. The legacy name domain_pair_adaptive remains as
        # an alias for backward-compatible configs.
        w_eff = adaptive_node_pair_weights(node, w, cfg)
        total = (
            w_eff["geometry"] * geometry
            + w_eff["terminal"] * terminal
            + w_eff["risk"] * risk
            + w_eff["invariance"] * invariance
            + w_eff["operational_coordinate"] * op
            + w_eff["nuisance"] * nuisance
        )
        w = w_eff
    elif mode == "full":
        total = (
            w["geometry"] * geometry
            + w["terminal"] * terminal
            + w["risk"] * risk
            + w["invariance"] * invariance
            + w["operational_coordinate"] * op
            + w["nuisance"] * nuisance
        )
    else:
        raise ValueError(f"Unknown cost mode: {mode}")

    return CostOutput(total=total, geometry=geometry, terminal=terminal, risk=risk, invariance=invariance, mode=mode, weights=w)
