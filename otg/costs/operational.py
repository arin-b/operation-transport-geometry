from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist

from otg.core.schema import NodeBatch, InvarianceOutput, CostOutput


def _normalize(M: np.ndarray) -> np.ndarray:
    positive = M[M > 0]
    scale = np.median(positive) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return M / scale


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
    elif mode == "full":
        total = (
            w["geometry"] * geometry
            + w["terminal"] * terminal
            + w["risk"] * risk
            + w["invariance"] * invariance
            + w["nuisance"] * nuisance
        )
    else:
        raise ValueError(f"Unknown cost mode: {mode}")

    return CostOutput(total=total, geometry=geometry, terminal=terminal, risk=risk, invariance=invariance, mode=mode, weights=w)
