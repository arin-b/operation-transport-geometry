from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

from otg.core.data import NodeData


@dataclass
class CostPack:
    cost: np.ndarray
    geometric: np.ndarray
    output: np.ndarray
    risk: np.ndarray
    invariance_penalty: np.ndarray
    operational_coordinate: np.ndarray
    nuisance: np.ndarray


def _normalize(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    scale = np.nanmedian(M[M > 0]) if np.any(M > 0) else 1.0
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    return M / (scale + eps)


def _pair_l2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return cdist(A.reshape(len(A), -1), B.reshape(len(B), -1), metric="euclidean")


def build_operational_cost(node: NodeData, cfg: dict) -> CostPack:
    cost_cfg = cfg.get("cost", {})
    mode = cost_cfg.get("mode", "full")

    z_dist = _normalize(_pair_l2(node.z_a, node.z_b))
    y_dist = _normalize(_pair_l2(node.y_a, node.y_b))
    r_dist = _normalize(np.abs(node.used_risk_a[:, None] - node.used_risk_b[None, :]))
    op_dist = _normalize(_pair_l2(node.op_coords_a, node.op_coords_b))
    nuisance_dist = _normalize(_pair_l2(node.nuisance_a, node.nuisance_b))
    desc_dist = _normalize(_pair_l2(node.descriptors_a, node.descriptors_b))
    anchor_mismatch = (node.anchors_a[:, None] != node.anchors_b[None, :]).astype(float)
    invariance_penalty = op_dist + 0.25 * desc_dist + anchor_mismatch

    weights = cost_cfg.get("weights", {})
    alpha = float(weights.get("geometry", 1.0))
    beta = float(weights.get("output", 1.0))
    gamma = float(weights.get("risk", 2.0))
    lam = float(weights.get("invariance", 3.0))
    opw = float(weights.get("operational_coordinate", 1.5))
    nuisance_w = float(weights.get("nuisance", 0.15))

    if mode == "geometry":
        total = alpha * z_dist
    elif mode == "geometry_risk":
        total = alpha * z_dist + gamma * r_dist
    elif mode == "geometry_output_risk":
        total = alpha * z_dist + beta * y_dist + gamma * r_dist
    elif mode == "operational_only":
        total = opw * op_dist + gamma * r_dist + beta * y_dist
    elif mode == "full":
        total = (
            alpha * z_dist
            + beta * y_dist
            + gamma * r_dist
            + lam * invariance_penalty
            + nuisance_w * nuisance_dist
        )
    else:
        raise ValueError(f"Unknown cost mode: {mode}")

    return CostPack(
        cost=np.asarray(total, dtype=float),
        geometric=z_dist,
        output=y_dist,
        risk=r_dist,
        invariance_penalty=invariance_penalty,
        operational_coordinate=op_dist,
        nuisance=nuisance_dist,
    )
