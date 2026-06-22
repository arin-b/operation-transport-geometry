from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from otg.core.costs import CostPack
from otg.core.data import NodeData


@dataclass
class AdmissibilityPack:
    allowed: np.ndarray
    penalty: np.ndarray
    mode: str
    threshold_report: dict
    risk_gap: np.ndarray
    op_gap: np.ndarray
    nuisance_gap: np.ndarray
    descriptor_gap: np.ndarray
    anchor_mismatch: np.ndarray


def _norm_gap(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.ndim == 1:
        A = A[:, None]
    if B.ndim == 1:
        B = B[:, None]
    return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)


def _normalize(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    med = np.nanmedian(M[M > 0]) if np.any(M > 0) else 1.0
    if not np.isfinite(med) or med < eps:
        med = 1.0
    return M / (med + eps)


def build_admissibility(node: NodeData, cost_pack: CostPack, cfg: dict) -> AdmissibilityPack:
    adm_cfg = cfg.get("admissibility", {})
    mode = adm_cfg.get("mode", "hybrid")

    risk_tol = float(adm_cfg.get("risk_tolerance", 0.25))
    op_tol = float(adm_cfg.get("op_tolerance", 0.65))
    nuisance_tol = float(adm_cfg.get("nuisance_tolerance", 3.0))
    descriptor_tol = float(adm_cfg.get("descriptor_tolerance", 1.25))
    severe_risk_tol = float(adm_cfg.get("severe_risk_tolerance", 0.65))

    risk_gap = np.abs(node.used_risk_a[:, None] - node.used_risk_b[None, :])
    op_gap = _normalize(_norm_gap(node.op_coords_a, node.op_coords_b))
    nuisance_gap = _normalize(_norm_gap(node.nuisance_a, node.nuisance_b))
    descriptor_gap = _normalize(_norm_gap(node.descriptors_a, node.descriptors_b))
    anchor_mismatch = (node.anchors_a[:, None] != node.anchors_b[None, :]).astype(float)

    # Invariance means nuisance may move, but operational/risk behavior must be compatible.
    soft_score = (
        risk_gap / max(risk_tol, 1e-12)
        + op_gap / max(op_tol, 1e-12)
        + 0.25 * descriptor_gap / max(descriptor_tol, 1e-12)
        + anchor_mismatch
    )

    base_allowed = (risk_gap <= risk_tol) & (op_gap <= op_tol) & (descriptor_gap <= descriptor_tol + nuisance_tol)
    severe_allowed = (risk_gap <= severe_risk_tol) & (op_gap <= 2.0 * op_tol)

    if mode == "hard":
        allowed = base_allowed & (anchor_mismatch == 0)
        penalty = np.zeros_like(cost_pack.cost)
    elif mode == "soft":
        allowed = np.ones_like(base_allowed, dtype=bool)
        penalty = soft_score
    elif mode == "hybrid":
        allowed = severe_allowed
        penalty = soft_score * (~base_allowed)
    elif mode == "adaptive":
        n = len(node.z_a) + len(node.z_b)
        sample_factor = np.sqrt(200.0 / max(n, 1))
        risk_unc = float(node.metadata.get("risk_estimation_error", 0.0)) if node.metadata else 0.0
        adaptive_risk_tol = risk_tol + 0.5 * risk_unc + 0.08 * sample_factor
        adaptive_op_tol = op_tol + 0.12 * sample_factor
        allowed = (risk_gap <= adaptive_risk_tol) & (op_gap <= adaptive_op_tol)
        penalty = soft_score / (1.0 + risk_unc + sample_factor)
    else:
        raise ValueError(f"Unknown admissibility mode: {mode}")

    return AdmissibilityPack(
        allowed=np.asarray(allowed, dtype=bool),
        penalty=np.asarray(penalty, dtype=float),
        mode=mode,
        threshold_report={
            "risk_tolerance": risk_tol,
            "op_tolerance": op_tol,
            "nuisance_tolerance": nuisance_tol,
            "descriptor_tolerance": descriptor_tol,
            "allowed_fraction": float(np.mean(allowed)),
        },
        risk_gap=risk_gap,
        op_gap=op_gap,
        nuisance_gap=nuisance_gap,
        descriptor_gap=descriptor_gap,
        anchor_mismatch=anchor_mismatch,
    )
