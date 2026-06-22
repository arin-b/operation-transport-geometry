from __future__ import annotations

from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist
import numpy as np

from otg.core.schema import NodeBatch, InvarianceOutput


def normalize_matrix(M: np.ndarray) -> np.ndarray:
    positive = M[M > 0]
    scale = np.median(positive) if positive.size else 1.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return M / scale


def pair_l2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A.reshape(len(A), -1)
    B = B.reshape(len(B), -1)
    return cdist(A, B)


class InvarianceBuilder(ABC):
    mode: str

    def __init__(self, cfg: dict, seed_bank):
        self.cfg = cfg
        self.seed_bank = seed_bank

    @abstractmethod
    def build(self, node: NodeBatch) -> InvarianceOutput:
        raise NotImplementedError

    def components(self, node: NodeBatch) -> dict[str, np.ndarray]:
        risk_gap = np.abs(node.used_risk_a[:, None] - node.used_risk_b[None, :])
        true_risk_gap = np.abs(node.true_risk_a[:, None] - node.true_risk_b[None, :])
        op_gap = pair_l2(node.operational_coords_a, node.operational_coords_b)
        nuis_gap = pair_l2(node.nuisance_coords_a, node.nuisance_coords_b) if node.nuisance_coords_a.size and node.nuisance_coords_b.size else np.zeros_like(risk_gap)
        desc_gap = pair_l2(node.descriptor_coords_a, node.descriptor_coords_b)
        anchor_gap = (node.anchors_a[:, None] != node.anchors_b[None, :]).astype(float)
        return {
            "risk_gap": risk_gap,
            "true_risk_gap": true_risk_gap,
            "operational_gap": op_gap,
            "nuisance_gap": nuis_gap,
            "descriptor_gap": desc_gap,
            "anchor_gap": anchor_gap,
            "risk_gap_norm": normalize_matrix(risk_gap),
            "true_risk_gap_norm": normalize_matrix(true_risk_gap),
            "operational_gap_norm": normalize_matrix(op_gap),
            "nuisance_gap_norm": normalize_matrix(nuis_gap),
            "descriptor_gap_norm": normalize_matrix(desc_gap),
        }


def feasibility_metadata(allowed: np.ndarray) -> dict:
    row_has = allowed.any(axis=1)
    col_has = allowed.any(axis=0)
    return {
        "allowed_fraction": float(allowed.mean()),
        "row_feasible_fraction": float(row_has.mean()),
        "col_feasible_fraction": float(col_has.mean()),
        "num_infeasible_rows": int((~row_has).sum()),
        "num_infeasible_cols": int((~col_has).sum()),
    }
