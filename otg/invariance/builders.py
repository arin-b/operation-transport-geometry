from __future__ import annotations

import numpy as np

from otg.core.schema import NodeBatch, InvarianceOutput
from otg.invariance.base import InvarianceBuilder, feasibility_metadata
from otg.invariance.registry import register_invariance


class MixedInvarianceBase(InvarianceBuilder):
    def tolerances(self, node: NodeBatch) -> dict[str, float]:
        cfg = self.cfg.get("admissibility", {})
        return {
            "risk": float(cfg.get("risk_tolerance", 0.25)),
            "operational": float(cfg.get("operational_tolerance", 1.50)),
            "descriptor": float(cfg.get("descriptor_tolerance", 1.50)),
            "anchor": float(cfg.get("anchor_tolerance", 0.0)),
            "severe_risk": float(cfg.get("severe_risk_tolerance", 0.65)),
            "severe_operational": float(cfg.get("severe_operational_tolerance", 3.00)),
        }

    def weighted_score(self, node: NodeBatch) -> tuple[np.ndarray, dict]:
        c = self.components(node)
        tol = self.tolerances(node)
        cfg = self.cfg.get("admissibility", {})
        w_risk = float(cfg.get("risk_weight", 1.0))
        w_op = float(cfg.get("operational_weight", 1.0))
        w_desc = float(cfg.get("descriptor_weight", 0.50))
        w_anchor = float(cfg.get("anchor_weight", 2.0))
        w_nuis = float(cfg.get("nuisance_weight", 0.0))

        score = (
            w_risk * c["risk_gap"] / max(tol["risk"], 1e-12)
            + w_op * c["operational_gap_norm"] / max(tol["operational"], 1e-12)
            + w_desc * c["descriptor_gap_norm"] / max(tol["descriptor"], 1e-12)
            + w_anchor * c["anchor_gap"]
            + w_nuis * c["nuisance_gap_norm"]
        )

        component_allowed = {
            "risk_allowed": c["risk_gap"] <= tol["risk"],
            "operational_allowed": c["operational_gap_norm"] <= tol["operational"],
            "descriptor_allowed": c["descriptor_gap_norm"] <= tol["descriptor"],
            "anchor_allowed": c["anchor_gap"] <= tol["anchor"],
            "severe_risk_allowed": c["risk_gap"] <= tol["severe_risk"],
            "severe_operational_allowed": c["operational_gap_norm"] <= tol["severe_operational"],
        }

        metadata = {
            **c,
            **tol,
            **component_allowed,
            "weights": {
                "risk": w_risk,
                "operational": w_op,
                "descriptor": w_desc,
                "anchor": w_anchor,
                "nuisance": w_nuis,
            },
        }
        return score, metadata

    def metadata(self, allowed: np.ndarray, penalty: np.ndarray, score: np.ndarray, meta: dict) -> dict:
        return {
            **meta,
            **feasibility_metadata(allowed),
            "penalty_mean": float(np.mean(penalty)),
            "penalty_max": float(np.max(penalty)) if penalty.size else 0.0,
            "score_mean": float(np.mean(score)),
            "score_max": float(np.max(score)) if score.size else 0.0,
        }


@register_invariance("hard")
class HardInvariance(MixedInvarianceBase):
    mode = "hard"

    def build(self, node: NodeBatch) -> InvarianceOutput:
        score, meta = self.weighted_score(node)
        allowed = (
            meta["risk_allowed"]
            & meta["operational_allowed"]
            & meta["descriptor_allowed"]
            & meta["anchor_allowed"]
        )
        penalty = np.zeros_like(score)
        hard_violations = ~allowed
        return InvarianceOutput(allowed, penalty, hard_violations, score, self.mode, self.metadata(allowed, penalty, score, meta))


@register_invariance("soft")
class SoftInvariance(MixedInvarianceBase):
    mode = "soft"

    def build(self, node: NodeBatch) -> InvarianceOutput:
        score, meta = self.weighted_score(node)
        allowed = np.ones_like(score, dtype=bool)
        penalty = score
        hard_violations = np.zeros_like(score, dtype=bool)
        return InvarianceOutput(allowed, penalty, hard_violations, score, self.mode, self.metadata(allowed, penalty, score, meta))


@register_invariance("hybrid")
class HybridInvariance(MixedInvarianceBase):
    mode = "hybrid"

    def build(self, node: NodeBatch) -> InvarianceOutput:
        score, meta = self.weighted_score(node)
        soft_start = float(self.cfg.get("admissibility", {}).get("soft_threshold", 2.0))
        severe = ~(meta["severe_risk_allowed"] & meta["severe_operational_allowed"])
        penalty = np.maximum(score - soft_start, 0.0)
        allowed = ~severe
        return InvarianceOutput(allowed, penalty, severe, score, self.mode, self.metadata(allowed, penalty, score, meta))


@register_invariance("adaptive")
class AdaptiveInvariance(MixedInvarianceBase):
    mode = "adaptive"

    def build(self, node: NodeBatch) -> InvarianceOutput:
        score, meta = self.weighted_score(node)
        cfg = self.cfg.get("admissibility", {})
        risk_err = float(node.metadata.get("risk_error_mae", 0.0))
        calib_err = float(node.metadata.get("risk_calibration_error", 0.0) or 0.0)
        n = len(node.z_a) + len(node.z_b)
        sample_term = float(np.sqrt(200.0 / max(n, 1)))

        base_threshold = float(cfg.get("adaptive_base_threshold", 3.0))
        relaxation = (
            float(cfg.get("adaptive_risk_error_weight", 1.0)) * risk_err
            + float(cfg.get("adaptive_calibration_weight", 0.5)) * calib_err
            + float(cfg.get("adaptive_sample_weight", 0.20)) * sample_term
        )
        threshold = base_threshold + relaxation
        hard_threshold = float(cfg.get("adaptive_hard_multiplier", 2.0)) * threshold
        allowed = score <= hard_threshold
        penalty = np.maximum(score - threshold, 0.0) / max(1.0 + relaxation, 1e-12)

        meta = {
            **meta,
            "adaptive_threshold": threshold,
            "adaptive_hard_threshold": hard_threshold,
            "adaptive_relaxation": relaxation,
            "risk_error_mae": risk_err,
            "calibration_error": calib_err,
            "sample_term": sample_term,
        }
        return InvarianceOutput(allowed, penalty, ~allowed, score, self.mode, self.metadata(allowed, penalty, score, meta))
