from __future__ import annotations

import numpy as np

from benchmarks.papers.base import PaperMethodBase, label_risk, metric_features, riemannian_metric_learning
from benchmarks.registry import register_method
from benchmarks.types import MethodRun


@register_method("mlot")
class MLOTAdapter(PaperMethodBase):
    paper_name = "A Riemannian Approach to Ground Metric Learning for Optimal Transport"
    implementation_kind = "reimplemented"
    implementation_source = "paper-faithful-SPD-ground-metric-learning"

    def run(self, batch, cfg: dict, seed: int):
        node = batch.nodes["repr"]
        Xa = np.c_[node.z_a, node.true_risk_a[:, None]]
        Xb = np.c_[node.z_b, node.true_risk_b[:, None]]
        M, plan, cost = riemannian_metric_learning(Xa, Xb, steps=int(cfg.get("benchmark", {}).get("metric_steps", 12)), lr=float(cfg.get("benchmark", {}).get("metric_lr", 0.2)))
        rep_a = metric_features(Xa, M)
        rep_b = metric_features(Xb, M)
        metrics = {
            "backend": "paper_faithful_spd_metric_learning",
            "metric_trace": float(np.trace(M)),
            "metric_condition_number": float(np.linalg.cond(M)),
            "transport_mass": float(np.sum(plan)),
            "transport_cost_mean": float(np.mean(cost)),
        }
        risk_a, risk_b = label_risk(node)
        return MethodRun("mlot", batch.name, seed, plan=plan, rep_a=rep_a, rep_b=rep_b, risk_a=risk_a, risk_b=risk_b, metrics=metrics, metadata={"implementation_kind": self.implementation_kind, "implementation_source": self.implementation_source, "metric_matrix": M.tolist()})
