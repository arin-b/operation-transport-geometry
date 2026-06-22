from __future__ import annotations

import numpy as np

from benchmarks.papers.base import PaperMethodBase, failure_labels, irm_scores, label_risk
from benchmarks.registry import register_method
from benchmarks.types import MethodRun


@register_method("irm")
class IRMAdapter(PaperMethodBase):
    paper_name = "Invariant Risk Minimization"
    implementation_kind = "reimplemented"
    implementation_source = "paper-faithful-environmental-penalty-logistic"

    def run(self, batch, cfg: dict, seed: int):
        node = batch.nodes["repr"]
        Xa = np.c_[node.z_a, node.operational_coords_a]
        Xb = np.c_[node.z_b, node.operational_coords_b]
        ya, yb = failure_labels(node)
        rep_a, rep_b, meta = irm_scores(Xa, Xb, ya, yb, penalty=float(cfg.get("benchmark", {}).get("irm_penalty", 1.0)))
        risk_a, risk_b = label_risk(node)
        return MethodRun("irm", batch.name, seed, rep_a=rep_a[:, None], rep_b=rep_b[:, None], risk_a=risk_a, risk_b=risk_b, metrics=meta, metadata={"implementation_kind": self.implementation_kind, "implementation_source": self.implementation_source})

