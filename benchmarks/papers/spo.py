from __future__ import annotations

import numpy as np

from benchmarks.papers.base import PaperMethodBase, failure_labels, label_risk, spo_scores
from benchmarks.registry import register_method
from benchmarks.types import MethodRun


@register_method("spo")
class SPOAdapter(PaperMethodBase):
    paper_name = "Smart Predict, then Optimize"
    implementation_kind = "reimplemented"
    implementation_source = "paper-faithful-ranking-surrogate"

    def run(self, batch, cfg: dict, seed: int):
        node = batch.nodes["repr"]
        Xa = np.c_[node.z_a, node.descriptor_coords_a]
        Xb = np.c_[node.z_b, node.descriptor_coords_b]
        ya, yb = failure_labels(node)
        rep_a, rep_b, meta = spo_scores(Xa, Xb, ya, yb, steps=int(cfg.get("benchmark", {}).get("spo_steps", 220)))
        risk_a, risk_b = label_risk(node)
        return MethodRun("spo", batch.name, seed, rep_a=rep_a[:, None], rep_b=rep_b[:, None], risk_a=risk_a, risk_b=risk_b, metrics=meta, metadata={"implementation_kind": self.implementation_kind, "implementation_source": self.implementation_source})

