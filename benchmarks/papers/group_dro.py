from __future__ import annotations

import numpy as np

from benchmarks.papers.base import PaperMethodBase, failure_labels, group_dro_scores, label_risk
from benchmarks.registry import register_method
from benchmarks.types import MethodRun


@register_method("group_dro")
class GroupDROAdapter(PaperMethodBase):
    paper_name = "Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization"
    implementation_kind = "reimplemented"
    implementation_source = "paper-faithful-worst-group-reweighting-with-regularization"

    def run(self, batch, cfg: dict, seed: int):
        node = batch.nodes["repr"]
        Xa = np.c_[node.z_a, node.nuisance_coords_a]
        Xb = np.c_[node.z_b, node.nuisance_coords_b]
        ya, yb = failure_labels(node)
        rep_a, rep_b, meta = group_dro_scores(
            Xa,
            Xb,
            ya,
            yb,
            steps=int(cfg.get("benchmark", {}).get("group_dro_steps", 250)),
            l2=float(cfg.get("benchmark", {}).get("group_dro_l2", 1e-2)),
        )
        meta["regularization"] = float(cfg.get("benchmark", {}).get("group_dro_l2", 1e-2))
        risk_a, risk_b = label_risk(node)
        return MethodRun("group_dro", batch.name, seed, rep_a=rep_a[:, None], rep_b=rep_b[:, None], risk_a=risk_a, risk_b=risk_b, metrics=meta, metadata={"implementation_kind": self.implementation_kind, "implementation_source": self.implementation_source})
