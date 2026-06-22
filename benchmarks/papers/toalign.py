from __future__ import annotations

import numpy as np

from benchmarks.papers.base import PaperMethodBase, failure_labels, label_risk, normalize_rows
from benchmarks.registry import register_method
from benchmarks.types import MethodRun


@register_method("toalign")
class ToAlignAdapter(PaperMethodBase):
    paper_name = "ToAlign"
    implementation_kind = "reimplemented"
    implementation_source = "paper-faithful-task-related/task-irrelevant-decomposition"

    def run(self, batch, cfg: dict, seed: int):
        node = batch.nodes["repr"]
        Xa = np.c_[node.z_a, node.descriptor_coords_a]
        Xb = np.c_[node.z_b, node.descriptor_coords_b]
        ya, yb = failure_labels(node)
        import ot

        Xa_n = normalize_rows(Xa)
        Xb_n = normalize_rows(Xb)
        # Task-related feature is the discriminative direction induced by failure labels.
        w = np.zeros(Xa_n.shape[1], dtype=float)
        pos = ya >= 0.5
        neg = ~pos
        if np.any(pos) and np.any(neg):
            mu_pos = Xa_n[pos].mean(axis=0)
            mu_neg = Xa_n[neg].mean(axis=0)
            w = mu_pos - mu_neg
        if np.linalg.norm(w) < 1e-8:
            w = np.ones(Xa_n.shape[1], dtype=float)
        w = w / (np.linalg.norm(w) + 1e-8)
        task_a = Xa_n @ w
        task_b = Xb_n @ w
        task_irrel_a = np.linalg.norm(Xa_n - np.outer(task_a, w), axis=1)
        task_irrel_b = np.linalg.norm(Xb_n - np.outer(task_b, w), axis=1)

        a = np.ones(len(task_a), dtype=float) / max(len(task_a), 1)
        b = np.ones(len(task_b), dtype=float) / max(len(task_b), 1)
        cost = np.abs(task_a[:, None] - task_b[None, :]) + 0.1 * np.abs(task_irrel_a[:, None] - task_irrel_b[None, :])
        plan = ot.sinkhorn(a, b, cost, reg=float(cfg.get("transport", {}).get("epsilon", 0.05)), numItermax=300)
        rep_a = np.c_[task_a, task_irrel_a]
        rep_b = np.c_[task_b, task_irrel_b]
        metrics = {
            "backend": "paper_faithful_task_decomposition",
            "task_direction_norm": float(np.linalg.norm(w)),
            "task_related_gap": float(np.abs(task_a.mean() - task_b.mean())),
            "task_irrelevant_gap": float(np.abs(task_irrel_a.mean() - task_irrel_b.mean())),
            "transport_mass": float(np.sum(plan)),
        }
        risk_a, risk_b = label_risk(node)
        return MethodRun("toalign", batch.name, seed, plan=plan, rep_a=rep_a, rep_b=rep_b, risk_a=risk_a, risk_b=risk_b, metrics=metrics, metadata={"implementation_kind": self.implementation_kind, "implementation_source": self.implementation_source, "task_direction": w.tolist()})
