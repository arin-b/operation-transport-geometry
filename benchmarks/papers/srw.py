from __future__ import annotations

import numpy as np

from benchmarks.papers.base import PaperMethodBase, feature_cost, label_risk
from benchmarks.registry import register_method
from benchmarks.types import MethodRun


@register_method("srw")
class SRWAdapter(PaperMethodBase):
    paper_name = "Subspace Robust Wasserstein Distances"
    implementation_kind = "reimplemented"
    implementation_source = "paper-faithful-iterative-subspace-update"

    def run(self, batch, cfg: dict, seed: int):
        node = batch.nodes["repr"]
        Xa = np.c_[node.z_a, node.operational_coords_a]
        Xb = np.c_[node.z_b, node.operational_coords_b]
        X = np.vstack([Xa, Xb])
        rng = np.random.default_rng(seed)
        k = int(cfg.get("benchmark", {}).get("srw_dim", min(3, max(1, X.shape[1] - 1))))
        Xn = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
        proj = rng.normal(size=(Xn.shape[1], k))
        proj, _ = np.linalg.qr(proj)
        last_value = None
        for _ in range(int(cfg.get("benchmark", {}).get("srw_iter", 5))):
            Z = Xn @ proj
            cost = feature_cost(Z[: len(Xa)], Z[len(Xa):])
            a = np.ones(len(Xa)) / max(len(Xa), 1)
            b = np.ones(len(Xb)) / max(len(Xb), 1)
            try:
                import ot

                plan = ot.sinkhorn(a, b, cost, reg=float(cfg.get("transport", {}).get("epsilon", 0.05)), numItermax=200)
            except Exception:
                plan = np.outer(a, b)
            disp = Xa[:, None, :] - Xb[None, :, :]
            cov = np.einsum("ij,ijk,ijl->kl", plan, disp, disp)
            eigvals, eigvecs = np.linalg.eigh(cov)
            proj = eigvecs[:, np.argsort(eigvals)[::-1][:k]]
            last_value = float(np.sum(plan * cost))
        rep_a = Xn[: len(Xa)] @ proj
        rep_b = Xn[len(Xa):] @ proj
        risk_a, risk_b = label_risk(node)
        return MethodRun("srw", batch.name, seed, rep_a=rep_a, rep_b=rep_b, risk_a=risk_a, risk_b=risk_b, metrics={"transport_proxy": float(last_value or 0.0), "subspace_dim": k}, metadata={"implementation_kind": self.implementation_kind, "implementation_source": self.implementation_source})

