from __future__ import annotations

from benchmarks.registry import register_method
from benchmarks.types import MethodRun


@register_method("otg")
class OTGBaseline:
    paper_name = "Operational Transport Geometry"
    implementation_kind = "in-repo"
    implementation_source = "otg.core.pipeline"

    def run(self, batch, cfg: dict, seed: int):
        from otg.core.pipeline import run_pipeline

        local = dict(cfg)
        local.setdefault("seed", {})["master"] = seed
        artifact = run_pipeline(local, None)
        node = artifact.node_results["repr"]
        return MethodRun(
            method="otg",
            world=batch.name,
            seed=seed,
            plan=node.transport.plan,
            rep_a=batch.nodes["repr"].z_a,
            rep_b=batch.nodes["repr"].z_b,
            risk_a=batch.nodes["repr"].used_risk_a,
            risk_b=batch.nodes["repr"].used_risk_b,
            allowed=batch.nodes["repr"].metadata.get("compatibility_mask"),
            metrics=node.diagnostics,
            metadata={"solver": node.transport.solver, "status": node.transport.status},
        )

