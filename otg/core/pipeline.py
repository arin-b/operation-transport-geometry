from __future__ import annotations

from pathlib import Path
import time
import numpy as np

from otg.worlds.registry import make_world
from otg.risks.registry import make_risk_model
from otg.invariance.registry import make_invariance_builder
from otg.costs.operational import build_cost
from otg.transport.registry import make_solver
from otg.core.schema import RunArtifact, NodeRunResult, TransportProblem
from otg.diagnostics.metrics import compute_node_diagnostics
from otg.diagnostics.assumptions import assumption_report
from otg.reporting.report import write_run_outputs
from otg.utils.io import ensure_dir, save_json, save_array
from otg.utils.seeding import SeedBank
from otg.utils.presets import apply_runtime_preset


def run_pipeline(cfg: dict, out_dir: str | Path | None = None) -> RunArtifact:
    cfg = apply_runtime_preset(cfg)
    start = time.time()
    seed_bank = SeedBank(int(cfg.get("seed", {}).get("master", 0)))

    world = make_world(cfg["world"]["name"], cfg, seed_bank)
    batch = world.generate()
    batch.validate()

    risk_model = make_risk_model(cfg.get("risk", {}).get("mode", "true"), cfg, seed_bank)
    inv_builder = make_invariance_builder(cfg.get("admissibility", {}).get("mode", "hybrid"), cfg, seed_bank)
    solver = make_solver(cfg.get("transport", {}).get("solver", "sinkhorn"), cfg, seed_bank)

    selected = cfg.get("nodes", {}).get("selected", list(batch.nodes))
    node_results: dict[str, NodeRunResult] = {}

    for node_name in selected:
        node = batch.nodes[node_name]
        risk_out = risk_model.estimate(node)
        node.used_risk_a = risk_out.used_risk_a
        node.used_risk_b = risk_out.used_risk_b
        node.metadata["risk_error_mae"] = risk_out.risk_error_mae
        node.metadata["risk_mode"] = risk_out.mode
        node.metadata["risk_output_metadata"] = risk_out.metadata
        node.metadata["risk_train_error_mae"] = risk_out.train_error_mae
        node.metadata["risk_calibration_error"] = risk_out.calibration_error
        node.metadata["risk_failure_accuracy"] = risk_out.failure_accuracy

        inv = inv_builder.build(node)
        cost = build_cost(node, inv, cfg)
        problem = TransportProblem(node.mass_a, node.mass_b, cost.total, allowed=inv.allowed, penalty=inv.penalty, metadata={"node": node_name, "invariance_mode": inv.mode})
        tr = solver.solve(problem)

        diag = compute_node_diagnostics(node, inv, cost, tr, cfg)
        node_results[node_name] = NodeRunResult(
            node=node_name,
            cost=cost,
            invariance=inv,
            transport=tr,
            diagnostics=diag,
        )

    system_score = _aggregate(node_results, batch, cfg)
    assumptions = assumption_report(batch, node_results, cfg)

    artifact = RunArtifact(
        config=cfg,
        world_name=batch.name,
        node_results=node_results,
        system_score=system_score,
        assumption_report=assumptions,
        metadata={"elapsed_sec": time.time() - start, "seeds": seed_bank.as_dict(), "world_metadata": batch.metadata},
    )

    if out_dir is not None:
        save_artifact(artifact, batch, out_dir)

    return artifact


def _aggregate(node_results: dict[str, NodeRunResult], batch, cfg: dict) -> dict:
    mode = cfg.get("aggregation", {}).get("mode", "weighted_sum")
    values = {k: float(v.transport.value) for k, v in node_results.items()}
    if not values:
        return {"mode": mode, "value": 0.0, "node_values": values}

    if mode == "weighted_sum":
        weights = cfg.get("aggregation", {}).get("weights", {})
        value = sum(float(weights.get(k, 1.0)) * v for k, v in values.items())
    elif mode == "max":
        value = max(values.values())
    elif mode == "risk_weighted":
        value = 0.0
        for k, v in values.items():
            node = batch.nodes[k]
            w = float((node.true_risk_a.mean() + node.true_risk_b.mean()) / 2 + 1e-9)
            value += w * v
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")
    return {"mode": mode, "value": float(value), "node_values": values}


def save_artifact(artifact: RunArtifact, batch, out_dir: str | Path) -> None:
    out = ensure_dir(out_dir)
    ensure_dir(out / "arrays")
    ensure_dir(out / "figures")

    save_json(out / "config.resolved.json", artifact.config)
    save_json(out / "summary.json", _json_summary(artifact))
    save_json(out / "assumptions.json", artifact.assumption_report)
    save_json(out / "seeds.json", artifact.metadata.get("seeds", {}))

    for node_name, res in artifact.node_results.items():
        node = batch.nodes[node_name]
        node_dir = ensure_dir(out / "arrays" / node_name)
        save_array(node_dir / "cost_total.npy", res.cost.total)
        save_array(node_dir / "cost_geometry.npy", res.cost.geometry)
        save_array(node_dir / "cost_terminal.npy", res.cost.terminal)
        save_array(node_dir / "cost_risk.npy", res.cost.risk)
        save_array(node_dir / "cost_invariance.npy", res.cost.invariance)
        save_array(node_dir / "transport_plan.npy", res.transport.plan)
        save_array(node_dir / "admissibility_allowed.npy", res.invariance.allowed.astype(float))
        save_array(node_dir / "invariance_soft_score.npy", res.invariance.soft_score)
        save_array(node_dir / "z_a.npy", node.z_a)
        save_array(node_dir / "z_b.npy", node.z_b)
        save_array(node_dir / "true_risk_a.npy", node.true_risk_a)
        save_array(node_dir / "true_risk_b.npy", node.true_risk_b)
        save_array(node_dir / "used_risk_a.npy", node.used_risk_a)
        save_array(node_dir / "used_risk_b.npy", node.used_risk_b)
        save_json(out / f"{node_name}_diagnostics.json", res.diagnostics)
        save_json(out / f"{node_name}_transport_metadata.json", res.transport.metadata)

    write_run_outputs(out, artifact, batch)


def _json_summary(artifact: RunArtifact) -> dict:
    return {
        "world": artifact.world_name,
        "system_score": artifact.system_score,
        "metadata": artifact.metadata,
        "nodes": {
            k: {
                "transport_value": float(v.transport.value),
                "solver": v.transport.solver,
                "status": v.transport.status,
                "diagnostics": v.diagnostics,
                "cost_mode": v.cost.mode,
                "invariance_mode": v.invariance.mode,
                "transport_metadata": v.transport.metadata,
            }
            for k, v in artifact.node_results.items()
        },
    }
