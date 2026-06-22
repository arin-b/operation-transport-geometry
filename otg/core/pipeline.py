from __future__ import annotations

from pathlib import Path
import time
import numpy as np

from otg.core.graph_ops import (
    aggregate_node_pair_results,
    derive_pairwise_problem,
    discrepancy_matrix,
    domain_pairs,
    node_pair_table,
)
from otg.core.schema import (
    GraphRunArtifact,
    GraphWorldBatch,
    NodePairResult,
    NodeRunResult,
    SystemComparisonResult,
    TransportProblem,
)
from otg.costs.operational import build_cost
from otg.core.adaptive import select_solver_name
from otg.diagnostics.metrics import compute_node_diagnostics
from otg.invariance.registry import make_invariance_builder
from otg.reporting.graph_report import write_graph_run_outputs
from otg.risks.registry import make_risk_model
from otg.transport.registry import make_solver
from otg.utils.presets import apply_runtime_preset
from otg.utils.seeding import SeedBank
from otg.worlds.registry import make_world


def run_pipeline(cfg: dict, out_dir: str | Path | None = None) -> GraphRunArtifact:
    """Run the proposal-aligned graph-level OTG pipeline.

    Primary object: deployed DAG + multiple domains + node-wise laws. Pairwise
    OT problems are derived internally for each (node, domain_i, domain_j).
    """
    cfg = apply_runtime_preset(cfg)
    start = time.time()
    seed_bank = SeedBank(int(cfg.get("seed", {}).get("master", 0)))

    world = make_world(cfg["world"]["name"], cfg, seed_bank)
    batch = world.generate()
    if not isinstance(batch, GraphWorldBatch):
        raise TypeError(
            "The primary OTG pipeline now requires GraphWorldBatch. "
            "Legacy pairwise worlds are internal compatibility objects only."
        )
    batch.validate()

    selected_nodes = list(cfg.get("nodes", {}).get("selected", batch.selected_nodes))
    pair_ordered = bool(cfg.get("domain_pairs", {}).get("ordered", False))
    requested_pairs = cfg.get("domain_pairs", {}).get("pairs")
    pairs = [tuple(p) for p in requested_pairs] if requested_pairs else domain_pairs(batch.domain_ids, ordered=pair_ordered)
    use_aligned = bool(cfg.get("projection", {}).get("use_aligned", True))

    risk_model = make_risk_model(cfg.get("risk", {}).get("mode", "true"), cfg, seed_bank)
    inv_builder = make_invariance_builder(cfg.get("admissibility", {}).get("mode", "hybrid"), cfg, seed_bank)

    comparisons: dict[tuple[str, str], SystemComparisonResult] = {}
    representative_node_results: dict[str, NodeRunResult] = {}

    for pair_idx, (da, db) in enumerate(pairs):
        node_results: dict[str, NodePairResult] = {}
        for node in selected_nodes:
            problem_node = derive_pairwise_problem(batch, node, da, db, use_aligned=use_aligned)
            risk_out = risk_model.estimate(problem_node)
            problem_node.used_risk_a = risk_out.used_risk_a
            problem_node.used_risk_b = risk_out.used_risk_b
            problem_node.metadata.update({
                "risk_error_mae": risk_out.risk_error_mae,
                "risk_mode": risk_out.mode,
                "risk_output_metadata": risk_out.metadata,
                "risk_train_error_mae": risk_out.train_error_mae,
                "risk_calibration_error": risk_out.calibration_error,
                "risk_failure_accuracy": risk_out.failure_accuracy,
            })

            inv = inv_builder.build(problem_node)
            cost = build_cost(problem_node, inv, cfg)
            transport_problem = TransportProblem(
                problem_node.mass_a,
                problem_node.mass_b,
                cost.total,
                allowed=inv.allowed,
                penalty=inv.penalty,
                metadata={
                    "node": node,
                    "domain_a": da,
                    "domain_b": db,
                    "invariance_mode": inv.mode,
                    "use_aligned": use_aligned,
                },
            )
            solver_name = select_solver_name(problem_node, cfg, default=str(cfg.get("transport", {}).get("solver", "masked_sinkhorn")), method_name="otg")
            solver = make_solver(solver_name, cfg, seed_bank)
            tr = solver.solve(transport_problem)
            diag = compute_node_diagnostics(problem_node, inv, cost, tr, cfg)
            for k, v in cost.weights.items():
                if isinstance(v, (int, float, bool)):
                    diag[f"cost_weight_{k}"] = float(v)
            diag.update({
                "node": node,
                "domain_a": da,
                "domain_b": db,
                "projection_used": use_aligned,
                "projection_a_kind": problem_node.projection_a.get("kind"),
                "projection_b_kind": problem_node.projection_b.get("kind"),
            })
            res = NodePairResult(
                node=node,
                domain_pair=(da, db),
                cost=cost,
                invariance=inv,
                transport=tr,
                diagnostics=diag,
                pairwise_problem=problem_node,
            )
            node_results[node] = res
            if pair_idx == 0:
                representative_node_results[node] = NodeRunResult(node=node, cost=cost, invariance=inv, transport=tr, diagnostics=diag)

        aggregate = aggregate_node_pair_results(node_results, batch, (da, db), cfg)
        comp_diag = _comparison_diagnostics(node_results, aggregate)
        comparisons[(da, db)] = SystemComparisonResult(domain_pair=(da, db), node_results=node_results, aggregate=aggregate, diagnostics=comp_diag)

    domain_order = batch.domain_ids
    matrix = discrepancy_matrix(comparisons, domain_order)
    table = node_pair_table(comparisons)
    system_score = _system_summary(comparisons, matrix, domain_order)
    assumptions = _graph_assumption_report(batch, comparisons, matrix, cfg)

    artifact = GraphRunArtifact(
        config=cfg,
        world_name=batch.name,
        graph=batch.graph,
        domains=batch.domains,
        selected_nodes=selected_nodes,
        comparisons=comparisons,
        discrepancy_matrix=matrix,
        domain_order=domain_order,
        node_pair_table=table,
        system_score=system_score,
        assumption_report=assumptions,
        metadata={
            "elapsed_sec": time.time() - start,
            "seeds": seed_bank.as_dict(),
            "world_metadata": batch.metadata,
            "projection_metadata": batch.projection_metadata,
            "terminal_evaluation_metadata": batch.terminal_evaluation_metadata,
            "pipeline": "graph_level_otg",
        },
        node_results=representative_node_results,
    )

    if out_dir is not None:
        write_graph_run_outputs(Path(out_dir), artifact, batch)
    return artifact


def _comparison_diagnostics(node_results: dict[str, NodePairResult], aggregate: dict | None = None) -> dict:
    if not node_results:
        return {}
    aggregate = aggregate or {}
    vals = [float(r.transport.value) for r in node_results.values()]
    danger = [float(r.diagnostics.get("dangerous_unmatched_mass_b", 0.0)) for r in node_results.values()]
    false_collapse = [float(r.diagnostics.get("false_collapse_mass", 0.0)) for r in node_results.values()]
    forbidden = [float(r.diagnostics.get("transport_forbidden_mass", 0.0)) for r in node_results.values()]
    return {
        "mean_node_transport": float(np.mean(vals)),
        "max_node_transport": float(np.max(vals)),
        "max_dangerous_unmatched_mass_b": float(np.max(danger)),
        "max_false_collapse_mass": float(np.max(false_collapse)),
        "max_forbidden_mass": float(np.max(forbidden)),
        "localizing_node": aggregate.get("localizing_node") or max(node_results, key=lambda n: float(node_results[n].transport.value)),
        "localization_scores": aggregate.get("localization_scores", {}),
    }


def _system_summary(comparisons: dict[tuple[str, str], SystemComparisonResult], matrix: np.ndarray, domain_order: list[str]) -> dict:
    vals = [float(c.aggregate.get("value", 0.0)) for c in comparisons.values()]
    if vals:
        max_pair = max(comparisons, key=lambda p: float(comparisons[p].aggregate.get("value", 0.0)))
        min_pair = min(comparisons, key=lambda p: float(comparisons[p].aggregate.get("value", 0.0)))
    else:
        max_pair = min_pair = (None, None)
    return {
        "mode": "domain_pair_matrix_mean",
        "value": float(np.mean(vals)) if vals else 0.0,
        "max_pair": list(max_pair),
        "max_pair_value": float(comparisons[max_pair].aggregate.get("value", 0.0)) if vals else 0.0,
        "min_pair": list(min_pair),
        "min_pair_value": float(comparisons[min_pair].aggregate.get("value", 0.0)) if vals else 0.0,
        "domain_order": domain_order,
        "num_domain_pairs": len(comparisons),
    }


def _graph_assumption_report(batch: GraphWorldBatch, comparisons: dict[tuple[str, str], SystemComparisonResult], matrix: np.ndarray, cfg: dict) -> dict:
    checks = {
        "multiple_domains": len(batch.domains) >= 4,
        "multiple_selected_nodes": len(batch.selected_nodes) >= 3,
        "node_law_complete": all((node, d) in batch.node_laws for node in batch.selected_nodes for d in batch.domains),
        "domain_pair_matrix_produced": matrix.shape == (len(batch.domains), len(batch.domains)),
        "graph_pipeline": True,
    }
    pair_reports = {}
    for pair, comp in comparisons.items():
        pair_reports[f"{pair[0]}::{pair[1]}"] = {
            "D_op": float(comp.aggregate.get("value", 0.0)),
            "node_values": comp.aggregate.get("node_values", {}),
            "localizing_node": comp.diagnostics.get("localizing_node"),
            "max_forbidden_mass": comp.diagnostics.get("max_forbidden_mass"),
            "max_dangerous_unmatched_mass_b": comp.diagnostics.get("max_dangerous_unmatched_mass_b"),
        }
    return {
        "world": batch.name,
        "proposal_mapping": {
            "D": list(batch.domains),
            "P_d": "DomainSpec.input_law_metadata",
            "mu_v^d": "GraphWorldBatch.node_laws[(v,d)].samples",
            "phi_v^d": "NodeLaw.projection_metadata and aligned_samples",
            "mutilde_v^d": "NodeLaw.aligned_samples",
            "r_v/rhat_v": "NodeLaw.true_risk / PairwiseNodeProblem.used_risk",
            "Pi_v_adm": "InvarianceOutput.allowed",
            "W_v_op": "NodePairResult.transport.value",
            "D_op(d,d')": "SystemComparisonResult.aggregate.value",
        },
        "checks": checks,
        "domain_pair_reports": pair_reports,
    }


# Compatibility alias for old callers that imported save_artifact directly. The
# graph pipeline writes through reporting.graph_report instead.
def save_artifact(artifact: GraphRunArtifact, batch: GraphWorldBatch, out_dir: str | Path) -> None:
    write_graph_run_outputs(Path(out_dir), artifact, batch)
