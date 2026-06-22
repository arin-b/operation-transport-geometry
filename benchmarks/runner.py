from __future__ import annotations

from pathlib import Path
import csv
import json
import zipfile
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist as _cdist

from benchmarks.metrics import (
    PRIMARY_CLAIMS,
    SECONDARY_CLAIMS,
    graph_claim_scores,
    legacy_composite_score,
    normalize_cost,
)
from benchmarks.registry import available_methods, get_method, method_provenance
from benchmarks.plots import write_benchmark_figures
from otg.core.graph_ops import aggregate_node_pair_results, derive_pairwise_problem, domain_pairs
from otg.core.adaptive import select_solver_name, should_use_unbalanced
from otg.core.schema import CostOutput, NodePairResult, SystemComparisonResult, TransportProblem
from otg.costs.operational import build_cost
from otg.diagnostics.metrics import compute_node_diagnostics
from otg.invariance.registry import make_invariance_builder
from otg.risks.registry import make_risk_model
from otg.transport.registry import make_solver
from otg.utils.config import deep_merge, load_config
from otg.utils.io import ensure_dir, save_json
from otg.utils.presets import apply_runtime_preset
from otg.utils.seeding import SeedBank
from otg.worlds.registry import make_world


DEFAULT_BENCHMARK_CFG = {
    "base_config": "configs/default.yaml",
    "methods": ["otg", "toalign", "mlot", "srw", "irm", "group_dro", "spo"],
    "worlds": ["synthetic_dag"],
    "seed_count_main": 1,
    "seed_count_stress": 0,
    "shared_seed_stride": 1000,
    "solver": "masked_sinkhorn",
}


def _pairwise_wrapper(problem, world_name: str):
    # Literature adapters remain pairwise internally. The benchmark-level object is
    # still the full graph/domain/node tensor; this wrapper is only a compatibility
    # slice for one (node, domain_i, domain_j) numerical comparison.
    return SimpleNamespace(name=world_name, nodes={"repr": problem})


def _dangerous_unmatched_enabled(cfg: dict) -> bool:
    return bool(cfg.get("benchmark", {}).get("dangerous_unmatched", {}).get("enabled", True))


def _target_domain(cfg: dict) -> str:
    return str(cfg.get("benchmark", {}).get("dangerous_unmatched", {}).get("target_domain", "occlusion"))


def _solver_name_for_problem(cfg: dict, problem) -> str:
    """Data-based hybrid solver policy.

    Unbalanced OT is used when the node-pair has evidence of dangerous unmatched
    high-risk mass. The old domain-name target remains only as a smoke-test
    fallback if explicitly requested; the primary trigger is terminally induced
    risk/failure structure.
    """
    default = str(cfg.get("benchmark", {}).get("solver", cfg.get("transport", {}).get("solver", "masked_sinkhorn")))
    if bool(cfg.get("benchmark", {}).get("dangerous_unmatched", {}).get("domain_name_fallback", False)):
        target = _target_domain(cfg)
        if _dangerous_unmatched_enabled(cfg) and target in {problem.domain_a, problem.domain_b}:
            return str(cfg.get("benchmark", {}).get("dangerous_unmatched", {}).get("solver", "unbalanced"))
    if _dangerous_unmatched_enabled(cfg) and should_use_unbalanced(problem, cfg):
        return str(cfg.get("benchmark", {}).get("dangerous_unmatched", {}).get("solver", "unbalanced"))
    return select_solver_name(problem, cfg, default=default)


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x[:, None]
    return x


def _benchmark_cost_for_method(problem, method_run, cfg: dict, method_name: str) -> tuple[CostOutput, dict[str, Any]]:
    rep_a = _as_2d(np.asarray(method_run.rep_a, dtype=float))
    rep_b = _as_2d(np.asarray(method_run.rep_b, dtype=float))
    risk_a = np.asarray(method_run.risk_a if method_run.risk_a is not None else problem.used_risk_a, dtype=float)
    risk_b = np.asarray(method_run.risk_b if method_run.risk_b is not None else problem.used_risk_b, dtype=float)

    geom = normalize_cost(_cdist(rep_a, rep_b))
    risk = normalize_cost(np.abs(risk_a[:, None] - risk_b[None, :]))
    terminal = normalize_cost(_cdist(problem.terminal_a.reshape(len(problem.terminal_a), -1), problem.terminal_b.reshape(len(problem.terminal_b), -1)))
    op = normalize_cost(_cdist(problem.operational_coords_a, problem.operational_coords_b))
    nuisance = normalize_cost(_cdist(problem.nuisance_coords_a, problem.nuisance_coords_b))

    weights = cfg.get("benchmark", {}).get("weights", {})
    total = (
        float(weights.get("geometry", 1.0)) * geom
        + float(weights.get("terminal", 1.0)) * terminal
        + float(weights.get("risk", 1.5)) * risk
        + float(weights.get("operational", 1.0)) * op
        + float(weights.get("nuisance", 0.2)) * nuisance
    )
    cost = CostOutput(
        total=total,
        geometry=geom,
        terminal=terminal,
        risk=risk,
        invariance=np.zeros_like(total),
        mode=f"benchmark_graph_tensor::{method_name}",
        weights=dict(weights),
    )
    return cost, {"rep_dim": int(rep_a.shape[1]), "risk_source": "method_or_node"}


def _run_node_pair_for_method(method_name: str, adapter, batch, problem, cfg: dict, seed_bank: SeedBank, seed: int, world_name: str):
    inv_builder = make_invariance_builder(cfg.get("admissibility", {}).get("mode", "hybrid"), cfg, seed_bank)
    inv = inv_builder.build(problem)

    status = "ok"
    metrics_extra: dict[str, Any] = {}
    if method_name == "otg":
        risk_model = make_risk_model(cfg.get("risk", {}).get("mode", "true"), cfg, seed_bank)
        risk_out = risk_model.estimate(problem)
        problem.used_risk_a = risk_out.used_risk_a
        problem.used_risk_b = risk_out.used_risk_b
        problem.metadata.update({
            "risk_error_mae": risk_out.risk_error_mae,
            "risk_mode": risk_out.mode,
            "risk_output_metadata": risk_out.metadata,
            "risk_train_error_mae": risk_out.train_error_mae,
            "risk_calibration_error": risk_out.calibration_error,
            "risk_failure_accuracy": risk_out.failure_accuracy,
        })
        # Rebuild admissibility after the method's risk access is fixed.
        inv = inv_builder.build(problem)
        otg_cfg = cfg
        otg_cost_mode = cfg.get("benchmark", {}).get("otg_cost_mode")
        if otg_cost_mode:
            otg_cfg = deep_merge(otg_cfg, {"cost": {"mode": otg_cost_mode}})
        cost = build_cost(problem, inv, otg_cfg)
        metrics_extra = {"method_backend": "native_otg_cost_admissibility", "otg_cost_mode": cost.mode}
    else:
        try:
            method_run = adapter.run(_pairwise_wrapper(problem, world_name), cfg, seed)
            cost, cost_meta = _benchmark_cost_for_method(problem, method_run, cfg, method_name)
            metrics_extra = {k: v for k, v in method_run.metrics.items() if isinstance(v, (int, float, bool, str))}
            metrics_extra.update(cost_meta)
        except Exception as exc:
            # Keep failure explicit; do not silently tune or discard bad methods.
            fallback = SimpleNamespace(rep_a=problem.z_a, rep_b=problem.z_b, risk_a=problem.used_risk_a, risk_b=problem.used_risk_b, metrics={})
            cost, cost_meta = _benchmark_cost_for_method(problem, fallback, cfg, method_name)
            metrics_extra = {"method_error": type(exc).__name__, **cost_meta}
            status = f"method_failed:{type(exc).__name__}"

    solver_name = _solver_name_for_problem(cfg, problem)
    if method_name == "otg" and solver_name != str(cfg.get("benchmark", {}).get("dangerous_unmatched", {}).get("solver", "unbalanced")):
        solver_name = str(cfg.get("benchmark", {}).get("otg_solver", solver_name))
        if solver_name in {"auto", "hybrid_auto"}:
            solver_name = select_solver_name(problem, cfg, default=solver_name, method_name=method_name)
    solver_cfg = deep_merge(cfg, {"transport": {"solver": solver_name}})
    solver = make_solver(solver_name, solver_cfg, seed_bank)
    tr = solver.solve(
        TransportProblem(
            problem.mass_a,
            problem.mass_b,
            cost.total,
            allowed=inv.allowed,
            penalty=inv.penalty,
            metadata={
                "method": method_name,
                "node": problem.node,
                "domain_a": problem.domain_a,
                "domain_b": problem.domain_b,
                "benchmark_object": "graph_domain_node_tensor",
                "solver_policy": "adaptive_unbalanced_for_dangerous_mass" if solver_name == "unbalanced" else "balanced_admissible",
            },
        )
    )
    diag = compute_node_diagnostics(problem, inv, cost, tr, cfg)
    for wk, wv in cost.weights.items():
        if isinstance(wv, (int, float, bool)):
            diag[f"cost_weight_{wk}"] = float(wv)
    diag.update(metrics_extra)
    diag.update({
        "node": problem.node,
        "domain_a": problem.domain_a,
        "domain_b": problem.domain_b,
        "status": status,
        "benchmark_solver": tr.solver,
        "benchmark_solver_policy": "adaptive_unbalanced_for_dangerous_mass" if solver_name == "unbalanced" else "balanced_admissible",
        "dangerous_unmatched_mass_total": float(diag.get("dangerous_unmatched_mass_a", 0.0) + diag.get("dangerous_unmatched_mass_b", 0.0)),
    })
    return NodePairResult(
        node=problem.node,
        domain_pair=(problem.domain_a, problem.domain_b),
        cost=cost,
        invariance=inv,
        transport=tr,
        diagnostics=diag,
        pairwise_problem=problem,
    )


def run_benchmark(cfg: dict, out_dir: str | Path, world_name: str, method_name: str, seed: int) -> dict:
    """Evaluate one method on the whole graph-domain-node tensor.

    Pairwise OT remains the solver primitive, but the benchmark unit is not a
    pair. It is the full multi-domain DAG-induced family of node laws.
    """
    out = ensure_dir(out_dir)
    cfg = apply_runtime_preset(cfg)
    seed_bank = SeedBank(int(seed))
    local_cfg = deep_merge(cfg, {"world": {"name": world_name}, "seed": {"master": seed}})

    world = make_world(world_name, local_cfg, seed_bank)
    batch = world.generate()
    batch.validate()

    domain_order = batch.domain_ids
    selected_nodes = list(local_cfg.get("nodes", {}).get("selected", batch.selected_nodes))
    pairs = domain_pairs(domain_order, ordered=bool(local_cfg.get("domain_pairs", {}).get("ordered", False)))
    use_aligned = bool(local_cfg.get("projection", {}).get("use_aligned", True))
    adapter = None if method_name == "otg" else get_method(method_name)()

    comparisons: dict[tuple[str, str], SystemComparisonResult] = {}
    rows: list[dict] = []
    k = len(domain_order)
    node_index = {node: i for i, node in enumerate(selected_nodes)}
    domain_index = {d: i for i, d in enumerate(domain_order)}
    node_tensor = np.zeros((len(selected_nodes), k, k), dtype=float)
    dangerous_tensor = np.zeros_like(node_tensor)
    forbidden_tensor = np.zeros_like(node_tensor)
    false_collapse_tensor = np.zeros_like(node_tensor)
    row_error_tensor = np.zeros_like(node_tensor)
    col_error_tensor = np.zeros_like(node_tensor)
    risk_error_tensor = np.zeros_like(node_tensor)

    localization_hits: dict[str, bool] = {}
    localization_expectations = local_cfg.get("benchmark", {}).get("localization_expectations", {
        "clear::glare": ["measurement"],
        "clear::occlusion": ["measurement", "representation"],
        "viewpoint_shift::occlusion": ["measurement", "representation"],
    })

    for da, db in pairs:
        node_results: dict[str, NodePairResult] = {}
        for node_name in selected_nodes:
            problem = derive_pairwise_problem(batch, node_name, da, db, use_aligned=use_aligned)
            res = _run_node_pair_for_method(method_name, adapter, batch, problem, local_cfg, seed_bank, seed, world_name)
            node_results[node_name] = res
            row = {
                "method": method_name,
                "world": world_name,
                "seed": seed,
                "domain_a": da,
                "domain_b": db,
                "node": node_name,
                "transport_value": float(res.transport.value),
                "solver": res.transport.solver,
                "status": res.diagnostics.get("status", res.transport.status),
                **{k2: v for k2, v in res.diagnostics.items() if isinstance(v, (int, float, bool, str))},
            }
            rows.append(row)

            ni, ia, ib = node_index[node_name], domain_index[da], domain_index[db]
            node_tensor[ni, ia, ib] = node_tensor[ni, ib, ia] = float(res.transport.value)
            dangerous = float(res.diagnostics.get("dangerous_unmatched_mass_total", 0.0))
            forbidden = float(res.diagnostics.get("transport_forbidden_mass", 0.0))
            false_collapse = float(res.diagnostics.get("false_collapse_mass", 0.0))
            row_err = float(res.diagnostics.get("transport_row_l1_error", 0.0))
            col_err = float(res.diagnostics.get("transport_col_l1_error", 0.0))
            risk_err = float(res.diagnostics.get("risk_estimation_mae", 0.0))
            dangerous_tensor[ni, ia, ib] = dangerous_tensor[ni, ib, ia] = max(dangerous_tensor[ni, ia, ib], dangerous)
            forbidden_tensor[ni, ia, ib] = forbidden_tensor[ni, ib, ia] = max(forbidden_tensor[ni, ia, ib], forbidden)
            false_collapse_tensor[ni, ia, ib] = false_collapse_tensor[ni, ib, ia] = max(false_collapse_tensor[ni, ia, ib], false_collapse)
            row_error_tensor[ni, ia, ib] = row_error_tensor[ni, ib, ia] = max(row_error_tensor[ni, ia, ib], row_err)
            col_error_tensor[ni, ia, ib] = col_error_tensor[ni, ib, ia] = max(col_error_tensor[ni, ia, ib], col_err)
            risk_error_tensor[ni, ia, ib] = risk_error_tensor[ni, ib, ia] = max(risk_error_tensor[ni, ia, ib], risk_err)

        aggregate = aggregate_node_pair_results(node_results, batch, (da, db), local_cfg)
        comp = SystemComparisonResult(domain_pair=(da, db), node_results=node_results, aggregate=aggregate, diagnostics=_comparison_diagnostics(node_results, aggregate))
        comparisons[(da, db)] = comp
        pair_id = f"{da}::{db}"
        expected_nodes = localization_expectations.get(pair_id, localization_expectations.get(f"{db}::{da}", []))
        if expected_nodes:
            localization_hits[pair_id] = comp.diagnostics.get("localizing_node") in set(expected_nodes)

    D_matrix = _matrix_from_comparisons(comparisons, domain_order)
    dangerous_matrix = np.max(dangerous_tensor, axis=0) if dangerous_tensor.size else np.zeros_like(D_matrix)
    forbidden_matrix = np.max(forbidden_tensor, axis=0) if forbidden_tensor.size else np.zeros_like(D_matrix)
    false_collapse_matrix = np.max(false_collapse_tensor, axis=0) if false_collapse_tensor.size else np.zeros_like(D_matrix)
    row_error_matrix = np.mean(row_error_tensor, axis=0) if row_error_tensor.size else np.zeros_like(D_matrix)
    col_error_matrix = np.mean(col_error_tensor, axis=0) if col_error_tensor.size else np.zeros_like(D_matrix)
    risk_error_matrix = np.mean(risk_error_tensor, axis=0) if risk_error_tensor.size else np.zeros_like(D_matrix)

    metrics = _aggregate_metrics_from_graph(rows)
    claim_scores = graph_claim_scores(
        D_matrix=D_matrix,
        dangerous_matrix=dangerous_matrix,
        forbidden_matrix=forbidden_matrix,
        false_collapse_matrix=false_collapse_matrix,
        row_error_matrix=row_error_matrix,
        col_error_matrix=col_error_matrix,
        risk_error_matrix=risk_error_matrix,
        localization_hits=localization_hits,
        domain_order=domain_order,
        config=local_cfg,
    )
    legacy_score = legacy_composite_score(metrics)
    cost_components = _cost_component_summary(rows)

    summary = {
        "method": method_name,
        "world": world_name,
        "seed": seed,
        "pipeline": "graph_domain_node_tensor",
        "domain_order": domain_order,
        "selected_nodes": selected_nodes,
        "node_pair_table": rows,
        "metrics": metrics,
        "claim_scores": claim_scores,
        "legacy_composite_score": legacy_score,
        # Backward compatibility field; not the primary benchmark objective.
        "composite_score": claim_scores["primary_claim_average_diagnostic"] * 100.0,
        "metadata": {
            "graph_level": True,
            "benchmark_object": "full_tensor_W[v,i,j]_with_system_matrix_D[i,j]",
            "dangerous_unmatched_policy": "data-based unbalanced solver when terminally induced dangerous mass is unmatched",
            "legacy_target_domain_fallback": _target_domain(local_cfg),
            **method_provenance(method_name),
        },
        "domain_pair_scores": {f"{a}::{b}": comp.aggregate for (a, b), comp in comparisons.items()},
        "domain_matrices": {
            "D_op": D_matrix.tolist(),
            "dangerous_unmatched": dangerous_matrix.tolist(),
            "forbidden_mass": forbidden_matrix.tolist(),
            "false_collapse": false_collapse_matrix.tolist(),
            "row_marginal_error": row_error_matrix.tolist(),
            "col_marginal_error": col_error_matrix.tolist(),
            "risk_error": risk_error_matrix.tolist(),
        },
        "node_value_tensor": node_tensor.tolist(),
        "node_dangerous_unmatched_tensor": dangerous_tensor.tolist(),
        "localization_hits": localization_hits,
        "cost_component_summary": cost_components,
    }
    _write_run_outputs(out, summary)
    return summary


def _comparison_diagnostics(node_results: dict[str, NodePairResult], aggregate: dict | None = None) -> dict:
    aggregate = aggregate or {}
    vals = [float(r.transport.value) for r in node_results.values()]
    danger = [float(r.diagnostics.get("dangerous_unmatched_mass_total", 0.0)) for r in node_results.values()]
    forbidden = [float(r.diagnostics.get("transport_forbidden_mass", 0.0)) for r in node_results.values()]
    return {
        "mean_node_transport": float(np.mean(vals)) if vals else 0.0,
        "max_node_transport": float(np.max(vals)) if vals else 0.0,
        "max_dangerous_unmatched_mass": float(np.max(danger)) if danger else 0.0,
        "max_forbidden_mass": float(np.max(forbidden)) if forbidden else 0.0,
        "localizing_node": aggregate.get("localizing_node") or (max(node_results, key=lambda n: float(node_results[n].transport.value)) if node_results else None),
        "localization_scores": aggregate.get("localization_scores", {}),
    }


def _matrix_from_comparisons(comparisons: dict[tuple[str, str], SystemComparisonResult], domain_order: list[str]) -> np.ndarray:
    n = len(domain_order)
    idx = {d: i for i, d in enumerate(domain_order)}
    M = np.zeros((n, n), dtype=float)
    for (a, b), comp in comparisons.items():
        i, j = idx[a], idx[b]
        M[i, j] = M[j, i] = float(comp.aggregate.get("value", 0.0))
    return M


def _aggregate_metrics_from_graph(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = set(k for row in rows for k, v in row.items() if isinstance(v, (int, float, bool)))
    out = {}
    for k in keys:
        vals = [float(row[k]) for row in rows if isinstance(row.get(k), (int, float, bool))]
        if vals:
            out[k] = float(np.mean(vals))
            out[f"max_{k}"] = float(np.max(vals))
    if "dangerous_unmatched_mass_total" in out:
        out["dangerous_unmatched_mass_total"] = out.get("max_dangerous_unmatched_mass_total", out["dangerous_unmatched_mass_total"])
    return out


def _cost_component_summary(rows: list[dict]) -> dict[str, float]:
    keys = [
        "ordinary_geometric_value_on_plan",
        "terminal_value_on_plan",
        "risk_value_on_plan",
        "invariance_value_on_plan",
        "operational_value_on_plan",
        "ordinary_vs_operational_gap",
        "false_separation_score",
        "cost_weight_adaptive_gate",
        "cost_weight_adaptive_raw_gate",
        "cost_weight_geometry",
        "cost_weight_terminal",
        "cost_weight_risk",
        "cost_weight_invariance",
        "cost_weight_operational_coordinate",
        "cost_weight_nuisance",
        "cost_weight_adaptive_risk_shift",
        "cost_weight_adaptive_terminal_shift",
        "cost_weight_adaptive_failure_shift",
        "cost_weight_adaptive_operational_shift",
        "cost_weight_adaptive_nuisance_shift",
        "cost_weight_adaptive_nuisance_ratio",
        "cost_weight_adaptive_harmless_nuisance",
    ]
    summary = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float, bool))]
        if vals:
            summary[f"mean_{key}"] = float(np.mean(vals))
            summary[f"max_{key}"] = float(np.max(vals))
    return summary


def _write_matrix_csv(path: Path, matrix: np.ndarray, domain_order: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain"] + domain_order)
        for domain, row in zip(domain_order, matrix):
            writer.writerow([domain] + [float(x) for x in row])


def _write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted(set().union(*(row.keys() for row in rows)))
    priority = ["method", "world", "seed", "domain_a", "domain_b", "node", "transport_value", "solver", "status"]
    fieldnames = priority + [k for k in keys if k not in priority]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_run_outputs(out: Path, summary: dict) -> None:
    save_json(out / "summary.json", summary)
    save_json(out / "metrics.json", summary["metrics"])
    save_json(out / "claim_scores.json", summary["claim_scores"])
    save_json(out / "domain_matrices.json", summary["domain_matrices"])
    save_json(out / "cost_component_summary.json", summary["cost_component_summary"])
    _write_rows_csv(out / "node_pair_table.csv", summary["node_pair_table"])
    domain_order = summary["domain_order"]
    for name, matrix in summary["domain_matrices"].items():
        _write_matrix_csv(out / f"{name}_matrix.csv", np.asarray(matrix, dtype=float), domain_order)


def run_benchmark_suite(cfg: dict, out_dir: str | Path) -> dict:
    out = ensure_dir(out_dir)
    base = load_config(cfg.get("base_config", "configs/default.yaml")) if cfg.get("base_config") else load_config("configs/default.yaml")
    cfg = deep_merge(base, cfg)
    cfg = apply_runtime_preset(cfg)
    world_names = cfg.get("worlds") or ["synthetic_dag"]
    methods = cfg.get("methods") or available_methods()
    seed_count_main = int(cfg.get("seed_count_main", 1))
    seed_count_stress = int(cfg.get("seed_count_stress", 0))
    stride = int(cfg.get("shared_seed_stride", 1000))
    main_seeds = [int(cfg.get("seed", {}).get("master", 0)) + i for i in range(seed_count_main)]
    stress_seeds = [int(cfg.get("seed", {}).get("master", 0)) + 10000 + i * stride for i in range(seed_count_stress)]

    results: list[dict] = []
    for world_name in world_names:
        for method in methods:
            for seed in main_seeds:
                run_dir = out / world_name / method / f"seed_{seed}"
                results.append({**run_benchmark(cfg, run_dir, world_name, method, seed), "out": str(run_dir), "phase": "main"})
            for seed in stress_seeds:
                run_dir = out / world_name / method / f"stress_{seed}"
                results.append({**run_benchmark(cfg, run_dir, world_name, method, seed), "out": str(run_dir), "phase": "stress"})

    aggregate = _aggregate_results(results)
    save_json(out / "benchmark_results.json", {"config": cfg, "results": results, "aggregate": aggregate})
    save_json(out / "benchmark_provenance.json", _provenance_catalog(methods))
    _write_csv(out, results)
    _write_claim_csv(out, aggregate)
    _write_method_matrix_files(out, results, aggregate)
    inspection = _write_otg_cost_weight_inspection(out, cfg, results)
    figure_paths = write_benchmark_figures(out, results, aggregate)
    _write_report(out, cfg, results, aggregate, inspection)
    return {"config": cfg, "results": results, "aggregate": aggregate, "out_dir": str(out), "figures": [str(p) for p in figure_paths], "weight_inspection": inspection}


def _aggregate_results(results: list[dict]) -> dict:
    by_method: dict[str, list[dict]] = {}
    for item in results:
        by_method.setdefault(item["method"], []).append(item)

    method_stats: dict[str, Any] = {}
    for method, runs in by_method.items():
        claim_stats = {}
        all_claim_keys = sorted(set(k for r in runs for k in r.get("claim_scores", {})))
        for key in all_claim_keys:
            vals = np.asarray([float(r["claim_scores"][key]) for r in runs if isinstance(r.get("claim_scores", {}).get(key), (int, float, bool))], dtype=float)
            if vals.size:
                claim_stats[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": int(vals.size)}
        legacy_scores = np.asarray([float(r.get("legacy_composite_score", 0.0)) for r in runs], dtype=float)
        diagnostic = claim_stats.get("primary_claim_average_diagnostic", {"mean": 0.0, "std": 0.0, "n": 0})
        method_stats[method] = {
            "primary_claim_average_diagnostic": diagnostic,
            "claim_scores": claim_stats,
            "legacy_composite_score": {
                "mean": float(np.mean(legacy_scores)) if legacy_scores.size else 0.0,
                "std": float(np.std(legacy_scores)) if legacy_scores.size else 0.0,
                "n": int(legacy_scores.size),
            },
            "n": len(runs),
        }

    claim_rankings: dict[str, list[str]] = {}
    for claim in PRIMARY_CLAIMS + SECONDARY_CLAIMS:
        claim_rankings[claim] = sorted(
            method_stats,
            key=lambda m: method_stats[m]["claim_scores"].get(claim, {}).get("mean", -1.0),
            reverse=True,
        )
    diagnostic_ranking = sorted(method_stats, key=lambda m: method_stats[m]["primary_claim_average_diagnostic"]["mean"], reverse=True)
    return {
        "method_stats": method_stats,
        "claim_rankings": claim_rankings,
        "diagnostic_ranking": diagnostic_ranking,
        # Backward-compatible name; reports no longer treat this as a scalar objective.
        "ranking": diagnostic_ranking,
        "primary_claims": PRIMARY_CLAIMS,
        "secondary_claims": SECONDARY_CLAIMS,
    }


def _write_csv(out: Path, results: list[dict]) -> None:
    rows = []
    keys = set()
    for item in results:
        row = {
            "world": item["world"],
            "method": item["method"],
            "seed": item["seed"],
            "phase": item.get("phase", "main"),
            "primary_claim_average_diagnostic": item["claim_scores"].get("primary_claim_average_diagnostic", 0.0),
            "legacy_composite_score": item.get("legacy_composite_score", 0.0),
            "out": item["out"],
        }
        row.update({f"claim_{k}": v for k, v in item["claim_scores"].items()})
        row.update({f"m_{k}": v for k, v in item["metrics"].items()})
        rows.append(row)
        keys.update(row.keys())
    priority = ["world", "method", "seed", "phase", "primary_claim_average_diagnostic", "legacy_composite_score", "out"]
    fieldnames = priority + sorted(k for k in keys if k not in set(priority))
    with (out / "benchmark_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_claim_csv(out: Path, aggregate: dict) -> None:
    rows = []
    for method, stats in aggregate["method_stats"].items():
        row = {"method": method}
        for claim, cstats in stats["claim_scores"].items():
            row[f"{claim}_mean"] = cstats["mean"]
            row[f"{claim}_std"] = cstats["std"]
        rows.append(row)
    if not rows:
        return
    keys = ["method"] + sorted({k for row in rows for k in row if k != "method"})
    with (out / "benchmark_claim_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _mean_matrix_for_method(results: list[dict], method: str, matrix_name: str) -> tuple[np.ndarray, list[str]] | None:
    mats = []
    domain_order: list[str] | None = None
    for r in results:
        if r["method"] != method:
            continue
        if matrix_name in r.get("domain_matrices", {}):
            mats.append(np.asarray(r["domain_matrices"][matrix_name], dtype=float))
            domain_order = r["domain_order"]
    if not mats or domain_order is None:
        return None
    return np.mean(np.stack(mats, axis=0), axis=0), domain_order


def _write_method_matrix_files(out: Path, results: list[dict], aggregate: dict) -> None:
    matrix_dir = ensure_dir(out / "method_domain_matrices")
    for method in aggregate["method_stats"]:
        for name in ["D_op", "dangerous_unmatched", "forbidden_mass", "false_collapse"]:
            got = _mean_matrix_for_method(results, method, name)
            if got is None:
                continue
            M, domain_order = got
            _write_matrix_csv(matrix_dir / f"{method}_{name}_mean.csv", M, domain_order)


def _write_otg_cost_weight_inspection(out: Path, cfg: dict, results: list[dict]) -> dict:
    otg_runs = [r for r in results if r["method"] == "otg"]
    if not otg_runs:
        return {}
    rows = []
    component_keys = sorted(set(k for r in otg_runs for k in r.get("cost_component_summary", {})))
    for r in otg_runs:
        row = {"seed": r["seed"], "world": r["world"]}
        row.update(r.get("cost_component_summary", {}))
        row.update({f"claim_{k}": v for k, v in r.get("claim_scores", {}).items() if isinstance(v, (int, float, bool))})
        rows.append(row)
    keys = ["world", "seed"] + sorted(k for row in rows for k in row if k not in {"world", "seed"})
    with (out / "otg_cost_weight_inspection.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    means = {}
    for k in component_keys:
        vals = [float(r.get("cost_component_summary", {}).get(k, 0.0)) for r in otg_runs]
        means[k] = float(np.mean(vals)) if vals else 0.0
    weights = cfg.get("cost", {}).get("weights", {})
    lines = [
        "# OTG Cost Weight Inspection",
        "",
        "This is an inspection artifact, not a tuning pass. It reports how the current OTG cost decomposes under the graph-level benchmark.",
        "",
        "## Configured OTG cost weights",
        "",
        "| Component | Weight |",
        "|---|---:|",
    ]
    for k, v in sorted(weights.items()):
        lines.append(f"| `{k}` | {float(v):.4f} |")
    lines.extend(["", "## Observed component contributions", "", "| Diagnostic | Mean |", "|---|---:|"])
    for k, v in sorted(means.items()):
        lines.append(f"| `{k}` | {v:.6f} |")
    lines.extend([
        "",
        "Interpretation rule: if `mean_ordinary_vs_operational_gap` and `mean_false_separation_score` are large while harmless-collapse is weak, the operational/risk/invariance terms are probably too conservative for harmless nuisance collapse.",
    ])
    (out / "otg_cost_weight_inspection.md").write_text("\n".join(lines), encoding="utf-8")
    return {"weights": weights, "component_means": means, "out_csv": str(out / "otg_cost_weight_inspection.csv"), "out_markdown": str(out / "otg_cost_weight_inspection.md")}


def _write_report(out: Path, cfg: dict, results: list[dict], aggregate: dict, inspection: dict) -> None:
    lines = [
        "# OTG Graph-Domain-Node Benchmark",
        "",
        "The benchmark object is the full multi-domain DAG-induced node-law tensor, not an isolated source-target pair. Pairwise OT appears only as the numerical primitive used to populate `W[v,i,j]` and the system matrix `D_op[i,j]`.",
        "",
        "Dangerous unmatched mass is tested by switching to unbalanced OT when the node-pair shows terminally induced high-risk unmatched mass; the old target-domain fallback is disabled by default.",
        "OTG uses the configured adaptive node-pair cost profile when `benchmark.otg_cost_mode=adaptive_node_pair`; the gate is computed from node-wise risk, terminal, and failure shifts, not from domain names.",
        "",
        "## Provenance",
        "",
        "| Method | Paper | Kind | Source |",
        "|---|---|---|---|",
    ]
    for method in cfg.get("methods") or available_methods():
        prov = method_provenance(method)
        lines.append(f"| `{method}` | {prov['paper']} | {prov['implementation_kind']} | {prov['implementation_source']} |")

    lines.extend(["", "## Primary claim scores", "", "Scores are separated by claim. `Primary mean` is a diagnostic average only; it is not used as a substitute for claim-level interpretation.", "", "| Method | Primary mean | Harmless collapse | Harmful separation | Admissibility | False collapse avoided | Dangerous unmatched | Localization |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for method in aggregate["diagnostic_ranking"]:
        stats = aggregate["method_stats"][method]
        claims = stats["claim_scores"]
        def mean(k: str) -> float:
            return float(claims.get(k, {}).get("mean", 0.0))
        lines.append(
            f"| `{method}` | {mean('primary_claim_average_diagnostic'):.3f} | {mean('harmless_operational_collapse'):.3f} | {mean('harmful_operational_separation'):.3f} | {mean('admissibility_respected'):.3f} | {mean('false_collapse_avoided'):.3f} | {mean('dangerous_unmatched_exposed'):.3f} | {mean('system_level_localization'):.3f} |"
        )

    lines.extend(["", "## Claim-wise rankings", ""])
    for claim in PRIMARY_CLAIMS:
        ranking = aggregate["claim_rankings"].get(claim, [])
        lines.append(f"- `{claim}`: " + ", ".join(f"`{m}`" for m in ranking))

    lines.extend([
        "",
        "## Matrix outputs",
        "",
        "Per-method mean domain matrices are written under `method_domain_matrices/`. Each method has `D_op`, `dangerous_unmatched`, `forbidden_mass`, and `false_collapse` matrices when available.",
        "",
        "## OTG cost inspection",
        "",
    ])
    if inspection:
        lines.append(f"Cost inspection CSV: `{Path(inspection['out_csv']).name}`")
        lines.append(f"Cost inspection note: `{Path(inspection['out_markdown']).name}`")
    else:
        lines.append("OTG was not included in this benchmark run.")
    (out / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def _provenance_catalog(methods: list[str]) -> dict:
    return {method: method_provenance(method) for method in methods}


def create_bundle_zip(out_dir: str | Path, zip_path: str | Path) -> Path:
    out = Path(out_dir)
    zp = Path(zip_path)
    if zp.exists():
        zp.unlink()
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for path in out.rglob("*"):
            if path.is_file():
                z.write(path, path.relative_to(out))
    return zp
