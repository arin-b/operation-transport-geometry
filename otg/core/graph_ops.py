from __future__ import annotations

from itertools import combinations, permutations
from typing import Iterable
import numpy as np

from otg.core.adaptive import localization_score_from_result, terminal_sensitivity, pair_shift_summary
from otg.core.schema import GraphWorldBatch, PairwiseNodeProblem, SystemComparisonResult


def domain_pairs(domains: Iterable[str], ordered: bool = False) -> list[tuple[str, str]]:
    ids = list(domains)
    if ordered:
        return [(a, b) for a, b in permutations(ids, 2) if a != b]
    return list(combinations(ids, 2))


def derive_pairwise_problem(
    batch: GraphWorldBatch,
    node: str,
    domain_a: str,
    domain_b: str,
    *,
    use_aligned: bool = True,
) -> PairwiseNodeProblem:
    law_a = batch.law(node, domain_a)
    law_b = batch.law(node, domain_b)
    z_a = law_a.aligned_samples if use_aligned else law_a.samples
    z_b = law_b.aligned_samples if use_aligned else law_b.samples
    problem = PairwiseNodeProblem(
        node=node,
        domain_a=domain_a,
        domain_b=domain_b,
        z_a=np.asarray(z_a, dtype=float),
        z_b=np.asarray(z_b, dtype=float),
        raw_z_a=np.asarray(law_a.samples, dtype=float),
        raw_z_b=np.asarray(law_b.samples, dtype=float),
        terminal_a=np.asarray(law_a.terminal_outputs, dtype=float),
        terminal_b=np.asarray(law_b.terminal_outputs, dtype=float),
        true_risk_a=np.asarray(law_a.true_risk, dtype=float),
        true_risk_b=np.asarray(law_b.true_risk, dtype=float),
        used_risk_a=np.asarray(law_a.used_risk, dtype=float),
        used_risk_b=np.asarray(law_b.used_risk, dtype=float),
        operational_coords_a=np.asarray(law_a.operational_coords, dtype=float),
        operational_coords_b=np.asarray(law_b.operational_coords, dtype=float),
        nuisance_coords_a=np.asarray(law_a.nuisance_coords, dtype=float),
        nuisance_coords_b=np.asarray(law_b.nuisance_coords, dtype=float),
        descriptor_coords_a=np.asarray(law_a.descriptor_coords, dtype=float),
        descriptor_coords_b=np.asarray(law_b.descriptor_coords, dtype=float),
        anchors_a=np.asarray(law_a.semantic_anchors),
        anchors_b=np.asarray(law_b.semantic_anchors),
        mass_a=np.asarray(law_a.mass, dtype=float),
        mass_b=np.asarray(law_b.mass, dtype=float),
        failure_a=np.asarray(law_a.failure, dtype=bool),
        failure_b=np.asarray(law_b.failure, dtype=bool),
        projection_a=law_a.projection_metadata,
        projection_b=law_b.projection_metadata,
        metadata={
            "graph_world": batch.name,
            "node": node,
            "domain_a": domain_a,
            "domain_b": domain_b,
            "use_aligned": use_aligned,
            "projection_a": law_a.projection_metadata,
            "projection_b": law_b.projection_metadata,
            "terminal_evaluation": batch.terminal_evaluation_metadata,
        },
    )
    problem.validate()
    return problem


def _node_aggregation_weight(result, batch: GraphWorldBatch, domain_pair: tuple[str, str], cfg: dict) -> float:
    problem = result.pairwise_problem
    if problem is None:
        return 1.0
    base = float(cfg.get("aggregation", {}).get("weights", {}).get(result.node, 1.0))
    sens = terminal_sensitivity(problem, cfg)
    shifts = pair_shift_summary(problem)
    evidence_floor = float(cfg.get("aggregation", {}).get("evidence_floor", 0.05))
    evidence = evidence_floor + shifts["risk_shift"] + shifts["terminal_shift"] + shifts["failure_shift"]
    return float(base * sens * evidence)


def aggregate_node_pair_results(node_results: dict, batch: GraphWorldBatch, domain_pair: tuple[str, str], cfg: dict) -> dict:
    mode = cfg.get("aggregation", {}).get("mode", "terminal_sensitivity_weighted")
    weights_cfg = cfg.get("aggregation", {}).get("weights", {})
    values = {node: float(res.transport.value) for node, res in node_results.items()}
    if not values:
        return {"mode": mode, "value": 0.0, "node_values": values}

    raw_weights: dict[str, float] = {}
    localization_scores: dict[str, float] = {}
    for node, res in node_results.items():
        if mode == "risk_weighted":
            la = batch.law(node, domain_pair[0])
            lb = batch.law(node, domain_pair[1])
            raw_weights[node] = float((np.mean(la.true_risk) + np.mean(lb.true_risk)) / 2.0 + 1e-9)
        elif mode in {"terminal_sensitivity_weighted", "sensitivity_weighted", "null_corrected_localization"}:
            raw_weights[node] = _node_aggregation_weight(res, batch, domain_pair, cfg)
        else:
            raw_weights[node] = float(weights_cfg.get(node, 1.0))
        localization_scores[node] = localization_score_from_result(res, cfg)

    if mode == "max":
        value = max(values.values())
        normalized_weights = {node: 1.0 if values[node] == value else 0.0 for node in values}
    elif mode == "null_corrected_localization":
        loc_sum = float(sum(localization_scores.values()))
        value = loc_sum
        denom = loc_sum if loc_sum > 1e-12 else 1.0
        normalized_weights = {node: localization_scores[node] / denom for node in values}
    elif mode in {"terminal_sensitivity_weighted", "sensitivity_weighted", "risk_weighted"}:
        denom = float(sum(max(w, 0.0) for w in raw_weights.values()))
        if denom <= 1e-12:
            normalized_weights = {node: 1.0 / len(values) for node in values}
        else:
            normalized_weights = {node: max(raw_weights[node], 0.0) / denom for node in values}
        value = sum(normalized_weights[node] * values[node] for node in values)
        if bool(cfg.get("aggregation", {}).get("claim_activation", True)) and mode in {"terminal_sensitivity_weighted", "sensitivity_weighted"}:
            gates = {
                node: float(getattr(res.cost, "weights", {}).get("adaptive_gate", 1.0))
                for node, res in node_results.items()
            }
            activation_floor = float(cfg.get("aggregation", {}).get("claim_activation_floor", 0.0))
            claim_activation = max(activation_floor, sum(normalized_weights[node] * gates[node] for node in values))
            value *= claim_activation
        else:
            claim_activation = 1.0
    else:
        normalized_weights = {node: float(weights_cfg.get(node, 1.0)) for node in values}
        value = sum(normalized_weights[node] * val for node, val in values.items())
        claim_activation = 1.0

    if localization_scores and max(localization_scores.values()) > 1e-12:
        localizing_node = max(localization_scores, key=lambda n: localization_scores[n])
    elif raw_weights:
        localizing_node = max(raw_weights, key=lambda n: raw_weights[n])
    else:
        localizing_node = max(values, key=values.get)
    return {
        "mode": mode,
        "value": float(value),
        "node_values": values,
        "node_weights": normalized_weights,
        "raw_node_weights": raw_weights,
        "claim_activation": float(claim_activation),
        "localization_scores": localization_scores,
        "localizing_node": localizing_node,
    }


def discrepancy_matrix(comparisons: dict[tuple[str, str], SystemComparisonResult], domain_order: list[str]) -> np.ndarray:
    n = len(domain_order)
    idx = {d: i for i, d in enumerate(domain_order)}
    M = np.zeros((n, n), dtype=float)
    for (a, b), result in comparisons.items():
        i, j = idx[a], idx[b]
        val = float(result.aggregate.get("value", 0.0))
        M[i, j] = val
        M[j, i] = val
    return M


def node_pair_table(comparisons: dict[tuple[str, str], SystemComparisonResult]) -> list[dict]:
    rows = []
    for (a, b), comp in comparisons.items():
        for node, res in comp.node_results.items():
            row = {
                "domain_a": a,
                "domain_b": b,
                "node": node,
                "transport_value": float(res.transport.value),
                "solver": res.transport.solver,
                "status": res.transport.status,
                "aggregation_weight": float(comp.aggregate.get("node_weights", {}).get(node, 0.0)),
                "localization_score": float(comp.aggregate.get("localization_scores", {}).get(node, 0.0)),
            }
            for k, v in res.diagnostics.items():
                if isinstance(v, (int, float, bool, str)):
                    row[k] = v
            rows.append(row)
    return rows
