from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import math
import numpy as np


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    value: float | str | bool | None
    threshold: float | str | bool | None
    severity: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_finite_number(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def check_le(name: str, value: float, threshold: float, severity: str, message: str) -> ValidationCheck:
    return ValidationCheck(name, bool(value <= threshold), float(value), float(threshold), severity, message)


def check_ge(name: str, value: float, threshold: float, severity: str, message: str) -> ValidationCheck:
    return ValidationCheck(name, bool(value >= threshold), float(value), float(threshold), severity, message)


def check_true(name: str, value: bool, severity: str, message: str) -> ValidationCheck:
    return ValidationCheck(name, bool(value), bool(value), True, severity, message)


def graph_artifact_checks(case_name: str, artifact) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    checks.append(check_true(f"{case_name}:graph_pipeline", artifact.metadata.get("pipeline") == "graph_level_otg", "error", "Default pipeline is graph-level OTG."))
    checks.append(check_ge(f"{case_name}:num_domains", len(artifact.domain_order), 4, "error", "Graph world has at least four deployment domains."))
    checks.append(check_ge(f"{case_name}:num_selected_nodes", len(artifact.selected_nodes), 3, "error", "Graph world has at least three selected internal nodes."))
    checks.append(check_true(f"{case_name}:D_op_matrix_shape", artifact.discrepancy_matrix.shape == (len(artifact.domain_order), len(artifact.domain_order)), "error", "D_op domain-pair matrix is produced."))
    checks.append(check_true(f"{case_name}:system_score_finite", is_finite_number(artifact.system_score.get("value")), "error", "System score is finite."))
    expected_pairs = len(artifact.domain_order) * (len(artifact.domain_order) - 1) // 2
    checks.append(check_ge(f"{case_name}:num_domain_pairs", len(artifact.comparisons), expected_pairs, "error", "All unordered domain pairs are compared."))

    # Proposal-claim checks.
    pair_values = {pair: float(comp.aggregate.get("value", 0.0)) for pair, comp in artifact.comparisons.items()}
    cv = pair_values.get(("clear", "viewpoint_shift"), pair_values.get(("viewpoint_shift", "clear"), None))
    cg = pair_values.get(("clear", "glare"), pair_values.get(("glare", "clear"), None))
    co = pair_values.get(("clear", "occlusion"), pair_values.get(("occlusion", "clear"), None))
    if cv is not None and cg is not None:
        checks.append(check_true(f"{case_name}:harmful_glare_exceeds_viewpoint", cg > cv, "error", "Harmful glare domain pair should exceed harmless viewpoint shift."))
    if cv is not None and co is not None:
        checks.append(check_true(f"{case_name}:harmful_occlusion_exceeds_viewpoint", co > cv, "error", "Harmful occlusion domain pair should exceed harmless viewpoint shift."))

    for pair, comp in artifact.comparisons.items():
        checks.append(check_true(f"{case_name}:{pair}:aggregate_finite", is_finite_number(comp.aggregate.get("value")), "error", "D_op(d,d') finite."))
        checks.append(check_true(f"{case_name}:{pair}:has_all_nodes", set(comp.node_results) >= set(artifact.selected_nodes), "error", "All selected nodes have W_v,op results."))
        for node, result in comp.node_results.items():
            diag = result.diagnostics
            prefix = f"{case_name}:{pair[0]}::{pair[1]}:{node}"
            checks.append(check_true(f"{prefix}:transport_finite", is_finite_number(result.transport.value), "error", "Node-pair transport value is finite."))
            checks.append(check_true(f"{prefix}:solver_ok", str(result.transport.status).startswith("ok"), "error", "Solver status starts with ok."))
            checks.append(check_le(f"{prefix}:forbidden_mass", float(diag.get("transport_forbidden_mass", 0.0)), 1e-3, "error", "Admissible solver should not leak forbidden mass."))
            checks.append(check_ge(f"{prefix}:allowed_pair_fraction", float(diag.get("allowed_pair_fraction", 0.0)), 0.001, "error", "Admissibility relation should not be empty."))
            if artifact.config.get("risk", {}).get("mode") in {"true", True}:
                checks.append(check_le(f"{prefix}:true_risk_mae", float(diag.get("risk_estimation_mae", 1.0)), 1e-9, "error", "True-risk mode should have zero risk-estimation error."))

    # Unbalanced dangerous mass should be exposed when unbalanced solver is active.
    if artifact.config.get("transport", {}).get("solver") == "unbalanced":
        vals = [float(res.diagnostics.get("dangerous_unmatched_mass_b", 0.0)) for comp in artifact.comparisons.values() for res in comp.node_results.values()]
        checks.append(check_ge(f"{case_name}:dangerous_unmatched_exposed", max(vals) if vals else 0.0, 0.001, "error", "Unbalanced OT exposes dangerous unmatched target mass."))
    return checks


# Backward compatible names.
def artifact_checks(case_name: str, artifact) -> list[ValidationCheck]:
    return graph_artifact_checks(case_name, artifact)


def reproducibility_check(a, b, tolerance: float = 1e-12) -> ValidationCheck:
    va = float(a.system_score.get("value", float("nan")))
    vb = float(b.system_score.get("value", float("nan")))
    return check_le("reproducibility:same_seed_system_score_delta", abs(va - vb), tolerance, "error", "Same seed and same graph config should reproduce the same system score.")
