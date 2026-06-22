from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import math


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


def artifact_checks(case_name: str, artifact) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    checks.append(check_true(f"{case_name}:artifact_has_nodes", bool(artifact.node_results), "error", "Artifact contains at least one node result."))
    checks.append(check_true(f"{case_name}:system_score_finite", is_finite_number(artifact.system_score.get("value")), "error", "System score is finite."))

    for node_name, result in artifact.node_results.items():
        diag = result.diagnostics
        prefix = f"{case_name}:{node_name}"
        checks.append(check_true(f"{prefix}:transport_value_finite", is_finite_number(result.transport.value), "error", "Transport value is finite."))
        checks.append(check_true(f"{prefix}:solver_status_ok", str(result.transport.status).startswith("ok"), "error", "Solver status starts with ok."))
        checks.append(check_le(f"{prefix}:forbidden_mass", float(diag.get("transport_forbidden_mass", 0.0)), 1e-3, "error", "Admissible solvers should not leak meaningful forbidden mass."))
        checks.append(check_ge(f"{prefix}:allowed_pair_fraction", float(diag.get("allowed_pair_fraction", 0.0)), 0.01, "error", "Admissibility mask should not be empty."))
        checks.append(check_ge(f"{prefix}:row_feasible_fraction", float(diag.get("row_feasible_fraction", 0.0)), 0.50, "warn", "Most source samples should have at least one allowed target."))
        checks.append(check_ge(f"{prefix}:col_feasible_fraction", float(diag.get("col_feasible_fraction", 0.0)), 0.50, "warn", "Most target samples should have at least one allowed source."))
        if artifact.config.get("risk", {}).get("mode") in {"true", True}:
            checks.append(check_le(f"{prefix}:true_risk_mae", float(diag.get("risk_estimation_mae", 1.0)), 1e-9, "error", "True-risk mode should have zero risk-estimation error."))

    return checks


def reproducibility_check(a, b, tolerance: float = 1e-12) -> ValidationCheck:
    va = float(a.system_score.get("value", float("nan")))
    vb = float(b.system_score.get("value", float("nan")))
    return check_le("reproducibility:same_seed_system_score_delta", abs(va - vb), tolerance, "error", "Same seed and same config should reproduce the same system score.")
