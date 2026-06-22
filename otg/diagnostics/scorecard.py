from __future__ import annotations

from dataclasses import dataclass
from typing import Any


LOWER_IS_BETTER = {
    "risk_estimation_mae",
    "risk_calibration_error",
    "false_collapse_mass",
    "false_separation_score",
    "transport_forbidden_mass",
    "transport_row_l1_error",
    "transport_col_l1_error",
    "dangerous_unmatched_mass_b",
}

HIGHER_IS_BETTER = {
    "harmless_shift_collapse",
    "harmful_shift_detection",
    "risk_failure_accuracy",
    "allowed_pair_fraction",
    "row_feasible_fraction",
    "col_feasible_fraction",
}


def default_thresholds(cfg: dict) -> dict[str, float]:
    user = cfg.get("diagnostics", {}).get("thresholds", {})
    base = {
        "risk_estimation_mae_ok": 0.20,
        "risk_calibration_error_ok": 0.20,
        "false_collapse_mass_ok": 0.15,
        "false_separation_score_ok": 12.0,
        "forbidden_mass_ok": 1e-4,
        "row_l1_error_ok": 1e-3,
        "col_l1_error_ok": 1e-3,
        "allowed_pair_fraction_min": 0.05,
        "row_feasible_fraction_min": 0.80,
        "col_feasible_fraction_min": 0.80,
        "harmless_shift_collapse_min": 0.30,
        "harmful_shift_detection_min": 0.10,
        "dangerous_unmatched_mass_min": 0.05,
    }
    base.update(user)
    return base


def status_from_diagnostics(world_name: str, diag: dict[str, Any], cfg: dict) -> dict[str, Any]:
    t = default_thresholds(cfg)
    checks = {}

    def check_le(name: str, value: float, threshold: float) -> None:
        checks[name] = {"value": float(value), "threshold": float(threshold), "pass": bool(value <= threshold)}

    def check_ge(name: str, value: float, threshold: float) -> None:
        checks[name] = {"value": float(value), "threshold": float(threshold), "pass": bool(value >= threshold)}

    check_le("risk_estimation_mae", diag.get("risk_estimation_mae", 0.0), t["risk_estimation_mae_ok"])
    check_le("risk_calibration_error", max(diag.get("risk_calibration_error", 0.0), 0.0), t["risk_calibration_error_ok"])
    check_le("false_collapse_mass", diag.get("false_collapse_mass", 0.0), t["false_collapse_mass_ok"])
    check_le("transport_forbidden_mass", diag.get("transport_forbidden_mass", 0.0), t["forbidden_mass_ok"])
    check_ge("allowed_pair_fraction", diag.get("allowed_pair_fraction", 1.0), t["allowed_pair_fraction_min"])
    check_ge("row_feasible_fraction", diag.get("row_feasible_fraction", 1.0), t["row_feasible_fraction_min"])
    check_ge("col_feasible_fraction", diag.get("col_feasible_fraction", 1.0), t["col_feasible_fraction_min"])

    world_checks = {}
    if world_name in {"harmless_nuisance", "sample_complexity"}:
        check_ge("harmless_shift_collapse", diag.get("harmless_shift_collapse", 0.0), t["harmless_shift_collapse_min"])
        world_checks["expected"] = "harmless nuisance shift should collapse under operational transport"
    elif world_name in {"harmful_boundary", "high_dimensional", "risk_degradation", "admissibility_stress", "invariance_misspecification"}:
        check_ge("harmful_shift_detection", diag.get("harmful_shift_detection", 0.0), t["harmful_shift_detection_min"])
        world_checks["expected"] = "harmful operational shift should remain visible"
    elif world_name == "unbalanced_dangerous_mass":
        check_ge("dangerous_unmatched_mass_b", diag.get("dangerous_unmatched_mass_b", 0.0), t["dangerous_unmatched_mass_min"])
        world_checks["expected"] = "dangerous target-side unmatched mass should be exposed"

    failures = [name for name, c in checks.items() if not c["pass"]]
    status = "pass" if not failures else ("warn" if len(failures) <= 2 else "fail")

    return {
        "status": status,
        "failed_checks": failures,
        "checks": checks,
        "world_expectation": world_checks.get("expected"),
    }


def interpret_node(world_name: str, node_name: str, diag: dict[str, Any], cfg: dict) -> str:
    score = status_from_diagnostics(world_name, diag, cfg)
    status = score["status"]
    parts = [f"Node `{node_name}` status: `{status}`."]

    if score.get("world_expectation"):
        parts.append(score["world_expectation"] + ".")

    if diag.get("false_collapse_mass", 0.0) > default_thresholds(cfg)["false_collapse_mass_ok"]:
        parts.append("False collapse is elevated: mass is being transported between states with large true-risk discrepancy.")
    if diag.get("false_separation_score", 0.0) > default_thresholds(cfg)["false_separation_score_ok"]:
        parts.append("False separation is elevated: states with similar true risk are expensive to match.")
    if diag.get("transport_forbidden_mass", 0.0) > default_thresholds(cfg)["forbidden_mass_ok"]:
        parts.append("Transport is leaking through forbidden or inadmissible pairs.")
    if diag.get("dangerous_unmatched_mass_b", 0.0) > 0:
        parts.append("There is target-side unmatched dangerous mass.")
    if diag.get("risk_estimation_mae", 0.0) > default_thresholds(cfg)["risk_estimation_mae_ok"]:
        parts.append("Risk estimation error is high enough to affect the transport interpretation.")

    return " ".join(parts)
