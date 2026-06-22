from __future__ import annotations


def assumption_report(batch, node_results: dict, cfg: dict) -> dict:
    report = {
        "world": batch.name,
        "world_expected_behavior": (batch.metadata.get("world_spec") or {}).get("expected_behavior"),
        "nodes": {},
    }

    for node_name, result in node_results.items():
        diag = result.diagnostics
        report["nodes"][node_name] = {
            "risk_accuracy": {
                "mae": diag["risk_estimation_mae"],
                "calibration_error": diag.get("risk_calibration_error", -1.0),
                "failure_accuracy": diag.get("risk_failure_accuracy", -1.0),
                "status": "ok" if diag["risk_estimation_mae"] < 0.20 else "violated_or_stressed",
            },
            "invariance_feasibility": {
                "allowed_pair_fraction": diag["allowed_pair_fraction"],
                "row_feasible_fraction": diag.get("row_feasible_fraction", 1.0),
                "col_feasible_fraction": diag.get("col_feasible_fraction", 1.0),
                "status": "ok" if diag["allowed_pair_fraction"] > 0.05 and diag.get("row_feasible_fraction", 1.0) > 0.80 else "too_restrictive",
            },
            "transport_numerics": {
                "solver": result.transport.solver,
                "status": result.transport.status,
                "plan_mass": diag.get("transport_plan_mass", -1.0),
                "row_l1_error": diag.get("transport_row_l1_error", -1.0),
                "col_l1_error": diag.get("transport_col_l1_error", -1.0),
                "forbidden_mass": diag.get("transport_forbidden_mass", -1.0),
                "status_label": "ok" if result.transport.status.startswith("ok") else "failed",
            },
            "unmatched_mass_decomposition": {
                "finite_sample_unmatched_a": diag.get("finite_sample_unmatched_a", 0.0),
                "finite_sample_unmatched_b": diag.get("finite_sample_unmatched_b", 0.0),
                "operational_nonsubstitutable_unmatched_a": diag.get("operational_nonsubstitutable_unmatched_a", 0.0),
                "operational_nonsubstitutable_unmatched_b": diag.get("operational_nonsubstitutable_unmatched_b", 0.0),
                "dangerous_unmatched_mass_a": diag.get("dangerous_unmatched_mass_a", 0.0),
                "dangerous_unmatched_mass_b": diag.get("dangerous_unmatched_mass_b", 0.0),
            },
            "collapse_separation": {
                "false_collapse_mass": diag["false_collapse_mass"],
                "false_separation_score": diag["false_separation_score"],
                "status": "ok" if diag["false_collapse_mass"] < 0.15 else "false_collapse_stressed",
            },
        }
    return report


def build_assumption_report(world_data, node_results: dict, cfg: dict) -> dict:
    return assumption_report(world_data, node_results, cfg)
