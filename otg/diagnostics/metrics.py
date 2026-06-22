from __future__ import annotations

import numpy as np


def _safe_float(x, default: float = -1.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _gini_like(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0 or np.allclose(x.sum(), 0):
        return 0.0
    x = np.sort(np.maximum(x, 0.0))
    n = x.size
    return float((2 * np.sum((np.arange(1, n + 1) * x)) / (n * np.sum(x))) - (n + 1) / n)


def _unmatched_decomposition(node, inv, tr) -> dict:
    if tr.unmatched_a is None or tr.unmatched_b is None:
        return {
            "finite_sample_unmatched_a": 0.0,
            "finite_sample_unmatched_b": 0.0,
            "operational_nonsubstitutable_unmatched_a": 0.0,
            "operational_nonsubstitutable_unmatched_b": 0.0,
            "dangerous_unmatched_mass_a": 0.0,
            "dangerous_unmatched_mass_b": 0.0,
        }

    row_has = inv.allowed.any(axis=1)
    col_has = inv.allowed.any(axis=0)

    danger_a = node.failure_a.astype(float)
    danger_b = node.failure_b.astype(float)

    finite_a = tr.unmatched_a * row_has.astype(float) * (1.0 - danger_a)
    finite_b = tr.unmatched_b * col_has.astype(float) * (1.0 - danger_b)
    nonsub_a = tr.unmatched_a * (~row_has).astype(float)
    nonsub_b = tr.unmatched_b * (~col_has).astype(float)
    dangerous_a = tr.unmatched_a * danger_a
    dangerous_b = tr.unmatched_b * danger_b

    return {
        "finite_sample_unmatched_a": float(np.sum(finite_a)),
        "finite_sample_unmatched_b": float(np.sum(finite_b)),
        "operational_nonsubstitutable_unmatched_a": float(np.sum(nonsub_a)),
        "operational_nonsubstitutable_unmatched_b": float(np.sum(nonsub_b)),
        "dangerous_unmatched_mass_a": float(np.sum(dangerous_a)),
        "dangerous_unmatched_mass_b": float(np.sum(dangerous_b)),
    }


def compute_node_diagnostics(node, inv, cost, tr, cfg: dict) -> dict:
    plan = np.asarray(tr.plan, dtype=float)
    total_mass = float(np.sum(plan)) if np.sum(plan) > 0 else 1.0

    true_gap = np.abs(node.true_risk_a[:, None] - node.true_risk_b[None, :])
    used_gap = np.abs(node.used_risk_a[:, None] - node.used_risk_b[None, :])
    failure_mismatch = node.failure_a[:, None] != node.failure_b[None, :]
    harmful_mismatch = true_gap > 0.5
    equivalent = true_gap < 0.1

    false_collapse_mass = float(np.sum(plan * harmful_mismatch * (used_gap < 0.25)))
    harmful_transport_mass = float(np.sum(plan * harmful_mismatch))
    failure_mismatch_mass = float(np.sum(plan * failure_mismatch))
    equivalent_transport_mass = float(np.sum(plan * equivalent))
    false_separation_score = float(np.mean(cost.total[equivalent])) if np.any(equivalent) else 0.0

    risk_error = float((np.mean(np.abs(node.used_risk_a - node.true_risk_a)) + np.mean(np.abs(node.used_risk_b - node.true_risk_b))) / 2)
    unmatched_a = float(np.sum(tr.unmatched_a)) if tr.unmatched_a is not None else 0.0
    unmatched_b = float(np.sum(tr.unmatched_b)) if tr.unmatched_b is not None else 0.0

    ordinary_geo_value = float(np.sum(plan * cost.geometry))
    terminal_value = float(np.sum(plan * cost.terminal))
    risk_value = float(np.sum(plan * cost.risk))
    invariance_value = float(np.sum(plan * cost.invariance))
    operational_value = float(np.sum(plan * cost.total))
    ordinary_vs_operational_gap = operational_value - ordinary_geo_value

    meta = inv.metadata or {}
    op_gap = meta.get("operational_gap")
    nuisance_gap = meta.get("nuisance_gap")
    decomp = _unmatched_decomposition(node, inv, tr)

    row_mass = plan.sum(axis=1)
    col_mass = plan.sum(axis=0)
    risk_shift = abs(float(np.mean(node.true_risk_b) - np.mean(node.true_risk_a)))
    used_risk_shift = abs(float(np.mean(node.used_risk_b) - np.mean(node.used_risk_a)))
    harmful_ratio_on_plan = harmful_transport_mass / total_mass
    equivalent_ratio_on_plan = equivalent_transport_mass / total_mass

    out = {
        "transported_harm_mismatch_mass": harmful_transport_mass,
        "failure_mismatch_transport_mass": failure_mismatch_mass,
        "risk_estimation_mae": risk_error,
        "risk_train_error_mae": _safe_float(node.metadata.get("risk_train_error_mae")),
        "risk_calibration_error": _safe_float(node.metadata.get("risk_calibration_error")),
        "risk_failure_accuracy": _safe_float(node.metadata.get("risk_failure_accuracy")),
        "false_collapse_mass": false_collapse_mass,
        "false_separation_score": false_separation_score,
        "infeasible_pair_fraction": float(1.0 - np.mean(inv.allowed)),
        "allowed_pair_fraction": float(np.mean(inv.allowed)),
        "row_feasible_fraction": _safe_float(meta.get("row_feasible_fraction"), 1.0),
        "col_feasible_fraction": _safe_float(meta.get("col_feasible_fraction"), 1.0),
        "unmatched_mass_a": unmatched_a,
        "unmatched_mass_b": unmatched_b,
        **decomp,
        "mean_true_risk_a": float(np.mean(node.true_risk_a)),
        "mean_true_risk_b": float(np.mean(node.true_risk_b)),
        "mean_used_risk_a": float(np.mean(node.used_risk_a)),
        "mean_used_risk_b": float(np.mean(node.used_risk_b)),
        "true_risk_shift": risk_shift,
        "used_risk_shift": used_risk_shift,
        "ordinary_geometric_value_on_plan": ordinary_geo_value,
        "terminal_value_on_plan": terminal_value,
        "risk_value_on_plan": risk_value,
        "invariance_value_on_plan": invariance_value,
        "operational_value_on_plan": operational_value,
        "ordinary_vs_operational_gap": ordinary_vs_operational_gap,
        "harmless_shift_collapse": equivalent_ratio_on_plan,
        "harmful_shift_detection": harmful_ratio_on_plan,
        "risk_gap_transport_mass": float(np.sum(plan * true_gap)),
        "transport_plan_mass": _safe_float(tr.metadata.get("plan_mass"), float(np.sum(plan))),
        "transport_row_l1_error": _safe_float(tr.metadata.get("row_l1_error"), 0.0),
        "transport_col_l1_error": _safe_float(tr.metadata.get("col_l1_error"), 0.0),
        "transport_forbidden_mass": _safe_float(tr.metadata.get("forbidden_mass"), 0.0),
        "transport_mask_respected": bool(tr.metadata.get("mask_respected", tr.solver in {"lp", "masked_sinkhorn", "unbalanced_pot", "unbalanced_fallback"})),
        "transport_row_mass_gini": _gini_like(row_mass),
        "transport_col_mass_gini": _gini_like(col_mass),
    }
    if op_gap is not None:
        out["op_gap_transport_mass"] = float(np.sum(plan * op_gap))
    if nuisance_gap is not None:
        out["nuisance_gap_transport_mass"] = float(np.sum(plan * nuisance_gap))
    return out


def compute_diagnostics(node, cost_pack, adm_pack, tr, cfg: dict) -> dict:
    return compute_node_diagnostics(node, adm_pack, cost_pack, tr, cfg)
