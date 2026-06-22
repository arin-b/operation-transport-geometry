from __future__ import annotations

from typing import Any
import numpy as np

from otg.core.schema import PairwiseNodeProblem


def sigmoid(x: float) -> float:
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


def mean_l2_gap(A: np.ndarray, B: np.ndarray) -> float:
    A = np.asarray(A, dtype=float).reshape(len(A), -1)
    B = np.asarray(B, dtype=float).reshape(len(B), -1)
    return float(np.linalg.norm(np.mean(A, axis=0) - np.mean(B, axis=0)))


def _corr_sensitivity(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(len(x), -1)
    y = np.asarray(y, dtype=float).reshape(-1)
    vals = []
    if np.std(y) <= 1e-12:
        return 0.0
    for j in range(x.shape[1]):
        col = x[:, j]
        if np.std(col) <= 1e-12:
            continue
        c = np.corrcoef(col, y)[0, 1]
        if np.isfinite(c):
            vals.append(abs(float(c)))
    return float(np.mean(vals)) if vals else 0.0


def node_profile(node_name: str, cfg: dict) -> dict[str, float]:
    profiles = cfg.get("cost", {}).get("node_profiles", {})
    defaults = {
        "detector": {
            "sensitivity_prior": 0.75,
            "geometry_scale": 0.85,
            "terminal_scale": 0.70,
            "risk_scale": 0.75,
            "invariance_scale": 0.85,
            "operational_scale": 0.85,
            "nuisance_scale": 0.50,
        },
        "representation": {
            "sensitivity_prior": 0.90,
            "geometry_scale": 0.80,
            "terminal_scale": 0.70,
            "risk_scale": 0.85,
            "invariance_scale": 0.75,
            "operational_scale": 0.90,
            "nuisance_scale": 0.35,
        },
        "measurement": {
            "sensitivity_prior": 1.30,
            "geometry_scale": 1.00,
            "terminal_scale": 1.30,
            "risk_scale": 1.25,
            "invariance_scale": 1.15,
            "operational_scale": 1.25,
            "nuisance_scale": 0.20,
        },
    }
    out = dict(defaults.get(node_name, {
        "sensitivity_prior": 1.0,
        "geometry_scale": 1.0,
        "terminal_scale": 1.0,
        "risk_scale": 1.0,
        "invariance_scale": 1.0,
        "operational_scale": 1.0,
        "nuisance_scale": 0.5,
    }))
    out.update({k: float(v) for k, v in profiles.get(node_name, {}).items()})
    return out


def pair_shift_summary(node: PairwiseNodeProblem) -> dict[str, float]:
    risk_shift = abs(float(np.mean(node.used_risk_b) - np.mean(node.used_risk_a)))
    true_risk_shift = abs(float(np.mean(node.true_risk_b) - np.mean(node.true_risk_a)))
    terminal_shift = mean_l2_gap(node.terminal_a, node.terminal_b)
    failure_shift = abs(float(np.mean(node.failure_b.astype(float)) - np.mean(node.failure_a.astype(float))))
    op_shift = mean_l2_gap(node.operational_coords_a, node.operational_coords_b)
    nuisance_shift = mean_l2_gap(node.nuisance_coords_a, node.nuisance_coords_b)
    source_danger = float(np.mean(node.failure_a.astype(float)))
    target_danger = float(np.mean(node.failure_b.astype(float)))
    dangerous_mass_gap = abs(target_danger - source_danger)
    return {
        "risk_shift": risk_shift,
        "true_risk_shift": true_risk_shift,
        "terminal_shift": terminal_shift,
        "failure_shift": failure_shift,
        "operational_shift": op_shift,
        "nuisance_shift": nuisance_shift,
        "nuisance_ratio": float(nuisance_shift / max(op_shift + terminal_shift + risk_shift, 1e-12)),
        "source_danger_rate": source_danger,
        "target_danger_rate": target_danger,
        "dangerous_mass_gap": dangerous_mass_gap,
    }


def terminal_sensitivity(node: PairwiseNodeProblem, cfg: dict | None = None) -> float:
    cfg = cfg or {}
    profile = node_profile(node.node, cfg)
    s_a = _corr_sensitivity(node.operational_coords_a, node.true_risk_a)
    s_b = _corr_sensitivity(node.operational_coords_b, node.true_risk_b)
    terminal_corr_a = _corr_sensitivity(node.terminal_a.reshape(len(node.terminal_a), -1), node.true_risk_a)
    terminal_corr_b = _corr_sensitivity(node.terminal_b.reshape(len(node.terminal_b), -1), node.true_risk_b)
    empirical = 0.65 * ((s_a + s_b) / 2.0) + 0.35 * ((terminal_corr_a + terminal_corr_b) / 2.0)
    floor = float(cfg.get("aggregation", {}).get("sensitivity_floor", 0.10))
    return float(max(floor, profile.get("sensitivity_prior", 1.0) * (0.50 + empirical)))


def adaptive_node_pair_weights(node: PairwiseNodeProblem, base: dict[str, float], cfg: dict) -> dict[str, float]:
    """Node-sensitive, domain-pair adaptive OTG weights.

    The gate is terminally induced. It does not use domain names. Harmless
    nuisance shifts receive low risk/terminal/invariance weights; harmful shifts
    recover conservative OTG weights, with node-specific profiles.
    """
    acfg = cfg.get("cost", {}).get("adaptive", {})
    prof = node_profile(node.node, cfg)
    s = pair_shift_summary(node)

    risk_threshold = float(acfg.get("risk_threshold", 0.060))
    risk_scale = float(acfg.get("risk_scale", 0.040))
    terminal_threshold = float(acfg.get("terminal_threshold", 0.100))
    terminal_scale = float(acfg.get("terminal_scale", 0.060))
    failure_threshold = float(acfg.get("failure_threshold", 0.050))
    failure_scale = float(acfg.get("failure_scale", 0.040))
    gate_bias = float(acfg.get("gate_bias", -0.50))
    gate_weights = acfg.get("gate_weights", {})
    wr = float(gate_weights.get("risk", 1.60))
    wt = float(gate_weights.get("terminal", 1.00))
    wf = float(gate_weights.get("failure", 0.80))

    risk_term = (s["risk_shift"] - risk_threshold) / max(risk_scale, 1e-12)
    terminal_term = (s["terminal_shift"] - terminal_threshold) / max(terminal_scale, 1e-12)
    failure_term = (s["failure_shift"] - failure_threshold) / max(failure_scale, 1e-12)
    raw_gate = gate_bias + wr * risk_term + wt * terminal_term + wf * failure_term
    gate = sigmoid(raw_gate)

    harmless_nuisance = (1.0 - gate) * sigmoid(
        float(acfg.get("nuisance_gate_slope", 0.75))
        * (s["nuisance_ratio"] - float(acfg.get("nuisance_ratio_threshold", 1.0)))
    )

    min_weights = acfg.get("min_weights", {})
    mins = {
        "geometry": float(min_weights.get("geometry", 0.35)),
        "terminal": float(min_weights.get("terminal", 0.10)),
        "risk": float(min_weights.get("risk", 0.20)),
        "invariance": float(min_weights.get("invariance", 0.15)),
        "operational_coordinate": float(min_weights.get("operational_coordinate", 0.25)),
        "nuisance": float(min_weights.get("nuisance", 0.02)),
    }
    profile_scales = {
        "geometry": prof.get("geometry_scale", 1.0),
        "terminal": prof.get("terminal_scale", 1.0),
        "risk": prof.get("risk_scale", 1.0),
        "invariance": prof.get("invariance_scale", 1.0),
        "operational_coordinate": prof.get("operational_scale", 1.0),
        "nuisance": prof.get("nuisance_scale", 1.0),
    }
    eff: dict[str, float] = {}
    for key, b in base.items():
        b_scaled = float(b) * float(profile_scales.get(key, 1.0))
        lo = min(b_scaled, float(mins.get(key, b_scaled)))
        eff[key] = lo + gate * (b_scaled - lo)

    quotient_strength = float(acfg.get("quotient_nuisance_strength", 0.35))
    relaxation = 1.0 - quotient_strength * harmless_nuisance
    eff["geometry"] *= relaxation
    eff["nuisance"] *= relaxation

    sens = terminal_sensitivity(node, cfg)
    eff.update({
        "adaptive_gate": float(gate),
        "adaptive_raw_gate": float(raw_gate),
        "adaptive_harmless_nuisance": float(harmless_nuisance),
        "terminal_sensitivity": float(sens),
        "node_sensitivity_prior": float(prof.get("sensitivity_prior", 1.0)),
        "adaptive_risk_shift": s["risk_shift"],
        "adaptive_true_risk_shift": s["true_risk_shift"],
        "adaptive_terminal_shift": s["terminal_shift"],
        "adaptive_failure_shift": s["failure_shift"],
        "adaptive_operational_shift": s["operational_shift"],
        "adaptive_nuisance_shift": s["nuisance_shift"],
        "adaptive_nuisance_ratio": s["nuisance_ratio"],
        "adaptive_dangerous_mass_gap": s["dangerous_mass_gap"],
    })
    return eff


def should_use_unbalanced(node: PairwiseNodeProblem, cfg: dict) -> bool:
    tcfg = cfg.get("transport", {}).get("auto", {})
    bcfg = cfg.get("benchmark", {}).get("dangerous_unmatched", {})
    if not bool(tcfg.get("enabled", True)) and not bool(bcfg.get("enabled", False)):
        return False
    high_risk_threshold = float(tcfg.get("high_risk_threshold", bcfg.get("high_risk_threshold", 0.50)))
    gap_threshold = float(tcfg.get("dangerous_mass_gap_threshold", bcfg.get("dangerous_mass_gap_threshold", 0.12)))
    min_target_rate = float(tcfg.get("min_target_danger_rate", bcfg.get("min_target_danger_rate", 0.12)))
    risk_gap_threshold = float(tcfg.get("risk_shift_threshold", bcfg.get("risk_shift_threshold", 0.10)))
    fa = node.true_risk_a > high_risk_threshold
    fb = node.true_risk_b > high_risk_threshold
    rate_a = float(np.mean(fa))
    rate_b = float(np.mean(fb))
    risk_shift = abs(float(np.mean(node.true_risk_b) - np.mean(node.true_risk_a)))
    danger_gap = abs(rate_b - rate_a)
    return bool((max(rate_a, rate_b) >= min_target_rate and danger_gap >= gap_threshold) or (max(rate_a, rate_b) >= min_target_rate and risk_shift >= risk_gap_threshold))


def select_solver_name(node: PairwiseNodeProblem, cfg: dict, *, default: str | None = None, method_name: str | None = None) -> str:
    requested = default or str(cfg.get("transport", {}).get("solver", "masked_sinkhorn"))
    if requested not in {"auto", "hybrid_auto"}:
        return requested
    if should_use_unbalanced(node, cfg):
        return str(cfg.get("transport", {}).get("auto", {}).get("unbalanced_solver", "unbalanced"))
    n = min(len(node.mass_a), len(node.mass_b))
    audit_n = int(cfg.get("transport", {}).get("auto", {}).get("lp_audit_n", 24))
    if n <= audit_n and bool(cfg.get("transport", {}).get("auto", {}).get("lp_audit", True)):
        return "lp"
    return str(cfg.get("transport", {}).get("auto", {}).get("balanced_solver", "masked_sinkhorn"))


def localization_score_from_result(result: Any, cfg: dict | None = None) -> float:
    cfg = cfg or {}
    d = result.diagnostics
    value = float(result.transport.value)
    sens = float(d.get("cost_weight_terminal_sensitivity", d.get("terminal_sensitivity", 1.0)))
    gate = float(d.get("cost_weight_adaptive_gate", 1.0))
    harmless = float(d.get("cost_weight_adaptive_harmless_nuisance", 0.0))
    false_sep = float(d.get("false_separation_score", 0.0))
    gap = float(d.get("true_risk_shift", d.get("used_risk_shift", 0.0)))
    failure = float(d.get("failure_mismatch_transport_mass", 0.0))
    null_scale = float(cfg.get("aggregation", {}).get("null_correction", {}).get("false_separation_scale", 0.60))
    null_baseline = null_scale * false_sep * (1.0 + harmless)
    excess = max(0.0, value - null_baseline)
    evidence = max(gate, min(1.0, 2.5 * gap + failure))
    return float(sens * evidence * excess)
