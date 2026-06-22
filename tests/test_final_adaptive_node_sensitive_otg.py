from otg.core.adaptive import select_solver_name, should_use_unbalanced, localization_score_from_result
from otg.core.graph_ops import aggregate_node_pair_results, derive_pairwise_problem
from otg.costs.operational import build_cost
from otg.invariance.registry import make_invariance_builder
from otg.risks.registry import make_risk_model
from otg.transport.registry import make_solver
from otg.core.schema import TransportProblem, NodePairResult
from otg.diagnostics.metrics import compute_node_diagnostics
from otg.utils.config import deep_merge, load_config
from otg.utils.presets import apply_runtime_preset
from otg.utils.seeding import SeedBank
from otg.worlds.registry import make_world


def _batch_cfg(n=32):
    cfg = deep_merge(load_config("configs/default.yaml"), {"runtime_values": {"n": n}, "transport": {"solver": "auto"}})
    cfg = apply_runtime_preset(cfg)
    seeds = SeedBank(0)
    return cfg, seeds, make_world("synthetic_dag", cfg, seeds).generate()


def _node_result(cfg, seeds, batch, node, a, b):
    p = derive_pairwise_problem(batch, node, a, b, use_aligned=True)
    risk = make_risk_model("true", cfg, seeds).estimate(p)
    p.used_risk_a = risk.used_risk_a
    p.used_risk_b = risk.used_risk_b
    inv = make_invariance_builder("hybrid", cfg, seeds).build(p)
    cost = build_cost(p, inv, cfg)
    solver_name = select_solver_name(p, cfg, default="auto")
    tr = make_solver(solver_name, cfg, seeds).solve(TransportProblem(p.mass_a, p.mass_b, cost.total, allowed=inv.allowed, penalty=inv.penalty))
    diag = compute_node_diagnostics(p, inv, cost, tr, cfg)
    for k, v in cost.weights.items():
        if isinstance(v, (int, float, bool)):
            diag[f"cost_weight_{k}"] = float(v)
    return NodePairResult(node=node, domain_pair=(a, b), cost=cost, invariance=inv, transport=tr, diagnostics=diag, pairwise_problem=p)


def test_adaptive_node_pair_cost_uses_node_sensitive_metadata():
    cfg, seeds, batch = _batch_cfg()
    p = derive_pairwise_problem(batch, "measurement", "clear", "glare", use_aligned=True)
    inv = make_invariance_builder("hybrid", cfg, seeds).build(p)
    cost = build_cost(p, inv, cfg)
    assert cost.mode == "adaptive_node_pair"
    assert cost.weights["adaptive_gate"] > 0.95
    assert cost.weights["terminal_sensitivity"] > 0.0
    assert cost.weights["risk"] > 2.0


def test_auto_solver_activates_unbalanced_from_dangerous_mass_not_domain_name():
    cfg, seeds, batch = _batch_cfg()
    p_safe = derive_pairwise_problem(batch, "measurement", "clear", "viewpoint_shift", use_aligned=True)
    p_danger = derive_pairwise_problem(batch, "measurement", "clear", "occlusion", use_aligned=True)
    assert not should_use_unbalanced(p_safe, cfg)
    assert should_use_unbalanced(p_danger, cfg)
    assert select_solver_name(p_safe, cfg, default="auto") == "masked_sinkhorn"
    assert select_solver_name(p_danger, cfg, default="auto") == "unbalanced"


def test_terminal_sensitivity_aggregation_localizes_measurement_for_glare():
    cfg, seeds, batch = _batch_cfg()
    results = {node: _node_result(cfg, seeds, batch, node, "clear", "glare") for node in batch.selected_nodes}
    agg = aggregate_node_pair_results(results, batch, ("clear", "glare"), cfg)
    assert agg["mode"] == "terminal_sensitivity_weighted"
    assert agg["localizing_node"] == "measurement"
    assert localization_score_from_result(results["measurement"], cfg) >= 0.0
