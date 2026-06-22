from otg.core.graph_ops import derive_pairwise_problem
from otg.costs.operational import build_cost
from otg.invariance.registry import make_invariance_builder
from otg.utils.config import deep_merge, load_config
from otg.utils.presets import apply_runtime_preset
from otg.utils.seeding import SeedBank
from otg.worlds.registry import make_world


def _problem(node: str, a: str, b: str):
    cfg = apply_runtime_preset(deep_merge(load_config("configs/default.yaml"), {"runtime_values": {"n": 32}}))
    seeds = SeedBank(0)
    batch = make_world("synthetic_dag", cfg, seeds).generate()
    inv = make_invariance_builder("hybrid", cfg, seeds)
    p = derive_pairwise_problem(batch, node, a, b, use_aligned=True)
    return cfg, seeds, p, inv.build(p)


def test_domain_pair_adaptive_gate_relaxes_harmless_viewpoint_shift():
    cfg, _, p, inv = _problem("measurement", "clear", "viewpoint_shift")
    cost = build_cost(p, inv, cfg)
    assert cost.mode in {"domain_pair_adaptive", "adaptive_node_pair"}
    assert cost.weights["adaptive_gate"] < 0.15
    assert cost.weights["risk"] < 0.5
    assert cost.weights["invariance"] < 0.5


def test_domain_pair_adaptive_gate_activates_for_harmful_glare():
    cfg, _, p, inv = _problem("measurement", "clear", "glare")
    cost = build_cost(p, inv, cfg)
    assert cost.weights["adaptive_gate"] > 0.95
    assert cost.weights["risk"] > 1.9
    assert cost.weights["invariance"] > 2.9
    assert cost.weights["terminal_sensitivity"] > 0.0
