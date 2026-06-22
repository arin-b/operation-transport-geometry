from otg.core.pipeline import run_pipeline
from otg.utils.config import load_config, deep_merge


def test_same_seed_reproducible():
    cfg = load_config("configs/default.yaml")
    cfg = deep_merge(cfg, {
        "world": {"name": "harmful_boundary"},
        "risk": {"mode": "true"},
        "transport": {"solver": "masked_sinkhorn"},
        "runtime_values": {"n": 28, "mc_rollouts": 8},
        "seed": {"master": 777},
    })
    a = run_pipeline(cfg)
    b = run_pipeline(cfg)
    assert a.system_score["value"] == b.system_score["value"]
