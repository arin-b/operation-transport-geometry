from otg.utils.config import load_config, deep_merge
from otg.core.pipeline import run_pipeline


def test_modes_run(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg = deep_merge(cfg, {
        "world": {"name": "synthetic_dag"},
        "risk": {"mode": "noisy"},
        "admissibility": {"mode": "adaptive"},
        "transport": {"solver": "sinkhorn"},
        "runtime_values": {"n": 24, "mc_rollouts": 8},
    })
    artifact = run_pipeline(cfg, tmp_path)
    row = artifact.node_pair_table[0]
    assert "risk_estimation_mae" in row
    assert row["allowed_pair_fraction"] >= 0.0
