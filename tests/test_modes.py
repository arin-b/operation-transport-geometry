from otg.utils.config import load_config
from otg.utils.config import deep_merge
from otg.core.pipeline import run_pipeline


def test_modes_run(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg = deep_merge(cfg, {
        "world": {"name": "harmless_nuisance"},
        "risk": {"mode": "noisy"},
        "admissibility": {"mode": "adaptive"},
        "transport": {"solver": "sinkhorn"},
    })
    artifact = run_pipeline(cfg, tmp_path)
    diag = artifact.node_results["repr"].diagnostics
    assert "risk_estimation_mae" in diag
    assert diag["allowed_pair_fraction"] >= 0.0
