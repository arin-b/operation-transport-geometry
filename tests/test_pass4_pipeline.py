from otg.core.pipeline import run_pipeline
from otg.utils.config import load_config, deep_merge


def test_pass4_pipeline_masked_sinkhorn(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg = deep_merge(cfg, {
        "world": {"name": "admissibility_stress"},
        "transport": {"solver": "masked_sinkhorn"},
        "runtime_values": {"n": 40, "mc_rollouts": 8},
    })
    artifact = run_pipeline(cfg, tmp_path)
    diag = artifact.node_results["repr"].diagnostics
    assert "transport_forbidden_mass" in diag
    assert "row_feasible_fraction" in diag


def test_pass4_pipeline_unbalanced(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg = deep_merge(cfg, {
        "world": {"name": "unbalanced_dangerous_mass"},
        "transport": {"solver": "unbalanced"},
        "runtime_values": {"n": 40, "mc_rollouts": 8},
    })
    artifact = run_pipeline(cfg, tmp_path)
    diag = artifact.node_results["repr"].diagnostics
    assert "dangerous_unmatched_mass_b" in diag
    assert "operational_nonsubstitutable_unmatched_b" in diag
