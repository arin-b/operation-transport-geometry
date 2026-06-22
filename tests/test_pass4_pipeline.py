from otg.core.pipeline import run_pipeline
from otg.utils.config import load_config, deep_merge


def test_pass4_pipeline_masked_sinkhorn(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg = deep_merge(cfg, {"transport": {"solver": "masked_sinkhorn"}, "runtime_values": {"n": 32, "mc_rollouts": 8}})
    artifact = run_pipeline(cfg, tmp_path)
    row = artifact.node_pair_table[0]
    assert "transport_forbidden_mass" in row
    assert "row_feasible_fraction" in row


def test_pass4_pipeline_unbalanced(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg = deep_merge(cfg, {"transport": {"solver": "unbalanced"}, "runtime_values": {"n": 32, "mc_rollouts": 8}})
    artifact = run_pipeline(cfg, tmp_path)
    vals = [r.get("dangerous_unmatched_mass_b", 0.0) for r in artifact.node_pair_table]
    assert max(vals) >= 0.0
    assert "operational_nonsubstitutable_unmatched_b" in artifact.node_pair_table[0]
