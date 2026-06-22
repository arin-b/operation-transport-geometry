from otg.core.pipeline import run_pipeline
from otg.utils.config import load_config, deep_merge


def test_pass5_outputs_exist(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg = deep_merge(cfg, {"runtime_values": {"n": 24, "mc_rollouts": 8}})
    run_pipeline(cfg, tmp_path)
    expected = [
        "summary.json",
        "graph_summary.json",
        "D_op_matrix.csv",
        "node_pair_metrics.csv",
        "report.md",
        "interpretation.md",
        "domain_pair_matrix_table.tex",
    ]
    for name in expected:
        assert (tmp_path / name).exists(), name
    assert (tmp_path / "figures" / "D_op_domain_pair_heatmap.png").exists()
    assert (tmp_path / "figures" / "node_by_domain_pair_heatmap.png").exists()


def test_scorecard_metrics_present(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg = deep_merge(cfg, {"transport": {"solver": "unbalanced"}, "runtime_values": {"n": 24, "mc_rollouts": 8}})
    artifact = run_pipeline(cfg, tmp_path)
    row = artifact.node_pair_table[0]
    assert "dangerous_unmatched_mass_b" in row
    assert "transport_forbidden_mass" in row
