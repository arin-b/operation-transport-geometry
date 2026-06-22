from otg.core.pipeline import run_pipeline
from otg.utils.config import load_config, deep_merge


def test_graph_pipeline_runs(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg = deep_merge(cfg, {"runtime_values": {"n": 24, "mc_rollouts": 8}})
    artifact = run_pipeline(cfg, tmp_path)
    assert artifact.world_name == "synthetic_dag"
    assert artifact.metadata["pipeline"] == "graph_level_otg"
    assert len(artifact.domain_order) >= 4
    assert set(artifact.selected_nodes) >= {"detector", "representation", "measurement"}
    assert artifact.discrepancy_matrix.shape == (len(artifact.domain_order), len(artifact.domain_order))
    assert artifact.comparisons
    assert (tmp_path / "D_op_matrix.csv").exists()
    assert (tmp_path / "node_pair_metrics.csv").exists()
    assert (tmp_path / "figures" / "D_op_domain_pair_heatmap.png").exists()


def test_graph_harmful_exceeds_harmless():
    cfg = load_config("configs/default.yaml")
    cfg = deep_merge(cfg, {"runtime_values": {"n": 36, "mc_rollouts": 8}, "seed": {"master": 42}})
    artifact = run_pipeline(cfg)
    vals = {pair: float(comp.aggregate["value"]) for pair, comp in artifact.comparisons.items()}
    assert vals[("clear", "glare")] > vals[("clear", "viewpoint_shift")]
    assert vals[("clear", "occlusion")] > vals[("clear", "viewpoint_shift")]


def test_graph_unbalanced_reports_dangerous_mass():
    cfg = load_config("configs/default.yaml")
    cfg = deep_merge(cfg, {"transport": {"solver": "unbalanced"}, "runtime_values": {"n": 36, "mc_rollouts": 8}})
    artifact = run_pipeline(cfg)
    vals = [float(res.diagnostics.get("dangerous_unmatched_mass_b", 0.0)) for comp in artifact.comparisons.values() for res in comp.node_results.values()]
    assert max(vals) > 0.0
