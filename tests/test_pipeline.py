from otg.core.pipeline import run_pipeline
from otg.utils.config import load_config


def test_pipeline_runs(tmp_path):
    cfg = load_config("configs/default.yaml")
    artifact = run_pipeline(cfg, tmp_path)
    assert artifact.world_name == "synthetic_dag"
    assert artifact.comparisons
    assert artifact.domain_order == ["clear", "viewpoint_shift", "glare", "occlusion"]
    assert (tmp_path / "summary.json").exists()
