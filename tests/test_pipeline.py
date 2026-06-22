from otg.utils.config import load_config
from otg.core.pipeline import run_pipeline


def test_pipeline_runs(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg["runtime"]["preset"] = "fast"
    artifact = run_pipeline(cfg, tmp_path)
    assert artifact.world_name == "harmful_boundary"
    assert "repr" in artifact.node_results
    assert (tmp_path / "summary.json").exists()
