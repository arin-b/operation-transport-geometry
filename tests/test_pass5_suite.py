from otg.core.pipeline import run_pipeline
from otg.reporting.suite import write_suite_outputs
from otg.utils.config import load_config, deep_merge
from otg.utils.io import ensure_dir


def test_suite_outputs_direct(tmp_path):
    base = load_config("configs/default.yaml")
    out = ensure_dir(tmp_path / "suite")
    results = []
    for i, solver in enumerate(["masked_sinkhorn", "unbalanced"]):
        cfg = deep_merge(base, {"runtime_values": {"n": 24, "mc_rollouts": 8}, "transport": {"solver": solver}})
        run_dir = out / f"{i:02d}_{solver}"
        artifact = run_pipeline(cfg, run_dir)
        results.append({"world": artifact.world_name, "system_score": artifact.system_score, "out": str(run_dir), "summary": {"world": artifact.world_name, "system_score": artifact.system_score}})
    write_suite_outputs(out, results)
    assert (out / "suite_report.md").exists()
    assert (out / "suite_metrics.csv").exists()
