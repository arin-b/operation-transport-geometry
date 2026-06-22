from pathlib import Path

from otg.core.pipeline import run_pipeline
from otg.utils.config import load_config, deep_merge


def test_pass5_outputs_exist(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg = deep_merge(cfg, {
        "world": {"name": "harmless_nuisance"},
        "transport": {"solver": "masked_sinkhorn"},
        "runtime_values": {"n": 35, "mc_rollouts": 8},
    })
    artifact = run_pipeline(cfg, tmp_path)
    expected = [
        "summary.json",
        "assumptions.json",
        "metrics.csv",
        "scorecard.csv",
        "report.md",
        "interpretation.md",
        "table_snippet.tex",
        "assumption_table_snippet.tex",
        "repr_diagnostics.json",
        "repr_transport_metadata.json",
        "world_spec.json",
    ]
    for name in expected:
        assert (tmp_path / name).exists(), name
    assert (tmp_path / "figures" / "repr_transport_plan.png").exists()
    assert (tmp_path / "arrays" / "repr" / "cost_total.npy").exists()


def test_scorecard_metrics_present(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg = deep_merge(cfg, {
        "world": {"name": "unbalanced_dangerous_mass"},
        "transport": {"solver": "unbalanced"},
        "runtime_values": {"n": 35, "mc_rollouts": 8},
    })
    artifact = run_pipeline(cfg, tmp_path)
    diag = artifact.node_results["repr"].diagnostics
    assert "harmless_shift_collapse" in diag
    assert "harmful_shift_detection" in diag
    assert "transport_row_mass_gini" in diag
    assert "dangerous_unmatched_mass_b" in diag
