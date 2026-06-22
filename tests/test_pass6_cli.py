from otg.worlds.registry import available_worlds


def test_world_registry_contains_synthetic_dag():
    assert "synthetic_dag" in available_worlds()


def test_validate_runner_smoke(tmp_path, monkeypatch):
    import otg.validation.runner as runner
    real_run_pipeline = runner.run_pipeline

    def run_without_case_artifacts(cfg, out_dir=None):
        return real_run_pipeline(cfg, None)

    monkeypatch.setattr(runner, "run_pipeline", run_without_case_artifacts)
    monkeypatch.setattr(runner, "VALIDATION_CASES", [{"name": "graph_core", "overrides": {"world": {"name": "synthetic_dag"}, "risk": {"mode": "true"}, "transport": {"solver": "masked_sinkhorn"}}}])
    summary = runner.run_validation(tmp_path / "validation", preset="fast", seed=9)
    assert summary["num_checks"] > 0
    assert (tmp_path / "validation" / "validation_report.md").exists()
