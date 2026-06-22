from otg.validation.doctor import collect_doctor_report, format_doctor_report
from otg.worlds.registry import available_worlds


def test_doctor_and_list_worlds_direct():
    report = collect_doctor_report()
    text = format_doctor_report(report)
    assert "Worlds:" in text
    assert "harmful_boundary" in available_worlds()


def test_validate_runner_smoke(tmp_path, monkeypatch):
    import otg.validation.runner as runner
    real_run_pipeline = runner.run_pipeline

    def run_without_case_artifacts(cfg, out_dir=None):
        return real_run_pipeline(cfg, None)

    monkeypatch.setattr(runner, "run_pipeline", run_without_case_artifacts)
    monkeypatch.setattr(runner, "VALIDATION_CASES", [
        {
            "name": "harmless_nuisance_core",
            "overrides": {"world": {"name": "harmless_nuisance"}, "risk": {"mode": "true"}, "transport": {"solver": "masked_sinkhorn"}},
        }
    ])
    summary = runner.run_validation(tmp_path / "validation", preset="fast", seed=9)
    assert summary["status"] == "pass"
    assert (tmp_path / "validation" / "validation_report.md").exists()
    assert (tmp_path / "validation" / "validation_results.json").exists()
