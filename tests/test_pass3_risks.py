from otg.utils.config import load_config, deep_merge
from otg.utils.seeding import SeedBank
from otg.worlds.registry import make_world
from otg.risks.registry import make_risk_model


def test_all_risk_modes_return_common_contract():
    cfg = load_config("configs/default.yaml")
    cfg = deep_merge(cfg, {"runtime_values": {"n": 30, "mc_rollouts": 8}, "world": {"name": "risk_degradation"}})
    world = make_world("risk_degradation", cfg, SeedBank(222))
    batch = world.generate()
    node = batch.nodes["repr"]

    modes = ["true", "noisy", "rollout", "learned_regression", "learned_classifier", "learned_mlp", "misspecified"]
    for mode in modes:
        local = deep_merge(cfg, {"risk": {"mode": mode, "mlp_max_iter": 80}})
        model = make_risk_model(mode, local, SeedBank(222))
        out = model.estimate(node)
        assert out.used_risk_a.shape == node.true_risk_a.shape
        assert out.used_risk_b.shape == node.true_risk_b.shape
        assert out.risk_error_mae >= 0.0
        assert out.calibration_error is not None
        assert out.failure_accuracy is not None


def test_learned_risk_pipeline_runs(tmp_path):
    from otg.core.pipeline import run_pipeline
    cfg = load_config("configs/default.yaml")
    cfg = deep_merge(cfg, {
        "world": {"name": "risk_degradation"},
        "risk": {"mode": "learned_regression"},
        "runtime_values": {"n": 40, "mc_rollouts": 8},
    })
    artifact = run_pipeline(cfg, tmp_path)
    diag = artifact.node_results["repr"].diagnostics
    assert "risk_calibration_error" in diag
    assert "risk_failure_accuracy" in diag
