from otg.utils.presets import apply_runtime_preset


def test_preset_fills_runtime_values():
    cfg = {"runtime": {"preset": "default"}}
    out = apply_runtime_preset(cfg)
    assert out["runtime_values"]["n"] == 180
    assert out["runtime_values"]["mc_rollouts"] == 128


def test_preset_does_not_override_explicit_values():
    cfg = {"runtime": {"preset": "heavy"}, "runtime_values": {"n": 12}}
    out = apply_runtime_preset(cfg)
    assert out["runtime_values"]["n"] == 12
    assert out["runtime_values"]["mc_rollouts"] == 512
