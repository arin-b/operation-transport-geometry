from otg.experiments import run_experiment
from otg.utils.config import load_config, deep_merge


def test_experiments_wrapper():
    cfg = load_config("configs/default.yaml")
    cfg = deep_merge(cfg, {"runtime_values": {"n": 24, "mc_rollouts": 8}})
    artifact = run_experiment(cfg, None)
    assert artifact.world_name == cfg["world"]["name"]
