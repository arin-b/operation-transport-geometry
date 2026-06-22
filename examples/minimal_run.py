from otg.core.pipeline import run_pipeline
from otg.utils.config import load_config, deep_merge

cfg = load_config("configs/default.yaml")
cfg = deep_merge(cfg, {
    "world": {"name": "harmful_boundary"},
    "risk": {"mode": "true"},
    "transport": {"solver": "masked_sinkhorn"},
    "runtime_values": {"n": 40, "mc_rollouts": 8},
})
artifact = run_pipeline(cfg, "runs/example_minimal")
print(artifact.system_score)
