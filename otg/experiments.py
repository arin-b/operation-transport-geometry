from __future__ import annotations

from pathlib import Path

from otg.core.pipeline import run_pipeline
from otg.utils.config import load_config, deep_merge
from otg.utils.io import ensure_dir
from otg.reporting.suite import write_suite_outputs


def run_experiment(cfg: dict, out_dir: str | Path):
    """Backward-compatible wrapper around `otg.core.pipeline.run_pipeline`."""
    return run_pipeline(cfg, out_dir)


def run_suite(suite_cfg: dict, out_dir: str | Path, preset: str = "fast", seed: int = 0):
    base = load_config(suite_cfg.get("base_config", "configs/default.yaml"))
    out = ensure_dir(out_dir)
    results = []
    for i, item in enumerate(suite_cfg.get("worlds", [])):
        name = item["name"] if isinstance(item, dict) else str(item)
        overrides = item.get("overrides", {}) if isinstance(item, dict) else {}
        cfg = deep_merge(base, overrides)
        cfg = deep_merge(cfg, {"world": {"name": name}, "runtime": {"preset": preset}, "seed": {"master": seed + 1000 * i}})
        artifact = run_pipeline(cfg, out / f"{i:02d}_{name}")
        results.append({"world": name, "system_score": artifact.system_score, "out": str(out / f"{i:02d}_{name}")})
    write_suite_outputs(out, results)
    return results
