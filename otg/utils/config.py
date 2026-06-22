from __future__ import annotations
from pathlib import Path
import copy
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, overrides: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def merge_overrides(base: dict, overrides: dict) -> dict:
    return deep_merge(base, overrides)
