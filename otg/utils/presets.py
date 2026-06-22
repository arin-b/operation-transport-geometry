from __future__ import annotations

from copy import deepcopy

PRESETS: dict[str, dict[str, int]] = {
    "fast": {"n": 60, "mc_rollouts": 32},
    "default": {"n": 180, "mc_rollouts": 128},
    "heavy": {"n": 600, "mc_rollouts": 512},
}


def apply_runtime_preset(cfg: dict, *, force: bool = False) -> dict:
    """Return a config with runtime_values filled from the selected preset.

    Explicit runtime_values are preserved by default. This keeps command-line
    presets useful while still allowing tests and suite configs to set exact
    sample sizes.
    """
    out = deepcopy(cfg)
    preset = out.get("runtime", {}).get("preset", "fast")
    if preset not in PRESETS:
        raise ValueError(f"Unknown runtime preset {preset!r}. Available: {sorted(PRESETS)}")
    rv = out.setdefault("runtime_values", {})
    for key, value in PRESETS[preset].items():
        if force or key not in rv:
            rv[key] = value
    return out
