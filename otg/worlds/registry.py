from __future__ import annotations

from otg.worlds.base import ControlledOperationalWorld

_WORLD_REGISTRY: dict[str, type[ControlledOperationalWorld]] = {}


def register_world(name: str):
    def deco(cls: type[ControlledOperationalWorld]):
        _WORLD_REGISTRY[name] = cls
        cls.world_name = name
        return cls
    return deco


def make_world(name: str, cfg: dict, seed_bank) -> ControlledOperationalWorld:
    import otg.worlds.continuous_worlds  # noqa: F401
    if name not in _WORLD_REGISTRY:
        raise KeyError(f"Unknown world '{name}'. Available: {sorted(_WORLD_REGISTRY)}")
    return _WORLD_REGISTRY[name](cfg=cfg, seed_bank=seed_bank)


def available_worlds() -> list[str]:
    import otg.worlds.continuous_worlds  # noqa: F401
    return sorted(_WORLD_REGISTRY)
