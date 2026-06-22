from __future__ import annotations
from otg.core.registry import Registry
from otg.invariance.base import InvarianceBuilder

INVARIANCE_REGISTRY: Registry[type[InvarianceBuilder]] = Registry("invariance builder")


def register_invariance(name: str):
    return INVARIANCE_REGISTRY.register(name)


def make_invariance_builder(name: str, cfg: dict, seed_bank) -> InvarianceBuilder:
    import otg.invariance.builders  # noqa: F401
    cls = INVARIANCE_REGISTRY.get(name)
    return cls(cfg, seed_bank)
