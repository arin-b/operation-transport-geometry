from __future__ import annotations
from otg.core.registry import Registry
from otg.transport.solvers import TransportSolver

TRANSPORT_REGISTRY: Registry[type[TransportSolver]] = Registry("transport solver")


def register_solver(name: str):
    return TRANSPORT_REGISTRY.register(name)


def make_solver(name: str, cfg: dict, seed_bank) -> TransportSolver:
    import otg.transport.backends  # noqa: F401
    cls = TRANSPORT_REGISTRY.get(name)
    return cls(cfg, seed_bank)
