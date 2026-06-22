from __future__ import annotations
from otg.core.registry import Registry
from otg.risks.base import RiskModel

RISK_REGISTRY: Registry[type[RiskModel]] = Registry("risk model")


def register_risk(name: str):
    return RISK_REGISTRY.register(name)


def make_risk_model(name: str, cfg: dict, seed_bank) -> RiskModel:
    import otg.risks.models  # noqa: F401
    if name is True:
        name = "true"
    if name is False:
        name = "false"
    name = str(name)
    cls = RISK_REGISTRY.get(name)
    return cls(cfg, seed_bank)
