from __future__ import annotations

from typing import Protocol


class MethodAdapter(Protocol):
    name: str

    def run(self, batch, cfg: dict, seed: int): ...


_METHOD_REGISTRY: dict[str, type[MethodAdapter]] = {}


def register_method(name: str):
    def deco(cls: type[MethodAdapter]):
        _METHOD_REGISTRY[name] = cls
        cls.name = name
        return cls

    return deco


def get_method(name: str) -> type[MethodAdapter]:
    if name not in _METHOD_REGISTRY:
        raise KeyError(f"Unknown benchmark method '{name}'. Available: {sorted(_METHOD_REGISTRY)}")
    return _METHOD_REGISTRY[name]


def available_methods() -> list[str]:
    return sorted(_METHOD_REGISTRY)


def method_provenance(name: str) -> dict[str, str]:
    cls = get_method(name)
    return {
        "paper": str(getattr(cls, "paper_name", name)),
        "implementation_kind": str(getattr(cls, "implementation_kind", "reimplemented")),
        "implementation_source": str(getattr(cls, "implementation_source", "paper-faithful-reimplementation")),
    }
