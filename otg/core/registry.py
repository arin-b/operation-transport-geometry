from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Small explicit registry used for worlds, risk models, invariance builders, and solvers."""

    def __init__(self, label: str):
        self.label = label
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            self._items[name] = obj
            return obj
        return deco

    def get(self, name: str) -> T:
        if name not in self._items:
            raise KeyError(f"Unknown {self.label} '{name}'. Available: {sorted(self._items)}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)
