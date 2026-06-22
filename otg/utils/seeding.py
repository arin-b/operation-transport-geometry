from __future__ import annotations
import hashlib
import numpy as np


class SeedBank:
    """Master-seed object that creates deterministic component seeds."""

    def __init__(self, master_seed: int):
        self.master_seed = int(master_seed)
        self._cache: dict[str, int] = {}

    def seed(self, name: str) -> int:
        if name not in self._cache:
            payload = f"{self.master_seed}:{name}".encode("utf-8")
            self._cache[name] = int(hashlib.sha256(payload).hexdigest()[:8], 16)
        return self._cache[name]

    def rng(self, name: str) -> np.random.Generator:
        return np.random.default_rng(self.seed(name))

    def as_dict(self) -> dict:
        return {"master_seed": self.master_seed, "component_seeds": dict(self._cache)}
