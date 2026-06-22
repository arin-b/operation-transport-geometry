from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np

from otg.core.data import WorldData, WorldSpec


@dataclass
class ControlledOperationalWorld(ABC):
    cfg: dict
    seed_bank: object
    world_name: str = "base"

    @abstractmethod
    def generate(self) -> WorldData:
        raise NotImplementedError

    @abstractmethod
    def spec(self) -> WorldSpec:
        raise NotImplementedError

    @property
    def rng(self) -> np.random.Generator:
        return self.seed_bank.rng(f"world:{self.world_name}")

    def n(self) -> int:
        return int(self.cfg.get("runtime_values", {}).get("n", self.cfg.get("world", {}).get("n", 100)))

    def dim(self) -> int:
        return int(self.cfg.get("world", {}).get("dim", 2))
