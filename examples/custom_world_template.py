"""Template for adding a custom controlled operational world.

Copy this file into `otg/worlds/`, import it from `otg/worlds/registry.py`, and
register your world with `@register_world("your_world_name")`.
"""

from __future__ import annotations
import numpy as np

from otg.worlds.continuous_worlds import ContinuousOperationalWorld
from otg.worlds.registry import register_world
from otg.core.schema import WorldBatch


@register_world("my_custom_world")
class MyCustomWorld(ContinuousOperationalWorld):
    def spec(self) -> dict:
        return {
            "name": "my_custom_world",
            "family": "custom",
            "paper_core": False,
            "mathematical_definition": "Define the source and target node laws here.",
            "operational_coordinate_rule": "Explain which coordinates control terminal behavior.",
            "nuisance_rule": "Explain which coordinates should be invariant/collapsible.",
            "expected_behavior": "State what the OTG pipeline should detect.",
            "failure_modes": ["false collapse", "false separation"],
            "recommended_diagnostics": ["false_collapse_mass", "false_separation_score"],
        }

    def generate(self) -> WorldBatch:
        rng, n, d = self.rng, self.n(), self.dim()
        z_a = rng.normal(0.0, 0.3, size=(n, d))
        z_b = rng.normal(0.2, 0.3, size=(n, d))
        return self.make_batch("my_custom_world", z_a, z_b)
