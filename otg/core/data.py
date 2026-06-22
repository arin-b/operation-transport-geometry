from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class NodeData:
    name: str
    z_a: np.ndarray
    z_b: np.ndarray
    y_a: np.ndarray
    y_b: np.ndarray
    true_risk_a: np.ndarray
    true_risk_b: np.ndarray
    used_risk_a: np.ndarray
    used_risk_b: np.ndarray
    descriptors_a: np.ndarray
    descriptors_b: np.ndarray
    anchors_a: np.ndarray
    anchors_b: np.ndarray
    mass_a: np.ndarray
    mass_b: np.ndarray
    harmful_a: np.ndarray
    harmful_b: np.ndarray
    op_coords_a: np.ndarray
    op_coords_b: np.ndarray
    nuisance_a: np.ndarray
    nuisance_b: np.ndarray
    hidden_a: np.ndarray | None = None
    hidden_b: np.ndarray | None = None
    expected_equivalent: np.ndarray | None = None
    expected_harmful_mismatch: np.ndarray | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldSpec:
    name: str
    family: str
    paper_core: bool
    mathematical_definition: str
    operational_coordinate_rule: str
    nuisance_rule: str
    expected_behavior: str
    failure_modes: list[str]
    recommended_diagnostics: list[str]


@dataclass
class WorldData:
    name: str
    nodes: Dict[str, NodeData]
    metadata: Dict[str, Any]
    spec: WorldSpec | None = None
