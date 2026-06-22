from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Mapping
import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class DomainSpec:
    """A deployment domain represented by a name and sampling metadata."""
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeBatch:
    """Empirical node law plus all fields needed by the OTG pipeline.

    `z_a` and `z_b` are samples from two domain-induced node laws.
    `terminal_a` and `terminal_b` are downstream terminal outputs induced by those states.
    `true_risk_*` is retained as the audit ground truth.
    `used_risk_*` is what the algorithm is allowed to use.
    """

    node: str
    z_a: Array
    z_b: Array
    terminal_a: Array
    terminal_b: Array
    true_risk_a: Array
    true_risk_b: Array
    used_risk_a: Array
    used_risk_b: Array
    operational_coords_a: Array
    operational_coords_b: Array
    nuisance_coords_a: Array
    nuisance_coords_b: Array
    descriptor_coords_a: Array
    descriptor_coords_b: Array
    anchors_a: Array
    anchors_b: Array
    mass_a: Array
    mass_b: Array
    failure_a: Array
    failure_b: Array
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        n, m = len(self.z_a), len(self.z_b)
        checks = [
            (self.terminal_a, n, "terminal_a"),
            (self.terminal_b, m, "terminal_b"),
            (self.true_risk_a, n, "true_risk_a"),
            (self.true_risk_b, m, "true_risk_b"),
            (self.used_risk_a, n, "used_risk_a"),
            (self.used_risk_b, m, "used_risk_b"),
            (self.anchors_a, n, "anchors_a"),
            (self.anchors_b, m, "anchors_b"),
            (self.mass_a, n, "mass_a"),
            (self.mass_b, m, "mass_b"),
        ]
        for arr, expected, name in checks:
            if len(arr) != expected:
                raise ValueError(f"{name} has length {len(arr)} but expected {expected}")
        if not np.isclose(np.sum(self.mass_a), 1.0):
            raise ValueError("mass_a must sum to 1")
        if not np.isclose(np.sum(self.mass_b), 1.0):
            raise ValueError("mass_b must sum to 1")


@dataclass
class WorldBatch:
    """Generated controlled operational world."""
    name: str
    domains: tuple[DomainSpec, DomainSpec]
    nodes: dict[str, NodeBatch]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("WorldBatch must contain at least one node")
        for node in self.nodes.values():
            node.validate()


@dataclass
class RiskOutput:
    used_risk_a: Array
    used_risk_b: Array
    risk_error_mae: float
    mode: str
    metadata: dict[str, Any] = field(default_factory=dict)
    train_error_mae: float | None = None
    calibration_error: float | None = None
    failure_accuracy: float | None = None
    uncertainty_a: Array | None = None
    uncertainty_b: Array | None = None


@dataclass
class InvarianceOutput:
    allowed: Array
    penalty: Array
    hard_violations: Array
    soft_score: Array
    mode: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CostOutput:
    total: Array
    geometry: Array
    terminal: Array
    risk: Array
    invariance: Array
    mode: str
    weights: dict[str, float]


@dataclass
class TransportProblem:
    mass_a: Array
    mass_b: Array
    cost: Array
    allowed: Array | None = None
    penalty: Array | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransportResult:
    value: float
    plan: Array
    solver: str
    status: str
    unmatched_a: Array | None = None
    unmatched_b: Array | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeRunResult:
    node: str
    cost: CostOutput
    invariance: InvarianceOutput
    transport: TransportResult
    diagnostics: dict[str, float | str | int | bool]


@dataclass
class RunArtifact:
    config: dict[str, Any]
    world_name: str
    node_results: dict[str, NodeRunResult]
    system_score: dict[str, Any]
    assumption_report: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
