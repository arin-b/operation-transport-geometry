from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class DomainSpec:
    """Deployment domain d with input-law metadata P_d."""
    name: str
    id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    input_law_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def domain_id(self) -> str:
        return self.id or self.name


@dataclass
class NodeLaw:
    """Empirical node-wise law mu_v^d and aligned law mutilde_v^d for one node/domain."""
    node: str
    domain: str
    samples: Array
    aligned_samples: Array
    terminal_outputs: Array
    true_risk: Array
    used_risk: Array
    operational_coords: Array
    nuisance_coords: Array
    descriptor_coords: Array
    semantic_anchors: Array
    mass: Array
    failure: Array
    projection_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        n = len(self.samples)
        checks = [
            (self.aligned_samples, n, "aligned_samples"),
            (self.terminal_outputs, n, "terminal_outputs"),
            (self.true_risk, n, "true_risk"),
            (self.used_risk, n, "used_risk"),
            (self.operational_coords, n, "operational_coords"),
            (self.nuisance_coords, n, "nuisance_coords"),
            (self.descriptor_coords, n, "descriptor_coords"),
            (self.semantic_anchors, n, "semantic_anchors"),
            (self.mass, n, "mass"),
            (self.failure, n, "failure"),
        ]
        for arr, expected, name in checks:
            if len(arr) != expected:
                raise ValueError(f"{self.node}/{self.domain}: {name} has length {len(arr)} but expected {expected}")
        if not np.isclose(np.sum(self.mass), 1.0):
            raise ValueError(f"{self.node}/{self.domain}: mass must sum to 1")


@dataclass
class GraphWorldBatch:
    """Proposal-aligned graph-level world: DAG, domains, selected nodes, and node laws."""
    name: str
    graph: dict[str, Any]
    domains: dict[str, DomainSpec]
    selected_nodes: list[str]
    node_laws: dict[tuple[str, str], NodeLaw]
    projection_metadata: dict[str, Any] = field(default_factory=dict)
    terminal_evaluation_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if len(self.domains) < 2:
            raise ValueError("GraphWorldBatch must contain at least two deployment domains")
        if len(self.selected_nodes) < 1:
            raise ValueError("GraphWorldBatch must contain selected internal nodes")
        for node in self.selected_nodes:
            for domain in self.domains:
                key = (node, domain)
                if key not in self.node_laws:
                    raise ValueError(f"Missing NodeLaw for node={node!r}, domain={domain!r}")
                self.node_laws[key].validate()

    def law(self, node: str, domain: str) -> NodeLaw:
        return self.node_laws[(node, domain)]

    @property
    def domain_ids(self) -> list[str]:
        return list(self.domains.keys())


@dataclass
class PairwiseNodeProblem:
    """Internal pairwise compatibility object derived from GraphWorldBatch.

    This preserves the old low-level solver/cost/risk API. It is not the primary
    mathematical object; it represents one (v, d, d') comparison.
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
    domain_a: str = "domain_a"
    domain_b: str = "domain_b"
    raw_z_a: Array | None = None
    raw_z_b: Array | None = None
    projection_a: dict[str, Any] = field(default_factory=dict)
    projection_b: dict[str, Any] = field(default_factory=dict)
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
                raise ValueError(f"{self.node}/{self.domain_a}->{self.domain_b}: {name} has length {len(arr)} but expected {expected}")
        if not np.isclose(np.sum(self.mass_a), 1.0):
            raise ValueError("mass_a must sum to 1")
        if not np.isclose(np.sum(self.mass_b), 1.0):
            raise ValueError("mass_b must sum to 1")


# Old low-level modules import NodeBatch; keep that as the internal pairwise object.
NodeBatch = PairwiseNodeProblem


@dataclass
class WorldBatch:
    """Legacy two-domain pairwise world, retained for internal/backward compatibility only."""
    name: str
    domains: tuple[DomainSpec, DomainSpec]
    nodes: dict[str, PairwiseNodeProblem]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("WorldBatch must contain at least one pairwise node problem")
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
    """Legacy/internal pairwise node result."""
    node: str
    cost: CostOutput
    invariance: InvarianceOutput
    transport: TransportResult
    diagnostics: dict[str, float | str | int | bool]


@dataclass
class NodePairResult:
    node: str
    domain_pair: tuple[str, str]
    cost: CostOutput
    invariance: InvarianceOutput
    transport: TransportResult
    diagnostics: dict[str, Any]
    pairwise_problem: PairwiseNodeProblem | None = None


@dataclass
class SystemComparisonResult:
    domain_pair: tuple[str, str]
    node_results: dict[str, NodePairResult]
    aggregate: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRunArtifact:
    config: dict[str, Any]
    world_name: str
    graph: dict[str, Any]
    domains: dict[str, DomainSpec]
    selected_nodes: list[str]
    comparisons: dict[tuple[str, str], SystemComparisonResult]
    discrepancy_matrix: Array
    domain_order: list[str]
    node_pair_table: list[dict[str, Any]]
    system_score: dict[str, Any]
    assumption_report: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    # Compatibility field: first domain-pair node results, not the primary object.
    node_results: dict[str, NodeRunResult] = field(default_factory=dict)


# Old artifact name now points to the graph-level artifact. Pairwise code should use
# NodeRunResult directly or run_pairwise_pipeline.
RunArtifact = GraphRunArtifact
