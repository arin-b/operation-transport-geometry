from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable
import numpy as np


NodeFn = Callable[[dict[str, np.ndarray]], np.ndarray]


@dataclass
class DAGNode:
    name: str
    parents: tuple[str, ...]
    fn: NodeFn
    metadata: dict = field(default_factory=dict)


@dataclass
class OperationalDAG:
    """Configurable finite DAG used to model a deployed multi-stage system."""

    nodes: dict[str, DAGNode] = field(default_factory=dict)

    def add(self, name: str, parents: Iterable[str], fn: NodeFn, **metadata) -> "OperationalDAG":
        if name in self.nodes:
            raise ValueError(f"Node already exists: {name}")
        self.nodes[name] = DAGNode(name=name, parents=tuple(parents), fn=fn, metadata=dict(metadata))
        return self

    def validate(self, provided_inputs: Iterable[str] = ()) -> None:
        provided = set(provided_inputs)
        for name, node in self.nodes.items():
            for parent in node.parents:
                if parent not in self.nodes and parent not in provided:
                    raise ValueError(f"Node '{name}' depends on missing parent '{parent}'")
        self.topological_order(provided_inputs=provided)

    def topological_order(self, provided_inputs: Iterable[str] = ()) -> list[str]:
        provided = set(provided_inputs)
        order: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set(provided)

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                raise ValueError(f"Cycle detected at node '{name}'")
            visiting.add(name)
            node = self.nodes[name]
            for parent in node.parents:
                if parent in self.nodes:
                    visit(parent)
            visiting.remove(name)
            visited.add(name)
            order.append(name)

        for name in self.nodes:
            visit(name)
        return order

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self.validate(provided_inputs=inputs.keys())
        states = dict(inputs)
        for name in self.topological_order(provided_inputs=inputs.keys()):
            node = self.nodes[name]
            states[name] = node.fn({p: states[p] for p in node.parents})
        return states
