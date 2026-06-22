from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MethodRun:
    method: str
    world: str
    seed: int
    plan: object | None = None
    rep_a: object | None = None
    rep_b: object | None = None
    risk_a: object | None = None
    risk_b: object | None = None
    allowed: object | None = None
    metrics: dict[str, float | int | bool | str] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class BenchmarkSummary:
    method: str
    world: str
    seeds: list[int]
    runs: list[dict]
    aggregate: dict[str, object]
    out_dir: Path
