import numpy as np

from otg.core.schema import TransportProblem
from otg.transport.registry import make_solver
from otg.utils.seeding import SeedBank


def test_masked_sinkhorn_reduces_forbidden_mass():
    a = np.ones(4) / 4
    b = np.ones(4) / 4
    C = np.ones((4, 4))
    np.fill_diagonal(C, 0.0)
    allowed = np.eye(4, dtype=bool)
    cfg = {"transport": {"epsilon": 0.05, "sinkhorn_iter": 100, "forbidden_cost": 1e6}}
    solver = make_solver("masked_sinkhorn", cfg, SeedBank(1))
    tr = solver.solve(TransportProblem(a, b, C, allowed=allowed))
    assert tr.status == "ok"
    assert tr.metadata["forbidden_mass"] < 1e-6


def test_plain_sinkhorn_reports_forbidden_mass():
    a = np.ones(3) / 3
    b = np.ones(3) / 3
    C = np.zeros((3, 3))
    allowed = np.eye(3, dtype=bool)
    cfg = {"transport": {"epsilon": 0.1, "sinkhorn_iter": 50}}
    solver = make_solver("sinkhorn", cfg, SeedBank(2))
    tr = solver.solve(TransportProblem(a, b, C, allowed=allowed))
    assert tr.metadata["mask_respected"] is False
    assert tr.metadata["forbidden_mass"] > 0.0


def test_unbalanced_reports_unmatched_mass():
    a = np.ones(5) / 5
    b = np.ones(5) / 5
    C = np.ones((5, 5))
    allowed = np.eye(5, dtype=bool)
    cfg = {"transport": {"epsilon": 0.05, "sinkhorn_iter": 50, "unbalanced_cost_quantile": 0.50, "forbidden_cost": 1e6}}
    solver = make_solver("unbalanced", cfg, SeedBank(3))
    tr = solver.solve(TransportProblem(a, b, C, allowed=allowed))
    assert tr.unmatched_a is not None
    assert tr.unmatched_b is not None
    assert tr.metadata["unmatched_mass_a"] >= 0.0
