import numpy as np
from otg.transport.backends import LPSolver
from otg.core.schema import TransportProblem
from otg.utils.seeding import SeedBank


def test_lp_solver_small():
    a = np.ones(3) / 3
    b = np.ones(3) / 3
    C = np.eye(3)
    tr = LPSolver({"transport": {}}, SeedBank(0)).solve(TransportProblem(a, b, C))
    assert tr.status == "ok"
    assert tr.plan.shape == (3, 3)
