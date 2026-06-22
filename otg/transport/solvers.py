from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from otg.core.schema import TransportProblem, TransportResult


class TransportSolver(ABC):
    solver: str

    def __init__(self, cfg: dict, seed_bank):
        self.cfg = cfg
        self.seed_bank = seed_bank

    @abstractmethod
    def solve(self, problem: TransportProblem) -> TransportResult:
        raise NotImplementedError


def feasibility_report(allowed: np.ndarray | None, n: int, m: int) -> dict:
    if allowed is None:
        return {
            "has_mask": False,
            "allowed_fraction": 1.0,
            "row_feasible_fraction": 1.0,
            "col_feasible_fraction": 1.0,
            "num_infeasible_rows": 0,
            "num_infeasible_cols": 0,
            "balanced_feasible_necessary": True,
        }
    row_has = allowed.any(axis=1)
    col_has = allowed.any(axis=0)
    return {
        "has_mask": True,
        "allowed_fraction": float(allowed.mean()),
        "row_feasible_fraction": float(row_has.mean()),
        "col_feasible_fraction": float(col_has.mean()),
        "num_infeasible_rows": int((~row_has).sum()),
        "num_infeasible_cols": int((~col_has).sum()),
        "balanced_feasible_necessary": bool(row_has.all() and col_has.all()),
    }


def apply_masked_cost(cost: np.ndarray, allowed: np.ndarray | None, forbidden_cost: float = 1e6) -> np.ndarray:
    C = np.asarray(cost, dtype=float).copy()
    if allowed is not None:
        C[~allowed] = forbidden_cost
    return C


def marginal_errors(plan: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    row = plan.sum(axis=1)
    col = plan.sum(axis=0)
    return {
        "row_l1_error": float(np.sum(np.abs(row - a))),
        "col_l1_error": float(np.sum(np.abs(col - b))),
        "plan_mass": float(np.sum(plan)),
    }


def forbidden_mass(plan: np.ndarray, allowed: np.ndarray | None) -> float:
    if allowed is None:
        return 0.0
    return float(np.sum(plan[~allowed]))


def sinkhorn_plan(a: np.ndarray, b: np.ndarray, C: np.ndarray, epsilon: float, iterations: int) -> tuple[np.ndarray, dict]:
    K = np.exp(-C / max(epsilon, 1e-12))
    K = np.maximum(K, 1e-300)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(iterations):
        u = a / (K @ v + 1e-300)
        v = b / (K.T @ u + 1e-300)
    plan = (u[:, None] * K) * v[None, :]
    meta = {"epsilon": float(epsilon), "iterations": int(iterations)}
    meta.update(marginal_errors(plan, a, b))
    return plan, meta
