from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from otg.core.schema import TransportProblem, TransportResult
from otg.transport.solvers import (
    TransportSolver,
    apply_masked_cost,
    feasibility_report,
    forbidden_mass,
    marginal_errors,
    sinkhorn_plan,
)
from otg.transport.registry import register_solver


@register_solver("lp")
class LPSolver(TransportSolver):
    solver = "lp"

    def solve(self, problem: TransportProblem) -> TransportResult:
        a, b = problem.mass_a, problem.mass_b
        C = np.asarray(problem.cost, dtype=float)
        n, m = C.shape
        feas = feasibility_report(problem.allowed, n, m)

        c = C.reshape(-1)
        A_eq = []
        rhs = []
        for i in range(n):
            row = np.zeros(n * m)
            row[i * m:(i + 1) * m] = 1.0
            A_eq.append(row)
            rhs.append(a[i])
        for j in range(m):
            col = np.zeros(n * m)
            col[j::m] = 1.0
            A_eq.append(col)
            rhs.append(b[j])

        bounds = []
        for i in range(n):
            for j in range(m):
                if problem.allowed is not None and not bool(problem.allowed[i, j]):
                    bounds.append((0.0, 0.0))
                else:
                    bounds.append((0.0, None))

        res = linprog(c, A_eq=np.vstack(A_eq), b_eq=np.asarray(rhs), bounds=bounds, method="highs")
        if not res.success:
            metadata = {**feas, "success": False, "message": res.message}
            return TransportResult(float("inf"), np.zeros_like(C), self.solver, f"failed:{res.message}", metadata=metadata)

        plan = res.x.reshape(n, m)
        metadata = {**feas, "success": True, **marginal_errors(plan, a, b), "forbidden_mass": forbidden_mass(plan, problem.allowed)}
        return TransportResult(float(np.sum(plan * C)), plan, self.solver, "ok", metadata=metadata)


@register_solver("sinkhorn")
class SinkhornSolver(TransportSolver):
    solver = "sinkhorn"

    def solve(self, problem: TransportProblem) -> TransportResult:
        # Plain Sinkhorn intentionally ignores the admissibility mask. Use masked_sinkhorn for admissible Sinkhorn.
        C = np.asarray(problem.cost, dtype=float)
        eps = float(self.cfg.get("transport", {}).get("epsilon", 0.05))
        iterations = int(self.cfg.get("transport", {}).get("sinkhorn_iter", 700))
        plan, meta = sinkhorn_plan(problem.mass_a, problem.mass_b, C, eps, iterations)
        meta.update({
            **feasibility_report(problem.allowed, *C.shape),
            "mask_respected": False,
            "forbidden_mass": forbidden_mass(plan, problem.allowed),
        })
        return TransportResult(float(np.sum(plan * C)), plan, self.solver, "ok", metadata=meta)


@register_solver("masked_sinkhorn")
class MaskedSinkhornSolver(TransportSolver):
    solver = "masked_sinkhorn"

    def solve(self, problem: TransportProblem) -> TransportResult:
        # Entropic OT on the admissible support. Forbidden entries are zeroed in
        # the kernel itself, not merely assigned a large finite cost. This avoids
        # numerical leakage through inadmissible pairs. If the mask is not fully
        # feasible, the returned plan may have marginal error; that error is
        # reported explicitly rather than hidden as forbidden transport.
        C = np.asarray(problem.cost, dtype=float)
        eps = float(self.cfg.get("transport", {}).get("epsilon", 0.05))
        iterations = int(self.cfg.get("transport", {}).get("sinkhorn_iter", 700))
        K = np.exp(-C / max(eps, 1e-12))
        if problem.allowed is not None:
            K = K.copy()
            K[~problem.allowed] = 0.0
        K = np.maximum(K, 0.0)
        u = np.ones_like(problem.mass_a)
        v = np.ones_like(problem.mass_b)
        for _ in range(iterations):
            u = problem.mass_a / (K @ v + 1e-300)
            v = problem.mass_b / (K.T @ u + 1e-300)
        plan = (u[:, None] * K) * v[None, :]
        if problem.allowed is not None:
            plan[~problem.allowed] = 0.0
        meta = {"epsilon": eps, "iterations": iterations}
        meta.update(marginal_errors(plan, problem.mass_a, problem.mass_b))
        meta.update({
            **feasibility_report(problem.allowed, *C.shape),
            "mask_respected": True,
            "forbidden_mass": forbidden_mass(plan, problem.allowed),
        })
        return TransportResult(float(np.sum(plan * C)), plan, self.solver, "ok", metadata=meta)


@register_solver("unbalanced")
class UnbalancedSolver(TransportSolver):
    solver = "unbalanced"

    def solve(self, problem: TransportProblem) -> TransportResult:
        forbidden_cost = float(self.cfg.get("transport", {}).get("forbidden_cost", 1e6))
        C = apply_masked_cost(problem.cost, problem.allowed, forbidden_cost=forbidden_cost)
        feas = feasibility_report(problem.allowed, *C.shape)

        try:
            import ot
            reg = float(self.cfg.get("transport", {}).get("epsilon", 0.05))
            reg_m = float(self.cfg.get("transport", {}).get("unbalanced_reg_m", 1.0))
            plan = ot.unbalanced.sinkhorn_unbalanced(problem.mass_a, problem.mass_b, C, reg=reg, reg_m=reg_m)
            status = "ok"
            solver = "unbalanced_pot"
            metadata = {"backend": "POT", "epsilon": reg, "unbalanced_reg_m": reg_m}
        except Exception as exc:
            eps = float(self.cfg.get("transport", {}).get("epsilon", 0.05))
            iterations = int(self.cfg.get("transport", {}).get("sinkhorn_iter", 700))
            plan, sink_meta = sinkhorn_plan(problem.mass_a, problem.mass_b, C, eps, iterations)
            cutoff = float(self.cfg.get("transport", {}).get("unbalanced_cost_quantile", 0.85))
            finite = C[np.isfinite(C)]
            threshold = float(np.quantile(finite, cutoff)) if finite.size else float(np.max(C))
            plan = plan.copy()
            plan[C > threshold] = 0.0
            status = "ok_fallback"
            solver = "unbalanced_fallback"
            metadata = {"backend": "fallback_trimmed_sinkhorn", "fallback_error": type(exc).__name__, "trim_threshold": threshold, **sink_meta}

        if problem.allowed is not None:
            # Even in unbalanced mode, inadmissible pairs should not carry plan mass.
            # Their mass is reclassified as unmatched rather than counted as a legal match.
            plan = plan.copy()
            plan[~problem.allowed] = 0.0

        unmatched_a = np.maximum(problem.mass_a - plan.sum(axis=1), 0.0)
        unmatched_b = np.maximum(problem.mass_b - plan.sum(axis=0), 0.0)
        metadata.update({
            **feas,
            **marginal_errors(plan, problem.mass_a, problem.mass_b),
            "forbidden_mass": forbidden_mass(plan, problem.allowed),
            "unmatched_mass_a": float(np.sum(unmatched_a)),
            "unmatched_mass_b": float(np.sum(unmatched_b)),
        })
        return TransportResult(float(np.sum(plan * C)), plan, solver, status, unmatched_a=unmatched_a, unmatched_b=unmatched_b, metadata=metadata)
