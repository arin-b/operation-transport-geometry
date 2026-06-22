from __future__ import annotations

import numpy as np

from otg.core.schema import DomainSpec, NodeBatch, WorldBatch
from otg.worlds.base import ControlledOperationalWorld
from otg.worlds.registry import register_world


def _masses(n: int) -> np.ndarray:
    return np.ones(n, dtype=float) / n


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _split_state(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    op = z[:, :1]
    nuisance = z[:, 1:] if z.shape[1] > 1 else np.zeros((len(z), 1))
    return op, nuisance


def _terminal(z: np.ndarray, cfg: dict, rng: np.random.Generator, hidden: np.ndarray | None = None) -> np.ndarray:
    op, nuisance = _split_state(z)
    terminal_noise = float(cfg.get("uncertainty", {}).get("terminal_noise", 0.03))
    nuisance_leak = float(cfg.get("world", {}).get("nuisance_leak", 0.05))
    hidden_leak = float(cfg.get("world", {}).get("hidden_leak", 0.15))
    y = op[:, 0].copy()
    if nuisance.shape[1] > 0:
        y = y + nuisance_leak * np.tanh(nuisance[:, 0])
    if hidden is not None:
        y = y + hidden_leak * hidden[:, 0]
    if terminal_noise > 0:
        y = y + rng.normal(0, terminal_noise, size=len(z))
    return y


def _risk(y: np.ndarray, cfg: dict) -> np.ndarray:
    threshold = float(cfg.get("world", {}).get("failure_threshold", 0.75))
    sharpness = float(cfg.get("world", {}).get("risk_sharpness", 10.0))
    return _sigmoid(sharpness * (y - threshold))


def _descriptor(z: np.ndarray, mode: str) -> np.ndarray:
    op, nuisance = _split_state(z)
    norm_n = np.linalg.norm(nuisance, axis=1)
    if mode == "correct":
        return np.c_[op[:, 0], norm_n]
    if mode == "nuisance_only":
        return np.c_[norm_n, np.zeros(len(z))]
    if mode == "op_only":
        return np.c_[op[:, 0], np.zeros(len(z))]
    if mode == "scrambled":
        return np.c_[np.sin(3 * op[:, 0]), np.cos(norm_n)]
    return np.c_[op[:, 0], norm_n]


def _anchors(z: np.ndarray, mode: str) -> np.ndarray:
    op, nuisance = _split_state(z)
    if mode == "none":
        return np.zeros(len(z), dtype=int)
    if mode == "nuisance_only":
        norm_n = np.linalg.norm(nuisance, axis=1)
        return (norm_n > np.median(norm_n)).astype(int)
    return np.floor((op[:, 0] + 2.0) / 0.75).astype(int)


def _spec(
    name: str,
    family: str,
    paper_core: bool,
    definition: str,
    op_rule: str,
    nuisance_rule: str,
    expected: str,
    failure_modes: list[str],
    diagnostics: list[str],
) -> dict:
    return {
        "name": name,
        "family": family,
        "paper_core": paper_core,
        "mathematical_definition": definition,
        "operational_coordinate_rule": op_rule,
        "nuisance_rule": nuisance_rule,
        "expected_behavior": expected,
        "failure_modes": failure_modes,
        "recommended_diagnostics": diagnostics,
    }


class ContinuousOperationalWorld(ControlledOperationalWorld):
    descriptor_mode_a = "correct"
    descriptor_mode_b = "correct"
    anchor_mode_a = "correct"
    anchor_mode_b = "correct"

    def n(self) -> int:
        return int(self.cfg.get("runtime_values", {}).get("n", self.cfg.get("world", {}).get("n", 100)))

    def dim(self) -> int:
        return int(self.cfg.get("world", {}).get("dim", 2))

    @property
    def rng(self) -> np.random.Generator:
        return self.seed_bank.rng(f"world:{self.world_name}")

    def spec(self) -> dict:
        raise NotImplementedError

    def make_batch(self, name: str, z_a: np.ndarray, z_b: np.ndarray, metadata: dict | None = None) -> WorldBatch:
        rng = self.rng
        hidden_a = rng.normal(0, 1, size=(len(z_a), 1)) if self.cfg.get("uncertainty", {}).get("hidden_nuisance", True) else None
        hidden_b = rng.normal(0, 1, size=(len(z_b), 1)) if self.cfg.get("uncertainty", {}).get("hidden_nuisance", True) else None

        terminal_a = _terminal(z_a, self.cfg, rng, hidden_a)
        terminal_b = _terminal(z_b, self.cfg, rng, hidden_b)
        true_risk_a = _risk(terminal_a, self.cfg)
        true_risk_b = _risk(terminal_b, self.cfg)

        op_a, nuisance_a = _split_state(z_a)
        op_b, nuisance_b = _split_state(z_b)

        node = NodeBatch(
            node="repr",
            z_a=z_a,
            z_b=z_b,
            terminal_a=terminal_a,
            terminal_b=terminal_b,
            true_risk_a=true_risk_a,
            true_risk_b=true_risk_b,
            used_risk_a=true_risk_a.copy(),
            used_risk_b=true_risk_b.copy(),
            operational_coords_a=op_a,
            operational_coords_b=op_b,
            nuisance_coords_a=nuisance_a,
            nuisance_coords_b=nuisance_b,
            descriptor_coords_a=_descriptor(z_a, self.descriptor_mode_a),
            descriptor_coords_b=_descriptor(z_b, self.descriptor_mode_b),
            anchors_a=_anchors(z_a, self.anchor_mode_a),
            anchors_b=_anchors(z_b, self.anchor_mode_b),
            mass_a=_masses(len(z_a)),
            mass_b=_masses(len(z_b)),
            failure_a=true_risk_a > 0.5,
            failure_b=true_risk_b > 0.5,
            metadata={
                "descriptor_mode_a": self.descriptor_mode_a,
                "descriptor_mode_b": self.descriptor_mode_b,
                "anchor_mode_a": self.anchor_mode_a,
                "anchor_mode_b": self.anchor_mode_b,
                "hidden_a_present": hidden_a is not None,
                "hidden_b_present": hidden_b is not None,
                **(metadata or {}),
            },
        )

        spec = self.spec()
        return WorldBatch(
            name=name,
            domains=(DomainSpec("domain_a"), DomainSpec("domain_b")),
            nodes={"repr": node},
            metadata={"world_spec": spec, **(metadata or {})},
        )


@register_world("harmless_nuisance")
class HarmlessNuisanceWorld(ContinuousOperationalWorld):
    def spec(self) -> dict:
        return _spec(
            "harmless_nuisance",
            "core_paper_example",
            True,
            "Two domains share the same operational-coordinate law while the target is shifted in nuisance coordinates.",
            "The first coordinate controls terminal behavior and risk.",
            "Remaining coordinates are nuisance directions; a large shift is injected there.",
            "Geometric transport should be nontrivial, while operational transport should collapse the harmless shift.",
            ["false separation under geometry-heavy cost", "overly strict admissibility"],
            ["harmless_shift_collapse", "false_separation_score", "ordinary_vs_operational_gap"],
        )

    def generate(self) -> WorldBatch:
        rng, n, d = self.rng, self.n(), self.dim()
        op = rng.normal(0.20, 0.22, size=(n, 1))
        nui_a = rng.normal(0.0, 0.35, size=(n, max(d - 1, 1)))
        nui_b = rng.normal(1.80, 0.35, size=(n, max(d - 1, 1)))
        z_a = np.c_[op, nui_a][:, :d]
        z_b = np.c_[op + rng.normal(0, 0.025, size=(n, 1)), nui_b][:, :d]
        return self.make_batch("harmless_nuisance", z_a, z_b, {"paper_core": True})


@register_world("harmful_boundary")
class HarmfulBoundaryWorld(ContinuousOperationalWorld):
    def spec(self) -> dict:
        return _spec(
            "harmful_boundary",
            "core_paper_example",
            True,
            "Two domains are close in state space but lie on different sides of the operational failure boundary.",
            "The first coordinate is shifted across the terminal threshold.",
            "Nuisance coordinates are matched so ordinary geometry can underestimate harm.",
            "Operational transport should detect the harmful shift even when geometric transport is small.",
            ["false collapse under noisy risk", "soft admissibility leaking harmful matches"],
            ["harmful_shift_detection", "false_collapse_mass", "risk_gap_transport_mass"],
        )

    def generate(self) -> WorldBatch:
        rng, n, d = self.rng, self.n(), self.dim()
        threshold = float(self.cfg.get("world", {}).get("failure_threshold", 0.75))
        op_a = rng.normal(threshold - 0.10, 0.045, size=(n, 1))
        op_b = rng.normal(threshold + 0.10, 0.045, size=(n, 1))
        nui = rng.normal(0.0, 0.10, size=(n, max(d - 1, 1)))
        z_a = np.c_[op_a, nui + rng.normal(0, 0.02, size=nui.shape)][:, :d]
        z_b = np.c_[op_b, nui + rng.normal(0, 0.02, size=nui.shape)][:, :d]
        return self.make_batch("harmful_boundary", z_a, z_b, {"paper_core": True})


@register_world("invariance_misspecification")
class InvarianceMisspecificationWorld(ContinuousOperationalWorld):
    descriptor_mode_a = "correct"
    descriptor_mode_b = "nuisance_only"
    anchor_mode_a = "correct"
    anchor_mode_b = "nuisance_only"

    def spec(self) -> dict:
        return _spec(
            "invariance_misspecification",
            "stress_test",
            False,
            "The generated law has a correct operational coordinate, but the target-domain descriptor is intentionally misspecified.",
            "The first coordinate remains the true operational coordinate.",
            "One domain uses nuisance magnitude as compatibility, creating wrong admissible matches.",
            "Diagnostics should expose false collapse, false separation, and admissibility instability.",
            ["descriptor-induced false collapse", "descriptor-induced false separation", "low allowed-pair fraction"],
            ["false_collapse_mass", "false_separation_score", "allowed_pair_fraction"],
        )

    def generate(self) -> WorldBatch:
        rng, n, d = self.rng, self.n(), self.dim()
        op_a = rng.normal(0.45, 0.20, size=(n, 1))
        op_b = op_a + rng.normal(0, 0.18, size=(n, 1))
        op_b[: n // 3] += 0.60
        nui_a = rng.normal(0.0, 0.30, size=(n, max(d - 1, 1)))
        nui_b = rng.normal(0.9, 0.30, size=(n, max(d - 1, 1)))
        z_a = np.c_[op_a, nui_a][:, :d]
        z_b = np.c_[op_b, nui_b][:, :d]
        return self.make_batch("invariance_misspecification", z_a, z_b)


@register_world("risk_degradation")
class RiskDegradationWorld(HarmfulBoundaryWorld):
    def spec(self) -> dict:
        return _spec(
            "risk_degradation",
            "stress_test",
            False,
            "A harmful-boundary world intended to be run under true, noisy, rollout, learned, and misspecified risk.",
            "The first coordinate controls true terminal risk.",
            "Nuisance coordinates should not control risk, but misspecified risk may rely on them.",
            "Transport should degrade as risk estimates degrade.",
            ["risk estimator collapse", "misspecified risk causing invalid transport"],
            ["risk_estimation_mae", "false_collapse_mass", "transport_value"],
        )


@register_world("sample_complexity")
class SampleComplexityWorld(HarmlessNuisanceWorld):
    def spec(self) -> dict:
        return _spec(
            "sample_complexity",
            "stress_test",
            False,
            "The harmless-nuisance world under small empirical sample regimes.",
            "Operational-coordinate law remains matched across domains.",
            "Finite support can create spurious separation despite true operational equivalence.",
            "Operational distance should stabilize as sample size increases.",
            ["finite-sample support mismatch", "unstable transport plans"],
            ["transport_value", "false_separation_score", "allowed_pair_fraction"],
        )


@register_world("high_dimensional")
class HighDimensionalWorld(HarmfulBoundaryWorld):
    def spec(self) -> dict:
        return _spec(
            "high_dimensional",
            "stress_test",
            False,
            "A harmful-boundary world embedded in many nuisance dimensions.",
            "Only the first coordinate is operationally relevant.",
            "All remaining dimensions are nuisance-heavy and can dominate geometry.",
            "Operational-coordinate and risk terms should preserve harmful-shift detection in high dimension.",
            ["geometry dominated by nuisance dimensions", "descriptor instability"],
            ["ordinary_vs_operational_gap", "harmful_shift_detection"],
        )

    def generate(self) -> WorldBatch:
        self.cfg.setdefault("world", {})["dim"] = max(20, int(self.cfg.get("world", {}).get("dim", 20)))
        return super().generate()


@register_world("admissibility_stress")
class AdmissibilityStressWorld(ContinuousOperationalWorld):
    def spec(self) -> dict:
        return _spec(
            "admissibility_stress",
            "stress_test",
            False,
            "A mixed-overlap world for comparing hard, soft, hybrid, and adaptive admissibility.",
            "Operational coordinates partly overlap but a harmful subpopulation crosses the boundary.",
            "Nuisance and descriptor structure partly agree and partly conflict.",
            "Hard mode may become infeasible; soft mode may leak harmful matches; hybrid/adaptive should trade off.",
            ["hard infeasibility", "soft harmful leakage", "adaptive over-relaxation"],
            ["allowed_pair_fraction", "false_collapse_mass", "infeasible_pair_fraction"],
        )

    def generate(self) -> WorldBatch:
        rng, n, d = self.rng, self.n(), self.dim()
        threshold = float(self.cfg.get("world", {}).get("failure_threshold", 0.75))
        op_a = rng.normal(threshold - 0.18, 0.18, size=(n, 1))
        op_b = rng.normal(threshold - 0.12, 0.18, size=(n, 1))
        op_b[: n // 4] = rng.normal(threshold + 0.25, 0.05, size=(n // 4, 1))
        nui_a = rng.normal(0.0, 0.45, size=(n, max(d - 1, 1)))
        nui_b = nui_a + rng.normal(0.30, 0.25, size=nui_a.shape)
        z_a = np.c_[op_a, nui_a][:, :d]
        z_b = np.c_[op_b, nui_b][:, :d]
        return self.make_batch("admissibility_stress", z_a, z_b)


@register_world("unbalanced_dangerous_mass")
class UnbalancedDangerousMassWorld(ContinuousOperationalWorld):
    def spec(self) -> dict:
        return _spec(
            "unbalanced_dangerous_mass",
            "core_paper_example",
            True,
            "The target domain contains additional harmful mass with no source-domain counterpart.",
            "The first coordinate controls operational failure; the extra component is far past threshold.",
            "Nuisance coordinates are not the main issue; unmatched harmful support is.",
            "Unbalanced OT should expose dangerous unmatched mass instead of forcing bad alignment.",
            ["balanced OT forcing invalid matches", "unbalanced fallback underestimating harmful mass"],
            ["unmatched_mass_b", "dangerous_unmatched_mass_b", "false_collapse_mass"],
        )

    def generate(self) -> WorldBatch:
        rng, n, d = self.rng, self.n(), self.dim()
        n_core = int(0.72 * n)
        n_danger = n - n_core
        op_a = rng.normal(0.30, 0.18, size=(n, 1))
        nui_a = rng.normal(0.0, 0.30, size=(n, max(d - 1, 1)))
        op_b_core = rng.normal(0.32, 0.18, size=(n_core, 1))
        nui_b_core = rng.normal(0.0, 0.30, size=(n_core, max(d - 1, 1)))
        op_b_danger = rng.normal(1.25, 0.08, size=(n_danger, 1))
        nui_b_danger = rng.normal(0.0, 0.30, size=(n_danger, max(d - 1, 1)))
        z_a = np.c_[op_a, nui_a][:, :d]
        z_b = np.vstack([np.c_[op_b_core, nui_b_core], np.c_[op_b_danger, nui_b_danger]])[:, :d]
        return self.make_batch("unbalanced_dangerous_mass", z_a, z_b, {"paper_core": True, "dangerous_component_start_b": n_core})
