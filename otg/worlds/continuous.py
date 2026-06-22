from __future__ import annotations
import numpy as np

from otg.core.schema import DomainSpec, NodeBatch, WorldBatch
from otg.worlds.base import ControlledOperationalWorld
from otg.worlds.registry import register_world


def _preset_values(cfg: dict) -> tuple[int, int]:
    preset = cfg.get("runtime", {}).get("preset", "fast")
    sizes = {
        "fast": (60, 32),
        "default": (180, 128),
        "heavy": (600, 512),
    }
    n, mc = sizes.get(preset, sizes["fast"])
    cfg.setdefault("runtime_values", {})["n"] = n
    cfg.setdefault("runtime_values", {})["mc_rollouts"] = mc
    return n, mc


def _mass(n: int) -> np.ndarray:
    return np.ones(n, dtype=float) / n


def _risk(y: np.ndarray, threshold: float, sharpness: float = 8.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-sharpness * (y - threshold)))


class ContinuousBaseWorld(ControlledOperationalWorld):
    world_name = "continuous_base"

    def make_node(self, z_a: np.ndarray, z_b: np.ndarray, expected: str) -> WorldBatch:
        threshold = float(self.cfg.get("world", {}).get("failure_threshold", 0.75))
        rng = self.rng("terminal")
        w = np.zeros(z_a.shape[1])
        w[0] = 1.0
        if z_a.shape[1] > 1:
            w[1] = float(self.cfg.get("world", {}).get("nuisance_terminal_leak", 0.20))
        terminal_noise = float(self.cfg.get("uncertainty", {}).get("terminal_noise", 0.03))
        y_a = z_a @ w + rng.normal(0, terminal_noise, len(z_a))
        y_b = z_b @ w + rng.normal(0, terminal_noise, len(z_b))
        true_a = _risk(y_a, threshold)
        true_b = _risk(y_b, threshold)

        op_a = z_a[:, :1]
        op_b = z_b[:, :1]
        nuis_a = z_a[:, 1:] if z_a.shape[1] > 1 else np.zeros((len(z_a), 0))
        nuis_b = z_b[:, 1:] if z_b.shape[1] > 1 else np.zeros((len(z_b), 0))
        desc_a = np.c_[op_a[:, 0], np.linalg.norm(nuis_a, axis=1) if nuis_a.size else np.zeros(len(z_a))]
        desc_b = np.c_[op_b[:, 0], np.linalg.norm(nuis_b, axis=1) if nuis_b.size else np.zeros(len(z_b))]
        anchors_a = (op_a[:, 0] > 0).astype(int)
        anchors_b = (op_b[:, 0] > 0).astype(int)

        node = NodeBatch(
            node="repr",
            z_a=z_a,
            z_b=z_b,
            terminal_a=y_a,
            terminal_b=y_b,
            true_risk_a=true_a,
            true_risk_b=true_b,
            used_risk_a=true_a.copy(),
            used_risk_b=true_b.copy(),
            operational_coords_a=op_a,
            operational_coords_b=op_b,
            nuisance_coords_a=nuis_a,
            nuisance_coords_b=nuis_b,
            descriptor_coords_a=desc_a,
            descriptor_coords_b=desc_b,
            anchors_a=anchors_a,
            anchors_b=anchors_b,
            mass_a=_mass(len(z_a)),
            mass_b=_mass(len(z_b)),
            failure_a=true_a > 0.5,
            failure_b=true_b > 0.5,
            metadata={"expected": expected},
        )
        return WorldBatch(
            name=self.world_name,
            domains=(DomainSpec("A"), DomainSpec("B")),
            nodes={"repr": node},
            metadata={"expected": expected, "dim": z_a.shape[1]},
        )


@register_world("harmless_nuisance")
class HarmlessNuisanceWorld(ContinuousBaseWorld):
    world_name = "harmless_nuisance"

    def generate(self) -> WorldBatch:
        rng = self.rng("sample")
        n, _ = _preset_values(self.cfg)
        d = int(self.cfg.get("world", {}).get("dim", 2))
        mean_a = np.zeros(d)
        mean_b = np.zeros(d)
        if d > 1:
            mean_b[1] = 1.6
        cov = np.eye(d) * 0.18
        z_a = rng.multivariate_normal(mean_a, cov, n)
        z_b = rng.multivariate_normal(mean_b, cov, n)
        return self.make_node(z_a, z_b, "ordinary geometry moves, downstream operational behavior mostly stable")


@register_world("harmful_boundary")
class HarmfulBoundaryWorld(ContinuousBaseWorld):
    world_name = "harmful_boundary"

    def generate(self) -> WorldBatch:
        rng = self.rng("sample")
        n, _ = _preset_values(self.cfg)
        d = int(self.cfg.get("world", {}).get("dim", 2))
        mean_a = np.zeros(d); mean_b = np.zeros(d)
        mean_a[0] = 0.55; mean_b[0] = 0.90
        cov = np.eye(d) * 0.035
        z_a = rng.multivariate_normal(mean_a, cov, n)
        z_b = rng.multivariate_normal(mean_b, cov, n)
        return self.make_node(z_a, z_b, "small geometric shift crosses operational boundary")


@register_world("invariance_misspecification")
class InvarianceMisspecificationWorld(ContinuousBaseWorld):
    world_name = "invariance_misspecification"

    def generate(self) -> WorldBatch:
        rng = self.rng("sample")
        n, _ = _preset_values(self.cfg)
        d = int(self.cfg.get("world", {}).get("dim", 2))
        z_a = rng.normal(0, 0.35, (n, d))
        z_b = rng.normal(0, 0.35, (n, d))
        z_b[: n // 3, 0] += 1.0
        if d > 1:
            z_b[:, 1] += 1.0
        return self.make_node(z_a, z_b, "invariance specification controls false collapse and false separation")


@register_world("risk_degradation")
class RiskDegradationWorld(HarmfulBoundaryWorld):
    world_name = "risk_degradation"


@register_world("sample_complexity")
class SampleComplexityWorld(HarmlessNuisanceWorld):
    world_name = "sample_complexity"


@register_world("high_dimensional")
class HighDimensionalWorld(HarmfulBoundaryWorld):
    world_name = "high_dimensional"

    def generate(self) -> WorldBatch:
        self.cfg.setdefault("world", {})["dim"] = max(20, int(self.cfg.get("world", {}).get("dim", 20)))
        return super().generate()


@register_world("admissibility_stress")
class AdmissibilityStressWorld(InvarianceMisspecificationWorld):
    world_name = "admissibility_stress"


@register_world("unbalanced_dangerous_mass")
class UnbalancedDangerousMassWorld(ContinuousBaseWorld):
    world_name = "unbalanced_dangerous_mass"

    def generate(self) -> WorldBatch:
        rng = self.rng("sample")
        n, _ = _preset_values(self.cfg)
        d = int(self.cfg.get("world", {}).get("dim", 2))
        n_safe = int(0.75 * n)
        n_danger = n - n_safe
        z_a = rng.normal(0, 0.32, (n, d))
        z_b_safe = rng.normal(0, 0.32, (n_safe, d))
        danger_mean = np.zeros(d); danger_mean[0] = 1.55
        z_b_danger = rng.multivariate_normal(danger_mean, np.eye(d) * 0.04, n_danger)
        z_b = np.vstack([z_b_safe, z_b_danger])
        return self.make_node(z_a, z_b, "dangerous mass has no safe counterpart and should remain unmatched")
