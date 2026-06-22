from __future__ import annotations

import numpy as np

from otg.core.schema import DomainSpec, GraphWorldBatch, NodeLaw
from otg.worlds.base import ControlledOperationalWorld
from otg.worlds.registry import register_world


def _soft_failure(terminal: np.ndarray) -> np.ndarray:
    adequacy = terminal[:, 0]
    coverage = terminal[:, 1]
    glare_flag = terminal[:, 2]
    uncertainty = terminal[:, 3]
    score = 3.2 * (0.58 - adequacy) + 2.5 * (0.55 - coverage) + 1.7 * glare_flag + 1.2 * uncertainty
    return 1.0 / (1.0 + np.exp(-score))


def _mass(n: int) -> np.ndarray:
    return np.ones(n, dtype=float) / n


def _anchors(adequacy: np.ndarray, coverage: np.ndarray, glare: np.ndarray) -> np.ndarray:
    # semantic compatibility classes: adequate/no-glare, low-coverage, glare-confounded, severe-fail
    severe = (adequacy < 0.42) | (coverage < 0.42)
    glare_c = glare > 0.55
    low_cov = coverage < 0.55
    return (severe.astype(int) * 3 + (~severe & glare_c).astype(int) * 2 + (~severe & ~glare_c & low_cov).astype(int)).astype(int)


def _projection(raw_shared: np.ndarray, domain: str, node: str, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    d = raw_shared.shape[1]
    if domain == "clear":
        A = np.eye(d)
        b = np.zeros(d)
        noise = 0.0
        raw = raw_shared.copy()
        kind = "identity"
    else:
        # Domain-specific coordinate realization M_v^(d). The graph stores both raw samples
        # and the aligned shared-space samples phi_v^d(raw) = mutilde-space.
        scale = {
            "viewpoint_shift": 1.15,
            "glare": 0.92,
            "occlusion": 1.05,
        }.get(domain, 1.0)
        angle = {
            "detector": 0.10,
            "representation": -0.08,
            "measurement": 0.04,
            "input": 0.02,
            "audit_report": 0.0,
        }.get(node, 0.0)
        A = np.eye(d) * scale
        if d >= 2:
            R = np.eye(d)
            c, s = np.cos(angle), np.sin(angle)
            R[:2, :2] = [[c, -s], [s, c]]
            A = R @ A
        b = np.linspace(0.02, 0.06, d) * ({"viewpoint_shift": 1.0, "glare": -1.0, "occlusion": 0.5}.get(domain, 0.0))
        noise = 0.01
        raw = raw_shared @ A.T + b + rng.normal(0.0, noise, size=raw_shared.shape)
        kind = "domain_specific_affine_noisy"
    return raw, {
        "kind": kind,
        "node": node,
        "domain": domain,
        "A": A.tolist(),
        "b": b.tolist(),
        "noise_std": noise,
        "map": "phi_v^d(raw) is represented by aligned_samples; raw samples are retained for projection diagnostics.",
    }


@register_world("synthetic_dag")
class SyntheticDAGWorld(ControlledOperationalWorld):
    """Minimum proposal-faithful synthetic deployed DAG world."""

    def spec(self) -> dict:
        return {
            "name": "synthetic_dag",
            "family": "proposal_faithful_graph_world",
            "paper_core": True,
            "mathematical_definition": "Four deployment domains induce empirical node-wise laws on a finite DAG with input, detector, representation, measurement, and terminal audit_report nodes.",
            "operational_coordinate_rule": "Terminal audit quantities are adequacy, coverage, glare/confounder flag, and uncertainty/confidence; node-wise risk is induced by terminal failure probability.",
            "nuisance_rule": "Viewpoint changes mostly affect nuisance coordinates, glare changes confounder/uncertainty, and occlusion changes coverage and produces dangerous unmatched target mass.",
            "expected_behavior": "clear-vs-viewpoint should mostly collapse; clear-vs-glare and clear-vs-occlusion should separate; occlusion should expose dangerous unmatched mass under unbalanced OT.",
            "failure_modes": ["semantic mismatch", "hidden confounding", "forced matching of dangerous target mass", "projection hurt/help"],
            "recommended_diagnostics": ["D_op_matrix", "node_by_domain_pair", "admissibility_forbidden_mass", "dangerous_unmatched_mass", "projection_gap"],
        }

    def generate(self) -> GraphWorldBatch:
        rng = self.rng
        n = int(self.cfg.get("runtime_values", {}).get("n", self.cfg.get("world", {}).get("n", 80)))
        domains = ["clear", "viewpoint_shift", "glare", "occlusion"]
        selected_nodes = list(self.cfg.get("nodes", {}).get("selected", ["detector", "representation", "measurement"]))
        all_nodes = ["input", "detector", "representation", "measurement", "audit_report"]

        # Shared scene variables represent P_d inputs before graph processing. Domains alter
        # the input law, not just a final pairwise proxy.
        base_adequacy = rng.beta(6.0, 3.0, size=n)
        base_coverage = rng.beta(6.5, 2.8, size=n)
        base_complexity = rng.beta(2.0, 5.0, size=n)
        base_viewpoint = rng.normal(0.0, 0.18, size=n)
        base_noise = rng.normal(0.0, 0.05, size=n)

        domain_specs: dict[str, DomainSpec] = {}
        node_laws: dict[tuple[str, str], NodeLaw] = {}
        projection_meta: dict[str, dict] = {}

        for domain in domains:
            adequacy = base_adequacy.copy()
            coverage = base_coverage.copy()
            complexity = base_complexity.copy()
            viewpoint = base_viewpoint.copy()
            glare = np.zeros(n)
            occlusion = np.zeros(n)
            domain_noise = base_noise.copy()

            if domain == "viewpoint_shift":
                viewpoint = viewpoint + rng.normal(0.65, 0.06, size=n)
                domain_noise = domain_noise + rng.normal(0.0, 0.035, size=n)
            elif domain == "glare":
                glare = np.clip(rng.normal(0.85, 0.10, size=n), 0, 1)
                adequacy = np.clip(adequacy - 0.05 * glare, 0, 1)
                domain_noise = domain_noise + rng.normal(0.05, 0.05, size=n)
            elif domain == "occlusion":
                occlusion = np.zeros(n)
                k = max(1, int(0.28 * n))
                dangerous = rng.choice(n, size=k, replace=False)
                occlusion[dangerous] = np.clip(rng.normal(0.88, 0.08, size=k), 0, 1)
                coverage = np.clip(coverage - 0.48 * occlusion, 0, 1)
                adequacy = np.clip(adequacy - 0.10 * occlusion, 0, 1)
                complexity = np.clip(complexity + 0.25 * occlusion, 0, 1)

            domain_specs[domain] = DomainSpec(
                name=domain,
                id=domain,
                input_law_metadata={
                    "P_d": "synthetic scene law with domain-specific nuisance/confounder perturbations",
                    "mean_adequacy": float(np.mean(adequacy)),
                    "mean_coverage": float(np.mean(coverage)),
                    "mean_glare": float(np.mean(glare)),
                    "mean_occlusion": float(np.mean(occlusion)),
                },
                metadata={"role": "deployment_domain"},
            )

            # Deployed DAG forward pass.
            input_shared = np.c_[adequacy, coverage, complexity, viewpoint, glare, occlusion, domain_noise]
            detection_score = np.clip(0.86 - 0.08 * complexity - 0.30 * glare - 0.34 * occlusion + rng.normal(0, 0.025, n), 0, 1)
            localization_error = np.clip(0.05 + 0.08 * np.abs(viewpoint) + 0.18 * occlusion + 0.10 * glare + rng.normal(0, 0.015, n), 0, 1)
            detector_shared = np.c_[detection_score, localization_error, viewpoint, glare, occlusion]

            rep_op_adequacy = np.clip(adequacy * detection_score - 0.05 * glare + rng.normal(0, 0.02, n), 0, 1)
            rep_op_coverage = np.clip(coverage * (1 - 0.35 * occlusion) - 0.04 * localization_error + rng.normal(0, 0.02, n), 0, 1)
            rep_shared = np.c_[rep_op_adequacy, rep_op_coverage, viewpoint, complexity, glare, occlusion]

            meas_adequacy = np.clip(rep_op_adequacy - 0.08 * glare - 0.06 * occlusion + rng.normal(0, 0.018, n), 0, 1)
            meas_coverage = np.clip(rep_op_coverage - 0.18 * occlusion + rng.normal(0, 0.018, n), 0, 1)
            meas_glare = np.clip(glare + 0.15 * domain_noise + rng.normal(0, 0.02, n), 0, 1)
            meas_uncertainty = np.clip(0.10 + 0.25 * glare + 0.35 * occlusion + 0.12 * localization_error + rng.normal(0, 0.02, n), 0, 1)
            measurement_shared = np.c_[meas_adequacy, meas_coverage, meas_glare, meas_uncertainty]

            audit_report = measurement_shared.copy()
            true_risk = _soft_failure(audit_report)
            failure = true_risk > 0.50
            anchors = _anchors(meas_adequacy, meas_coverage, meas_glare)
            terminal = audit_report

            node_shared = {
                "input": input_shared,
                "detector": detector_shared,
                "representation": rep_shared,
                "measurement": measurement_shared,
                "audit_report": audit_report,
            }
            node_operational = {
                "input": np.c_[adequacy, coverage],
                "detector": np.c_[detection_score, 1 - localization_error],
                "representation": np.c_[rep_op_adequacy, rep_op_coverage],
                "measurement": np.c_[meas_adequacy, meas_coverage, 1 - meas_uncertainty],
                "audit_report": np.c_[meas_adequacy, meas_coverage, 1 - meas_uncertainty],
            }
            node_nuisance = {
                "input": np.c_[viewpoint, complexity, domain_noise],
                "detector": np.c_[viewpoint, localization_error],
                "representation": np.c_[viewpoint, complexity],
                "measurement": np.c_[meas_glare, meas_uncertainty],
                "audit_report": np.c_[meas_glare, meas_uncertainty],
            }
            node_descriptors = {
                "input": np.c_[adequacy, coverage, glare, occlusion],
                "detector": np.c_[detection_score, localization_error, glare, occlusion],
                "representation": np.c_[rep_op_adequacy, rep_op_coverage, glare, occlusion],
                "measurement": np.c_[meas_adequacy, meas_coverage, meas_glare, meas_uncertainty],
                "audit_report": np.c_[meas_adequacy, meas_coverage, meas_glare, meas_uncertainty],
            }

            for node in all_nodes:
                aligned = np.asarray(node_shared[node], dtype=float)
                raw, proj = _projection(aligned, domain, node, rng)
                projection_meta[f"{node}::{domain}"] = proj
                node_laws[(node, domain)] = NodeLaw(
                    node=node,
                    domain=domain,
                    samples=raw,
                    aligned_samples=aligned,
                    terminal_outputs=terminal,
                    true_risk=true_risk.copy(),
                    used_risk=true_risk.copy(),
                    operational_coords=np.asarray(node_operational[node], dtype=float),
                    nuisance_coords=np.asarray(node_nuisance[node], dtype=float),
                    descriptor_coords=np.asarray(node_descriptors[node], dtype=float),
                    semantic_anchors=anchors.copy(),
                    mass=_mass(n),
                    failure=failure.copy(),
                    projection_metadata=proj,
                    metadata={
                        "mu_v^d": "empirical node-wise law induced by pushing P_d through the deployed DAG",
                        "mutilde_v^d": "aligned_samples after phi_v^d projection",
                        "risk_induction": "true_risk is computed from terminal audit_report evaluation, not assigned independently",
                        "domain": domain,
                    },
                )

        graph = {
            "V": all_nodes,
            "E": [
                ["input", "detector"],
                ["detector", "representation"],
                ["representation", "measurement"],
                ["measurement", "audit_report"],
            ],
            "terminal": "audit_report",
            "selected_nodes": selected_nodes,
        }
        return GraphWorldBatch(
            name="synthetic_dag",
            graph=graph,
            domains=domain_specs,
            selected_nodes=selected_nodes,
            node_laws=node_laws,
            projection_metadata=projection_meta,
            terminal_evaluation_metadata={
                "E_op(Y)": "audit_report=(adequacy, coverage, glare_or_confounder, uncertainty)",
                "rho": "sigmoid threshold risk from adequacy, coverage, glare/confounder, and uncertainty",
                "risk_formula": "sigmoid(3.2(0.58-adequacy)+2.5(0.55-coverage)+1.7*glare+1.2*uncertainty)",
            },
            metadata={"world_spec": self.spec(), "proposal_aligned": True},
        )
