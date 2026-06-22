from __future__ import annotations

from pathlib import Path
import csv
import json
from dataclasses import asdict

from otg.core.pipeline import run_pipeline
from otg.utils.config import load_config, deep_merge
from otg.utils.io import ensure_dir, save_json
from otg.validation.expectations import artifact_checks, reproducibility_check, ValidationCheck


VALIDATION_CASES: list[dict] = [
    {
        "name": "harmless_nuisance_core",
        "overrides": {"world": {"name": "harmless_nuisance"}, "risk": {"mode": "true"}, "transport": {"solver": "masked_sinkhorn"}},
    },
    {
        "name": "harmful_boundary_core",
        "overrides": {"world": {"name": "harmful_boundary"}, "risk": {"mode": "true"}, "transport": {"solver": "masked_sinkhorn"}},
    },
    {
        "name": "risk_degradation_noisy",
        "overrides": {"world": {"name": "risk_degradation"}, "risk": {"mode": "noisy"}, "transport": {"solver": "masked_sinkhorn"}},
    },
    {
        "name": "invariance_misspecification_core",
        "overrides": {"world": {"name": "invariance_misspecification"}, "risk": {"mode": "true"}, "transport": {"solver": "masked_sinkhorn"}},
    },
    {
        "name": "unbalanced_dangerous_mass_core",
        "overrides": {"world": {"name": "unbalanced_dangerous_mass"}, "risk": {"mode": "true"}, "transport": {"solver": "unbalanced"}},
    },
]


def run_validation(out_dir: str | Path, *, preset: str = "fast", seed: int = 0, strict: bool = False) -> dict:
    out = ensure_dir(out_dir)
    base = load_config("configs/default.yaml")
    checks: list[ValidationCheck] = []
    case_summaries: list[dict] = []

    for i, case in enumerate(VALIDATION_CASES):
        cfg = deep_merge(base, case["overrides"])
        cfg = deep_merge(cfg, {
            "runtime": {"preset": preset},
            "runtime_values": {"n": 36 if preset == "fast" else 80, "mc_rollouts": 16 if preset == "fast" else 64},
            "seed": {"master": seed + 1000 * i},
        })
        case_dir = out / case["name"]
        artifact = run_pipeline(cfg, case_dir)
        checks.extend(artifact_checks(case["name"], artifact))
        case_summaries.append({
            "case": case["name"],
            "world": artifact.world_name,
            "system_score": artifact.system_score,
            "out": str(case_dir),
        })

    # Reproducibility check on a small deterministic case.
    repro_cfg = deep_merge(base, {
        "world": {"name": "harmful_boundary"},
        "risk": {"mode": "true"},
        "transport": {"solver": "masked_sinkhorn"},
        "runtime": {"preset": preset},
        "runtime_values": {"n": 32, "mc_rollouts": 8},
        "seed": {"master": seed + 99991},
    })
    a = run_pipeline(repro_cfg, out / "repro_a")
    b = run_pipeline(repro_cfg, out / "repro_b")
    checks.append(reproducibility_check(a, b))

    hard_failures = [c for c in checks if not c.passed and c.severity == "error"]
    warnings = [c for c in checks if not c.passed and c.severity == "warn"]
    status = "pass" if not hard_failures else "fail"
    if strict and warnings:
        status = "fail"

    summary = {
        "status": status,
        "strict": strict,
        "num_checks": len(checks),
        "num_errors": len(hard_failures),
        "num_warnings": len(warnings),
        "cases": case_summaries,
        "checks": [c.to_dict() for c in checks],
    }
    save_json(out / "validation_results.json", summary)
    _write_csv(out / "validation_checks.csv", checks)
    _write_markdown(out / "validation_report.md", summary)
    return summary


def _write_csv(path: Path, checks: list[ValidationCheck]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "passed", "severity", "value", "threshold", "message"])
        writer.writeheader()
        for c in checks:
            writer.writerow(c.to_dict())


def _write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# OTG Validation Report",
        "",
        f"Status: `{summary['status']}`",
        f"Checks: `{summary['num_checks']}`",
        f"Errors: `{summary['num_errors']}`",
        f"Warnings: `{summary['num_warnings']}`",
        "",
        "## Cases",
        "",
    ]
    for case in summary["cases"]:
        lines.append(f"- `{case['case']}` -> world `{case['world']}`, system score `{case['system_score']['value']:.6f}`")
    lines.extend(["", "## Failed checks", ""])
    failed = [c for c in summary["checks"] if not c["passed"]]
    if not failed:
        lines.append("No failed checks.")
    else:
        for c in failed:
            lines.append(f"- `{c['name']}` [{c['severity']}]: value `{c['value']}`, threshold `{c['threshold']}`. {c['message']}")
    path.write_text("\n".join(lines), encoding="utf-8")
