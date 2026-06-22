from __future__ import annotations

from pathlib import Path
import csv

from otg.core.pipeline import run_pipeline
from otg.utils.config import load_config, deep_merge
from otg.utils.io import ensure_dir, save_json
from otg.validation.expectations import graph_artifact_checks, reproducibility_check, ValidationCheck


VALIDATION_CASES: list[dict] = [
    {
        "name": "graph_masked_sinkhorn",
        "overrides": {"world": {"name": "synthetic_dag"}, "risk": {"mode": "true"}, "transport": {"solver": "masked_sinkhorn"}},
    },
    {
        "name": "graph_unbalanced_dangerous_mass",
        "overrides": {"world": {"name": "synthetic_dag"}, "risk": {"mode": "true"}, "transport": {"solver": "unbalanced"}},
    },
    {
        "name": "graph_projection_off",
        "overrides": {"world": {"name": "synthetic_dag"}, "projection": {"use_aligned": False}, "transport": {"solver": "masked_sinkhorn"}},
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
            "runtime_values": {"n": 48 if preset == "fast" else 100, "mc_rollouts": 16 if preset == "fast" else 64},
            "seed": {"master": seed + 1000 * i},
        })
        case_dir = out / case["name"]
        artifact = run_pipeline(cfg, case_dir)
        checks.extend(graph_artifact_checks(case["name"], artifact))
        case_summaries.append({
            "case": case["name"],
            "world": artifact.world_name,
            "system_score": artifact.system_score,
            "domain_order": artifact.domain_order,
            "selected_nodes": artifact.selected_nodes,
            "out": str(case_dir),
        })

    repro_cfg = deep_merge(base, {
        "world": {"name": "synthetic_dag"},
        "risk": {"mode": "true"},
        "transport": {"solver": "masked_sinkhorn"},
        "runtime": {"preset": preset},
        "runtime_values": {"n": 40, "mc_rollouts": 8},
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
        "# OTG Graph-Level Validation Report",
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
        lines.append(f"- `{case['case']}` -> world `{case['world']}`, domains `{case['domain_order']}`, nodes `{case['selected_nodes']}`, score `{case['system_score']['value']:.6f}`")
    lines.extend(["", "## Failed checks", ""])
    failed = [c for c in summary["checks"] if not c["passed"]]
    if not failed:
        lines.append("No failed checks.")
    else:
        for c in failed:
            lines.append(f"- `{c['name']}` [{c['severity']}]: value `{c['value']}`, threshold `{c['threshold']}`. {c['message']}")
    path.write_text("\n".join(lines), encoding="utf-8")
