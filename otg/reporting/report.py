from __future__ import annotations

from pathlib import Path
import csv
import json

from otg.diagnostics.scorecard import interpret_node, status_from_diagnostics


def write_run_outputs(out_dir: str | Path, artifact, batch) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = batch.metadata.get("world_spec") or {}

    summary = build_summary(artifact, batch, spec)
    write_report(out, artifact.config, summary)
    write_metrics_csv(out, summary)
    write_latex_tables(out, summary)
    write_interpretation(out, artifact.config, summary)
    (out / "world_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    try:
        from otg.diagnostics.plots import plot_run_outputs
        plot_run_outputs(out, artifact, batch)
    except Exception as exc:
        (out / "plot_error.txt").write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")


def build_summary(artifact, batch, spec: dict) -> dict:
    return {
        "world": artifact.world_name,
        "preset": artifact.config.get("runtime", {}).get("preset", "unknown"),
        "system_score": artifact.system_score,
        "assumptions": artifact.assumption_report,
        "world_spec": spec,
        "node_results": {
            k: {
                "transport_value": float(v.transport.value),
                "solver": v.transport.solver,
                "status": v.transport.status,
                "diagnostics": v.diagnostics,
                "cost_mode": v.cost.mode,
                "invariance_mode": v.invariance.mode,
                "transport_metadata": v.transport.metadata,
                "invariance_metadata": summarize_invariance_metadata(v.invariance.metadata),
            }
            for k, v in artifact.node_results.items()
        },
    }


def summarize_invariance_metadata(meta: dict) -> dict:
    out = {}
    for k, v in (meta or {}).items():
        if hasattr(v, "shape"):
            out[k] = {"shape": list(v.shape)}
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, dict):
            out[k] = {str(kk): vv for kk, vv in v.items() if isinstance(vv, (int, float, str, bool)) or vv is None}
    return out


def write_report(out_dir: str | Path, cfg: dict, summary: dict) -> None:
    out = Path(out_dir)
    spec = summary.get("world_spec") or {}

    md = []
    md.append("# OTG Experiment Report")
    md.append("")
    md.append(f"World: `{summary['world']}`")
    md.append(f"Preset: `{summary.get('preset', 'unknown')}`")
    md.append(f"System score: `{summary['system_score']['value']:.6f}`")
    md.append("")
    if spec:
        md.append("## Controlled operational world")
        md.append("")
        md.append(f"Family: `{spec.get('family')}`")
        md.append(f"Paper core example: `{spec.get('paper_core')}`")
        md.append("")
        md.append(f"Mathematical definition: {spec.get('mathematical_definition')}")
        md.append("")
        md.append(f"Operational coordinate rule: {spec.get('operational_coordinate_rule')}")
        md.append("")
        md.append(f"Nuisance rule: {spec.get('nuisance_rule')}")
        md.append("")
        md.append(f"Expected behavior: {spec.get('expected_behavior')}")
        md.append("")

    md.append("## Node results")
    md.append("")
    for node, result in summary["node_results"].items():
        score = status_from_diagnostics(summary["world"], result["diagnostics"], cfg)
        md.append(f"### {node}")
        md.append("")
        md.append(f"Status: `{score['status']}`")
        md.append(f"Transport value: `{result['transport_value']:.6f}`")
        md.append(f"Solver: `{result['solver']}`")
        md.append(f"Solver status: `{result['status']}`")
        md.append(f"Cost mode: `{result.get('cost_mode')}`")
        md.append(f"Invariance mode: `{result.get('invariance_mode')}`")
        md.append("")
        md.append("Interpretation: " + interpret_node(summary["world"], node, result["diagnostics"], cfg))
        md.append("")
        md.append("| Metric | Value |")
        md.append("|---|---:|")
        for k, v in sorted(result["diagnostics"].items()):
            try:
                md.append(f"| `{k}` | {float(v):.6f} |")
            except Exception:
                md.append(f"| `{k}` | {v} |")
        md.append("")
        md.append("Scorecard checks:")
        md.append("")
        md.append("| Check | Value | Threshold | Pass |")
        md.append("|---|---:|---:|---|")
        for k, c in score["checks"].items():
            md.append(f"| `{k}` | {c['value']:.6f} | {c['threshold']:.6f} | `{c['pass']}` |")
        md.append("")

    md.append("## Assumption report")
    md.append("")
    md.append("```json")
    md.append(json.dumps(summary["assumptions"], indent=2))
    md.append("```")
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")


def write_metrics_csv(out_dir: str | Path, summary: dict) -> None:
    out = Path(out_dir)
    rows = []
    all_keys = set()
    for node, result in summary["node_results"].items():
        row = {
            "world": summary["world"],
            "node": node,
            "transport_value": result["transport_value"],
            "solver": result["solver"],
            "status": result["status"],
            "cost_mode": result.get("cost_mode"),
            "invariance_mode": result.get("invariance_mode"),
        }
        row.update(result["diagnostics"])
        rows.append(row)
        all_keys.update(row.keys())

    fieldnames = ["world", "node", "transport_value", "solver", "status", "cost_mode", "invariance_mode"] + sorted(k for k in all_keys if k not in {"world", "node", "transport_value", "solver", "status", "cost_mode", "invariance_mode"})
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Compact scorecard CSV.
    with (out / "scorecard.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["world", "node", "check", "value", "threshold", "pass"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for node, result in summary["node_results"].items():
            score = status_from_diagnostics(summary["world"], result["diagnostics"], {})
            for check, data in score["checks"].items():
                writer.writerow({
                    "world": summary["world"],
                    "node": node,
                    "check": check,
                    "value": data["value"],
                    "threshold": data["threshold"],
                    "pass": data["pass"],
                })


def write_latex_tables(out_dir: str | Path, summary: dict) -> None:
    out = Path(out_dir)
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Node & OT & Risk MAE & False Collapse & Forbidden & Unmatched B & Dangerous B \\",
        r"\midrule",
    ]
    for node, result in summary["node_results"].items():
        d = result["diagnostics"]
        lines.append(
            f"{node} & {result['transport_value']:.4f} & {d['risk_estimation_mae']:.4f} & {d['false_collapse_mass']:.4f} & {d.get('transport_forbidden_mass', 0.0):.4f} & {d.get('unmatched_mass_b', 0.0):.4f} & {d.get('dangerous_unmatched_mass_b', 0.0):.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "table_snippet.tex").write_text("\n".join(lines), encoding="utf-8")

    assumption_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Node & Allowed & Row feasible & Col feasible & Risk MAE \\",
        r"\midrule",
    ]
    for node, result in summary["node_results"].items():
        d = result["diagnostics"]
        assumption_lines.append(
            f"{node} & {d.get('allowed_pair_fraction', 0.0):.4f} & {d.get('row_feasible_fraction', 0.0):.4f} & {d.get('col_feasible_fraction', 0.0):.4f} & {d.get('risk_estimation_mae', 0.0):.4f} \\\\"
        )
    assumption_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "assumption_table_snippet.tex").write_text("\n".join(assumption_lines), encoding="utf-8")


def write_interpretation(out_dir: str | Path, cfg: dict, summary: dict) -> None:
    out = Path(out_dir)
    lines = ["# Compact Interpretation", ""]
    lines.append(f"World: `{summary['world']}`.")
    spec = summary.get("world_spec") or {}
    if spec.get("expected_behavior"):
        lines.append("")
        lines.append(f"Expected behavior: {spec['expected_behavior']}")
    lines.append("")
    for node, result in summary["node_results"].items():
        lines.append(f"## {node}")
        lines.append("")
        lines.append(interpret_node(summary["world"], node, result["diagnostics"], cfg))
        lines.append("")
    (out / "interpretation.md").write_text("\n".join(lines), encoding="utf-8")
