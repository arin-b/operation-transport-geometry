from __future__ import annotations

from pathlib import Path
import csv
import json


def write_suite_outputs(out_dir: str | Path, results: list[dict]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in results:
        artifact_summary = item.get("summary")
        row = {
            "world": item.get("world"),
            "out": item.get("out"),
            "system_score": (artifact_summary or {}).get("system_score", {}).get("value"),
        }
        if artifact_summary:
            nodes = artifact_summary.get("nodes", {})
            for node, data in nodes.items():
                diag = data.get("diagnostics", {})
                row.update({
                    f"{node}_transport_value": data.get("transport_value"),
                    f"{node}_risk_mae": diag.get("risk_estimation_mae"),
                    f"{node}_false_collapse": diag.get("false_collapse_mass"),
                    f"{node}_forbidden_mass": diag.get("transport_forbidden_mass"),
                    f"{node}_dangerous_unmatched_b": diag.get("dangerous_unmatched_mass_b"),
                })
        rows.append(row)

    keys = sorted({k for row in rows for k in row})
    with (out / "suite_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    md = ["# OTG Suite Report", ""]
    md.append("| World | System score | Output |")
    md.append("|---|---:|---|")
    for row in rows:
        val = row.get("system_score")
        val_s = "" if val is None else f"{float(val):.6f}"
        md.append(f"| `{row.get('world')}` | {val_s} | `{row.get('out')}` |")
    md.append("")
    md.append("Full per-world reports are stored in the corresponding run directories.")
    (out / "suite_report.md").write_text("\n".join(md), encoding="utf-8")
    (out / "suite_summary.json").write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
