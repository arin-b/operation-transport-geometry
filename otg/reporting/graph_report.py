from __future__ import annotations

from pathlib import Path
import csv
import json
import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from otg.utils.io import ensure_dir, save_array, save_json


def write_graph_run_outputs(out_dir: str | Path, artifact, batch) -> None:
    out = ensure_dir(out_dir)
    ensure_dir(out / "arrays")
    ensure_dir(out / "figures")
    ensure_dir(out / "node_pairs")

    save_json(out / "config.resolved.json", artifact.config)
    save_json(out / "graph_summary.json", graph_summary(artifact, batch))
    save_json(out / "summary.json", graph_summary(artifact, batch))
    save_json(out / "assumptions.json", artifact.assumption_report)
    save_json(out / "domain_pair_results.json", _domain_pair_json(artifact))
    save_json(out / "node_pair_table.json", artifact.node_pair_table)
    save_json(out / "world_spec.json", batch.metadata.get("world_spec", {}))
    save_json(out / "projection_metadata.json", batch.projection_metadata)
    save_json(out / "terminal_evaluation_metadata.json", batch.terminal_evaluation_metadata)
    save_json(out / "seeds.json", artifact.metadata.get("seeds", {}))

    save_array(out / "arrays" / "D_op_matrix.npy", artifact.discrepancy_matrix)
    write_domain_pair_matrix_csv(out / "D_op_matrix.csv", artifact)
    write_node_pair_metrics_csv(out / "node_pair_metrics.csv", artifact.node_pair_table)
    write_reports(out, artifact, batch)
    write_latex_tables(out, artifact)
    write_arrays(out, artifact)
    write_figures(out, artifact, batch)


def graph_summary(artifact, batch) -> dict:
    return {
        "pipeline": "graph_level_otg",
        "world": artifact.world_name,
        "graph": artifact.graph,
        "domains": list(artifact.domains),
        "selected_nodes": artifact.selected_nodes,
        "system_score": artifact.system_score,
        "domain_order": artifact.domain_order,
        "D_op_matrix": artifact.discrepancy_matrix.tolist(),
        "num_domain_pairs": len(artifact.comparisons),
        "world_spec": batch.metadata.get("world_spec", {}),
        "metadata": artifact.metadata,
    }


def _domain_pair_json(artifact) -> dict:
    out = {}
    for pair, comp in artifact.comparisons.items():
        key = f"{pair[0]}::{pair[1]}"
        out[key] = {
            "aggregate": comp.aggregate,
            "diagnostics": comp.diagnostics,
            "nodes": {
                node: {
                    "transport_value": float(res.transport.value),
                    "solver": res.transport.solver,
                    "status": res.transport.status,
                    "diagnostics": res.diagnostics,
                    "transport_metadata": res.transport.metadata,
                }
                for node, res in comp.node_results.items()
            },
        }
    return out


def write_domain_pair_matrix_csv(path: Path, artifact) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain"] + artifact.domain_order)
        for domain, row in zip(artifact.domain_order, artifact.discrepancy_matrix):
            writer.writerow([domain] + [float(x) for x in row])


def write_node_pair_metrics_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = ["domain_a", "domain_b", "node", "transport_value", "solver", "status"]
    rest = sorted(k for row in rows for k in row if k not in keys)
    fieldnames = keys + rest
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(out: Path, artifact, batch) -> None:
    spec = batch.metadata.get("world_spec", {})
    lines = [
        "# Operational Transport Geometry Graph-Level Report",
        "",
        f"World: `{artifact.world_name}`",
        f"Pipeline: `graph_level_otg`",
        f"Domains: `{', '.join(artifact.domain_order)}`",
        f"Selected nodes: `{', '.join(artifact.selected_nodes)}`",
        f"System score / mean D_op: `{artifact.system_score['value']:.6f}`",
        "",
        "## Proposal mapping",
        "",
        "| Proposal object | Implementation object |",
        "|---|---|",
        "| `D` | `GraphWorldBatch.domains` |",
        "| `P_d` | `DomainSpec.input_law_metadata` |",
        "| `mu_v^d` | `NodeLaw.samples` keyed by `(node, domain)` |",
        "| `phi_v^d` | `NodeLaw.projection_metadata` |",
        "| `mutilde_v^d` | `NodeLaw.aligned_samples` |",
        "| `r_v`, `rhat_v` | `NodeLaw.true_risk`, pairwise estimated risk |",
        "| `Pi_{v,adm}` | `InvarianceOutput.allowed` |",
        "| `W_{v,op}` | `NodePairResult.transport.value` |",
        "| `D_op(d,d')` | `SystemComparisonResult.aggregate.value` |",
        "",
        "## Controlled graph world",
        "",
        f"Mathematical definition: {spec.get('mathematical_definition', '')}",
        "",
        f"Expected behavior: {spec.get('expected_behavior', '')}",
        "",
        "## Domain-pair discrepancy matrix `D_op(d,d')`",
        "",
    ]
    lines.append("| domain | " + " | ".join(artifact.domain_order) + " |")
    lines.append("|---" + "|---:" * len(artifact.domain_order) + "|")
    for domain, row in zip(artifact.domain_order, artifact.discrepancy_matrix):
        lines.append("| " + domain + " | " + " | ".join(f"{float(x):.4f}" for x in row) + " |")

    lines.extend(["", "## Node-wise localization", "", "| Domain pair | Localizing node | D_op | Node values |", "|---|---|---:|---|"])
    for pair, comp in artifact.comparisons.items():
        vals = comp.aggregate.get("node_values", {})
        lines.append(f"| `{pair[0]} ↔ {pair[1]}` | `{comp.diagnostics.get('localizing_node')}` | {float(comp.aggregate.get('value',0.0)):.4f} | `{vals}` |")

    lines.extend(["", "## Assumption report", "", "```json", json.dumps(artifact.assumption_report, indent=2), "```"])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    interp = [
        "# Graph-Level Interpretation",
        "",
        "This run is graph-level: it compares multiple deployment domains across multiple internal nodes of a deployed DAG.",
        "",
        f"Lowest discrepancy pair: `{artifact.system_score['min_pair']}` = `{artifact.system_score['min_pair_value']:.6f}`.",
        f"Highest discrepancy pair: `{artifact.system_score['max_pair']}` = `{artifact.system_score['max_pair_value']:.6f}`.",
        "",
        "Inspect `node_pair_metrics.csv` to localize which internal node generated each domain-pair discrepancy.",
    ]
    (out / "interpretation.md").write_text("\n".join(interp), encoding="utf-8")


def write_latex_tables(out: Path, artifact) -> None:
    lines = [r"\begin{tabular}{l" + "r" * len(artifact.domain_order) + "}", r"\toprule"]
    lines.append("Domain & " + " & ".join(artifact.domain_order) + r" \\")
    lines.append(r"\midrule")
    for domain, row in zip(artifact.domain_order, artifact.discrepancy_matrix):
        lines.append(domain + " & " + " & ".join(f"{float(x):.4f}" for x in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "domain_pair_matrix_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_arrays(out: Path, artifact) -> None:
    for pair, comp in artifact.comparisons.items():
        pair_key = f"{pair[0]}__{pair[1]}"
        for node, res in comp.node_results.items():
            d = ensure_dir(out / "node_pairs" / pair_key / node)
            save_array(d / "cost_total.npy", res.cost.total)
            save_array(d / "transport_plan.npy", res.transport.plan)
            save_array(d / "admissibility_allowed.npy", res.invariance.allowed.astype(float))
            save_array(d / "invariance_soft_score.npy", res.invariance.soft_score)
            if res.transport.unmatched_a is not None:
                save_array(d / "unmatched_a.npy", res.transport.unmatched_a)
            if res.transport.unmatched_b is not None:
                save_array(d / "unmatched_b.npy", res.transport.unmatched_b)
            save_json(d / "diagnostics.json", res.diagnostics)
            save_json(d / "transport_metadata.json", res.transport.metadata)


def write_figures(out: Path, artifact, batch) -> None:
    fig_dir = ensure_dir(out / "figures")
    _heatmap(artifact.discrepancy_matrix, artifact.domain_order, artifact.domain_order, fig_dir / "D_op_domain_pair_heatmap.png", "System discrepancy D_op(d,d')")

    pair_labels = [f"{a}\n{b}" for (a, b) in artifact.comparisons]
    node_matrix = np.zeros((len(artifact.selected_nodes), len(pair_labels)), dtype=float)
    for j, comp in enumerate(artifact.comparisons.values()):
        for i, node in enumerate(artifact.selected_nodes):
            node_matrix[i, j] = float(comp.node_results[node].transport.value)
    _heatmap(node_matrix, pair_labels, artifact.selected_nodes, fig_dir / "node_by_domain_pair_heatmap.png", "Node-wise W_v,op by domain pair")

    # Risk distribution per node/domain.
    for node in artifact.selected_nodes:
        plt.figure(figsize=(7, 4))
        for domain in artifact.domain_order:
            law = batch.law(node, domain)
            plt.hist(law.true_risk, bins=16, alpha=0.45, label=domain)
        plt.xlabel("terminal-induced node risk")
        plt.ylabel("count")
        plt.title(f"Risk distributions: {node}")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_dir / f"risk_distributions_{node}.png", dpi=180)
        plt.close()

    # Representative harmless/harmful pair plans and admissibility masks.
    representatives = []
    for wanted in [("clear", "viewpoint_shift"), ("clear", "glare"), ("clear", "occlusion")]:
        if wanted in artifact.comparisons:
            representatives.append(wanted)
    for pair in representatives:
        comp = artifact.comparisons[pair]
        local_node = comp.diagnostics.get("localizing_node") or artifact.selected_nodes[0]
        res = comp.node_results[local_node]
        prefix = f"{pair[0]}__{pair[1]}__{local_node}"
        _matrix(res.invariance.allowed.astype(float), fig_dir / f"admissibility_{prefix}.png", f"Admissibility: {pair[0]} vs {pair[1]} @ {local_node}")
        _matrix(res.transport.plan, fig_dir / f"transport_plan_{prefix}.png", f"Transport plan: {pair[0]} vs {pair[1]} @ {local_node}")

    # Unmatched dangerous mass by pair/node.
    labels, vals = [], []
    for pair, comp in artifact.comparisons.items():
        for node, res in comp.node_results.items():
            v = float(res.diagnostics.get("dangerous_unmatched_mass_b", 0.0))
            if v > 0:
                labels.append(f"{pair[0]}-{pair[1]}\n{node}")
                vals.append(v)
    if vals:
        plt.figure(figsize=(max(6, 0.45 * len(vals)), 4))
        plt.bar(range(len(vals)), vals)
        plt.xticks(range(len(vals)), labels, rotation=60, ha="right", fontsize=7)
        plt.ylabel("dangerous unmatched mass")
        plt.title("Unbalanced dangerous unmatched target mass")
        plt.tight_layout()
        plt.savefig(fig_dir / "unmatched_dangerous_mass.png", dpi=180)
        plt.close()


def _heatmap(M: np.ndarray, xlabels: list[str], ylabels: list[str], path: Path, title: str) -> None:
    plt.figure(figsize=(max(5, 0.65 * len(xlabels)), max(4, 0.45 * len(ylabels))))
    plt.imshow(M, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(xlabels)), xlabels, rotation=45, ha="right")
    plt.yticks(range(len(ylabels)), ylabels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _matrix(M: np.ndarray, path: Path, title: str) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(M, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
