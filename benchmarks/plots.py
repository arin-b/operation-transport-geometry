from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.metrics import PRIMARY_CLAIMS


def write_benchmark_figures(out_dir: str | Path, results: list[dict], aggregate: dict) -> list[Path]:
    out = Path(out_dir)
    figures_dir = out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    methods = list(aggregate.get("diagnostic_ranking", [])) or sorted({item["method"] for item in results})
    written: list[Path] = []
    written.append(_plot_primary_claim_table(figures_dir, methods, aggregate, plt))
    written.append(_plot_claim_distribution(figures_dir, results, methods, plt, "dangerous_unmatched_exposed"))
    written.extend(_plot_mean_domain_matrices(figures_dir, results, methods, plt, "D_op"))
    written.extend(_plot_mean_domain_matrices(figures_dir, results, methods, plt, "dangerous_unmatched"))
    written.append(_plot_method_claim_heatmap(figures_dir, methods, aggregate, plt))
    plt.close("all")
    return written


def _claim_mean(aggregate: dict, method: str, claim: str) -> float:
    return float(aggregate.get("method_stats", {}).get(method, {}).get("claim_scores", {}).get(claim, {}).get("mean", 0.0))


def _plot_primary_claim_table(figures_dir: Path, methods: list[str], aggregate: dict, plt) -> Path:
    claims = ["primary_claim_average_diagnostic"] + PRIMARY_CLAIMS
    M = np.asarray([[_claim_mean(aggregate, method, claim) for claim in claims] for method in methods], dtype=float)
    fig, ax = plt.subplots(figsize=(1.15 * len(claims) + 4, max(4, 0.55 * len(methods))))
    im = ax.imshow(M, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(claims)), [c.replace("_", "\n") for c in claims], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(methods)), [m.upper() for m in methods])
    ax.set_title("Claim-specific benchmark scores")
    fig.colorbar(im, ax=ax, label="score")
    fig.tight_layout()
    path = figures_dir / "claim_scores_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    return path


def _plot_method_claim_heatmap(figures_dir: Path, methods: list[str], aggregate: dict, plt) -> Path:
    claims = PRIMARY_CLAIMS
    M = np.asarray([[_claim_mean(aggregate, method, claim) for method in methods] for claim in claims], dtype=float)
    fig, ax = plt.subplots(figsize=(1.1 * len(methods) + 3, 5))
    im = ax.imshow(M, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(methods)), [m.upper() for m in methods], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(claims)), [c.replace("_", " ") for c in claims])
    ax.set_title("Primary claims by method")
    fig.colorbar(im, ax=ax, label="score")
    fig.tight_layout()
    path = figures_dir / "method_claim_matrix.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    return path


def _plot_claim_distribution(figures_dir: Path, results: list[dict], methods: list[str], plt, claim: str) -> Path:
    data, labels = [], []
    for method in methods:
        vals = [float(item.get("claim_scores", {}).get(claim, 0.0)) for item in results if item["method"] == method]
        if vals:
            data.append(vals)
            labels.append(method.upper())
    fig, ax = plt.subplots(figsize=(10, 5))
    if data:
        ax.boxplot(data, tick_labels=labels, showmeans=True, meanline=True)
    ax.set_ylabel(claim.replace("_", " "))
    ax.set_title(f"Claim distribution: {claim}")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = figures_dir / f"{claim}_distribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    return path


def _mean_matrix(results: list[dict], method: str, matrix_name: str) -> tuple[np.ndarray, list[str]] | None:
    mats = []
    order = None
    for item in results:
        if item["method"] == method and matrix_name in item.get("domain_matrices", {}):
            mats.append(np.asarray(item["domain_matrices"][matrix_name], dtype=float))
            order = item.get("domain_order")
    if not mats or order is None:
        return None
    return np.mean(np.stack(mats, axis=0), axis=0), list(order)


def _plot_mean_domain_matrices(figures_dir: Path, results: list[dict], methods: list[str], plt, matrix_name: str) -> list[Path]:
    written: list[Path] = []
    for method in methods:
        got = _mean_matrix(results, method, matrix_name)
        if got is None:
            continue
        M, order = got
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        im = ax.imshow(M, aspect="equal")
        ax.set_xticks(np.arange(len(order)), order, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(order)), order)
        ax.set_title(f"{method.upper()} mean {matrix_name} matrix")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        path = figures_dir / f"{method}_{matrix_name}_matrix.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        written.append(path)
    return written
