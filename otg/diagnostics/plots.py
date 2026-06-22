from __future__ import annotations

from pathlib import Path
import shutil
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_run_outputs(out_dir: str | Path, artifact, batch) -> None:
    out = Path(out_dir)
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    for node_name, res in artifact.node_results.items():
        node = batch.nodes[node_name]
        plot_node_pointcloud(node, fig / f"{node_name}_pointcloud.png", batch.name)
        plot_risk_hist(node, fig / f"{node_name}_risk_hist.png", batch.name)
        plot_risk_calibration(node, fig / f"{node_name}_risk_calibration.png", batch.name)
        plot_matrix(res.cost.total, fig / f"{node_name}_cost_total.png", f"{node_name}: total operational cost")
        plot_matrix(res.cost.geometry, fig / f"{node_name}_cost_geometry.png", f"{node_name}: geometric cost")
        plot_matrix(res.cost.risk, fig / f"{node_name}_cost_risk.png", f"{node_name}: risk cost")
        plot_matrix(res.cost.invariance, fig / f"{node_name}_cost_invariance.png", f"{node_name}: invariance cost")
        plot_matrix(res.invariance.allowed.astype(float), fig / f"{node_name}_admissibility_mask.png", f"{node_name}: admissibility mask")
        plot_matrix(res.invariance.soft_score, fig / f"{node_name}_invariance_soft_score.png", f"{node_name}: invariance soft score")
        transport_plan_path = fig / f"{node_name}_transport_plan.png"
        plot_matrix(res.transport.plan, transport_plan_path, f"{node_name}: transport plan")
        if node_name == "repr":
            shutil.copyfile(transport_plan_path, fig / "repr_transport_plan.png")
        plot_diagnostics_bar(res.diagnostics, fig / f"{node_name}_diagnostics_bar.png", node_name)
        plot_unmatched(res, fig / f"{node_name}_unmatched_mass.png", node_name)
        if node.z_a.shape[1] >= 2:
            plot_transport_arrows(node, res.transport.plan, fig / f"{node_name}_transport_arrows.png", batch.name)


def plot_node_pointcloud(node, path: Path, world_name: str) -> None:
    if node.z_a.shape[1] < 2:
        return
    plt.figure(figsize=(6, 5))
    plt.scatter(node.z_a[:, 0], node.z_a[:, 1], s=14, alpha=0.7, label="domain A")
    plt.scatter(node.z_b[:, 0], node.z_b[:, 1], s=14, alpha=0.7, label="domain B")
    plt.xlabel("operational coordinate")
    plt.ylabel("nuisance coordinate 1")
    plt.title(f"{world_name}: {node.node} point cloud")
    plt.legend()
    _savefig(path)


def plot_transport_arrows(node, plan: np.ndarray, path: Path, world_name: str, max_arrows: int = 60) -> None:
    if node.z_a.shape[1] < 2 or node.z_b.shape[1] < 2:
        return
    flat = plan.reshape(-1)
    if flat.size == 0 or np.all(flat <= 0):
        return
    idx = np.argsort(flat)[-min(max_arrows, flat.size):]
    n_b = plan.shape[1]
    plt.figure(figsize=(6, 5))
    plt.scatter(node.z_a[:, 0], node.z_a[:, 1], s=10, alpha=0.5, label="A")
    plt.scatter(node.z_b[:, 0], node.z_b[:, 1], s=10, alpha=0.5, label="B")
    max_mass = float(flat[idx].max()) if idx.size else 1.0
    for k in idx:
        i, j = divmod(int(k), n_b)
        if flat[k] <= 0:
            continue
        lw = 0.2 + 1.5 * float(flat[k] / max_mass)
        plt.plot([node.z_a[i, 0], node.z_b[j, 0]], [node.z_a[i, 1], node.z_b[j, 1]], alpha=0.25, linewidth=lw)
    plt.xlabel("operational coordinate")
    plt.ylabel("nuisance coordinate 1")
    plt.title(f"{world_name}: strongest transport links")
    plt.legend()
    _savefig(path)


def plot_risk_hist(node, path: Path, world_name: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(node.true_risk_a, bins=20, alpha=0.6, label="A true")
    plt.hist(node.true_risk_b, bins=20, alpha=0.6, label="B true")
    plt.hist(node.used_risk_a, bins=20, alpha=0.35, label="A used")
    plt.hist(node.used_risk_b, bins=20, alpha=0.35, label="B used")
    plt.xlabel("risk")
    plt.ylabel("count")
    plt.title(f"{world_name}: risk distributions")
    plt.legend()
    _savefig(path)


def plot_risk_calibration(node, path: Path, world_name: str) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(node.true_risk_a, node.used_risk_a, s=12, alpha=0.7, label="A")
    plt.scatter(node.true_risk_b, node.used_risk_b, s=12, alpha=0.7, label="B")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    plt.xlabel("true risk")
    plt.ylabel("used risk")
    plt.title(f"{world_name}: risk calibration")
    plt.legend()
    _savefig(path)


def plot_matrix(M: np.ndarray, path: Path, title: str) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(M, aspect="auto")
    plt.colorbar()
    plt.title(title)
    _savefig(path)


def plot_diagnostics_bar(diag: dict, path: Path, node_name: str) -> None:
    keys = [
        "risk_estimation_mae",
        "false_collapse_mass",
        "false_separation_score",
        "allowed_pair_fraction",
        "transport_forbidden_mass",
        "unmatched_mass_b",
        "dangerous_unmatched_mass_b",
    ]
    vals = [float(diag.get(k, 0.0)) for k in keys]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(keys)), vals)
    plt.xticks(range(len(keys)), keys, rotation=45, ha="right")
    plt.ylabel("value")
    plt.title(f"{node_name}: key diagnostics")
    _savefig(path)


def plot_unmatched(res, path: Path, node_name: str) -> None:
    diag = res.diagnostics
    keys = [
        "finite_sample_unmatched_a",
        "finite_sample_unmatched_b",
        "operational_nonsubstitutable_unmatched_a",
        "operational_nonsubstitutable_unmatched_b",
        "dangerous_unmatched_mass_a",
        "dangerous_unmatched_mass_b",
    ]
    vals = [float(diag.get(k, 0.0)) for k in keys]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(keys)), vals)
    plt.xticks(range(len(keys)), keys, rotation=45, ha="right")
    plt.ylabel("mass")
    plt.title(f"{node_name}: unmatched mass decomposition")
    _savefig(path)
