from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from otg.core.schema import NodeBatch, RiskOutput


@dataclass
class RiskDataset:
    x: np.ndarray
    risk: np.ndarray
    failure: np.ndarray
    domain: np.ndarray

    @classmethod
    def from_node(cls, node: NodeBatch) -> "RiskDataset":
        x = np.vstack([node.z_a, node.z_b])
        risk = np.r_[node.true_risk_a, node.true_risk_b]
        failure = np.r_[node.failure_a, node.failure_b].astype(int)
        domain = np.r_[np.zeros(len(node.z_a), dtype=int), np.ones(len(node.z_b), dtype=int)]
        return cls(x=x, risk=risk, failure=failure, domain=domain)


class RiskModel(ABC):
    mode: str

    def __init__(self, cfg: dict, seed_bank):
        self.cfg = cfg
        self.seed_bank = seed_bank

    @abstractmethod
    def estimate(self, node: NodeBatch) -> RiskOutput:
        raise NotImplementedError

    def rng(self, suffix: str):
        return self.seed_bank.rng(f"risk:{self.mode}:{suffix}")

    def split_dataset(self, dataset: RiskDataset, suffix: str) -> tuple[np.ndarray, np.ndarray]:
        frac = float(self.cfg.get("risk", {}).get("train_fraction", 0.70))
        n = len(dataset.x)
        rng = self.rng(f"split:{suffix}")
        idx = rng.permutation(n)
        n_train = max(2, min(n - 1, int(frac * n)))
        return idx[:n_train], idx[n_train:]


def risk_mae(pred_a: np.ndarray, pred_b: np.ndarray, true_a: np.ndarray, true_b: np.ndarray) -> float:
    return float((np.mean(np.abs(pred_a - true_a)) + np.mean(np.abs(pred_b - true_b))) / 2)


def calibration_error(pred: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    count = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (pred >= lo) & (pred <= hi if hi == 1.0 else pred < hi)
        if not np.any(mask):
            continue
        total += float(np.mean(mask)) * abs(float(np.mean(pred[mask])) - float(np.mean(target[mask])))
        count += 1
    return float(total)


def failure_accuracy(pred: np.ndarray, failure: np.ndarray, threshold: float = 0.5) -> float:
    return float(np.mean((pred >= threshold).astype(int) == failure.astype(int)))


def make_risk_output(
    mode: str,
    node: NodeBatch,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metadata: dict | None = None,
    train_pred: np.ndarray | None = None,
    train_true: np.ndarray | None = None,
) -> RiskOutput:
    pred_a = np.clip(np.asarray(pred_a, dtype=float), 0.0, 1.0)
    pred_b = np.clip(np.asarray(pred_b, dtype=float), 0.0, 1.0)
    all_pred = np.r_[pred_a, pred_b]
    all_true = np.r_[node.true_risk_a, node.true_risk_b]
    all_failure = np.r_[node.failure_a, node.failure_b].astype(int)

    out = RiskOutput(
        used_risk_a=pred_a,
        used_risk_b=pred_b,
        risk_error_mae=risk_mae(pred_a, pred_b, node.true_risk_a, node.true_risk_b),
        mode=mode,
        metadata=metadata or {},
        train_error_mae=None if train_pred is None or train_true is None else float(np.mean(np.abs(train_pred - train_true))),
        calibration_error=calibration_error(all_pred, all_true),
        failure_accuracy=failure_accuracy(all_pred, all_failure),
    )
    return out
