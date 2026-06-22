from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from benchmarks.metrics import normalize_cost


def node_from_batch(batch):
    return batch.nodes["repr"]


def default_features(node) -> tuple[np.ndarray, np.ndarray]:
    Xa = np.c_[node.operational_coords_a, node.nuisance_coords_a, node.descriptor_coords_a]
    Xb = np.c_[node.operational_coords_b, node.nuisance_coords_b, node.descriptor_coords_b]
    return Xa, Xb


def feature_cost(Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
    return normalize_cost(cdist(Xa, Xb))


def risk_gap_cost(risk_a: np.ndarray, risk_b: np.ndarray) -> np.ndarray:
    return normalize_cost(np.abs(np.asarray(risk_a, dtype=float)[:, None] - np.asarray(risk_b, dtype=float)[None, :]))


def label_risk(node) -> tuple[np.ndarray, np.ndarray]:
    return node.true_risk_a.astype(float), node.true_risk_b.astype(float)


def failure_labels(node) -> tuple[np.ndarray, np.ndarray]:
    return node.failure_a.astype(int), node.failure_b.astype(int)
