from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from benchmarks.shared import default_features, feature_cost, failure_labels, label_risk, node_from_batch


@dataclass(frozen=True)
class MethodProvenance:
    paper: str
    implementation_kind: str
    implementation_source: str


class PaperMethodBase:
    paper_name: str = ""
    implementation_kind: str = "reimplemented"
    implementation_source: str = "paper-faithful-reimplementation"

    @classmethod
    def provenance(cls) -> MethodProvenance:
        return MethodProvenance(
            paper=cls.paper_name or cls.__name__,
            implementation_kind=cls.implementation_kind,
            implementation_source=cls.implementation_source,
        )


def fit_pot_transport(cls, Xa, ya, Xb, yb):
    import ot

    obj = cls()
    obj.fit(Xa, ya, Xb, yb)
    return obj


def transform_or_self(obj, X):
    if hasattr(obj, "transform"):
        try:
            return np.asarray(obj.transform(X), dtype=float)
        except Exception:
            pass
    return np.asarray(X, dtype=float)


def learn_linear_scores(Xa, Xb, ya, group_weights=None, l2=1e-2, steps=200, lr=0.05):
    X = np.vstack([Xa, Xb]).astype(float)
    y = np.concatenate([ya, ya[: len(Xb)] if len(ya) >= len(Xb) else np.resize(ya, len(Xb))]).astype(float)
    if group_weights is None:
        w = np.ones(len(X), dtype=float)
    else:
        w = np.asarray(group_weights, dtype=float)
    Xn = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    beta = np.zeros(Xn.shape[1], dtype=float)
    bias = 0.0
    for _ in range(steps):
        logits = Xn @ beta + bias
        pred = expit(logits)
        err = pred - y
        grad_b = (Xn.T @ (w * err)) / len(Xn) + l2 * beta
        grad_bias = float(np.mean(w * err))
        beta -= lr * grad_b
        bias -= lr * grad_bias
    scores = expit(Xn @ beta + bias)
    return scores[: len(Xa)], scores[len(Xa):], {"beta_norm": float(np.linalg.norm(beta)), "bias": float(bias)}


def irm_scores(Xa, Xb, ya, yb, penalty=1.0):
    X = np.vstack([Xa, Xb]).astype(float)
    y = np.concatenate([ya, yb]).astype(float)
    env = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))]).astype(int)
    Xn = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    beta = np.zeros(Xn.shape[1], dtype=float)
    bias = 0.0
    lr = 0.05
    for _ in range(250):
        logits = Xn @ beta + bias
        pred = expit(logits)
        err = pred - y
        base_grad = Xn.T @ err / len(Xn)
        base_bias = float(np.mean(err))
        env_grads = []
        for e in [0, 1]:
            idx = env == e
            if np.any(idx):
                pe = pred[idx]
                ye = y[idx]
                xe = Xn[idx]
                ee = pe - ye
                env_grads.append(np.r_[xe.T @ ee / len(xe), float(np.mean(ee))])
        if len(env_grads) == 2:
            env_grads = np.asarray(env_grads)
            var_pen = env_grads[0] - env_grads[1]
            grad_b = base_grad + penalty * var_pen[:-1]
            grad_bias = base_bias + penalty * float(var_pen[-1])
        else:
            grad_b = base_grad
            grad_bias = base_bias
        beta -= lr * grad_b
        bias -= lr * grad_bias
    scores = expit(Xn @ beta + bias)
    return scores[: len(Xa)], scores[len(Xa):], {"beta_norm": float(np.linalg.norm(beta)), "bias": float(bias)}


def group_dro_scores(Xa, Xb, ya, yb, steps=250, l2=1e-2):
    X = np.vstack([Xa, Xb]).astype(float)
    y = np.concatenate([ya, yb]).astype(float)
    env = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))]).astype(int)
    Xn = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    beta = np.zeros(Xn.shape[1], dtype=float)
    bias = 0.0
    lr = 0.05
    group_w = np.ones(2, dtype=float) / 2
    for _ in range(steps):
        logits = Xn @ beta + bias
        pred = expit(logits)
        losses = []
        grads = []
        for e in [0, 1]:
            idx = env == e
            if not np.any(idx):
                losses.append(0.0)
                grads.append((np.zeros_like(beta), 0.0))
                continue
            pe = pred[idx]
            ye = y[idx]
            xe = Xn[idx]
            loss = -np.mean(ye * np.log(pe + 1e-8) + (1 - ye) * np.log(1 - pe + 1e-8))
            err = pe - ye
            grad = xe.T @ err / len(xe)
            losses.append(float(loss))
            grads.append((grad, float(np.mean(err))))
        worst = int(np.argmax(losses))
        group_w = 0.9 * group_w
        group_w[worst] += 0.1
        group_w = group_w / group_w.sum()
        grad_b = sum(group_w[i] * grads[i][0] for i in [0, 1]) + l2 * beta
        grad_bias = sum(group_w[i] * grads[i][1] for i in [0, 1])
        beta -= lr * grad_b
        bias -= lr * grad_bias
    scores = expit(Xn @ beta + bias)
    return scores[: len(Xa)], scores[len(Xa):], {"beta_norm": float(np.linalg.norm(beta)), "bias": float(bias), "group_weights": group_w.tolist()}


def spo_scores(Xa, Xb, ya, yb, steps=220):
    X = np.vstack([Xa, Xb]).astype(float)
    y = np.concatenate([ya, yb]).astype(float)
    Xn = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    beta = np.zeros(Xn.shape[1], dtype=float)
    bias = 0.0
    lr = 0.05
    for _ in range(steps):
        scores = Xn @ beta + bias
        pred = expit(scores)
        pos = y >= 0.5
        neg = ~pos
        if np.any(pos) and np.any(neg):
            margin = scores[pos][:, None] - scores[neg][None, :]
            loss_grad = -expit(-margin)
            grad_scores = np.zeros_like(scores)
            grad_scores[pos] += np.mean(loss_grad, axis=1)
            grad_scores[neg] -= np.mean(loss_grad, axis=0)
        else:
            grad_scores = pred - y
        beta -= lr * (Xn.T @ grad_scores / len(Xn))
        bias -= lr * float(np.mean(grad_scores))
    out = expit(Xn @ beta + bias)
    return out[: len(Xa)], out[len(Xa):], {"beta_norm": float(np.linalg.norm(beta)), "bias": float(bias)}


def normalize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd


def project_spd(M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    M = 0.5 * (np.asarray(M, dtype=float) + np.asarray(M, dtype=float).T)
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, eps)
    return (eigvecs * eigvals) @ eigvecs.T


def riemannian_metric_learning(Xa: np.ndarray, Xb: np.ndarray, steps: int = 12, lr: float = 0.2):
    Xa = normalize_rows(Xa)
    Xb = normalize_rows(Xb)
    d = Xa.shape[1]
    M = np.eye(d, dtype=float)
    a = np.ones(len(Xa), dtype=float) / max(len(Xa), 1)
    b = np.ones(len(Xb), dtype=float) / max(len(Xb), 1)
    last_plan = np.outer(a, b)
    last_cost = None
    for _ in range(steps):
        diff = Xa[:, None, :] - Xb[None, :, :]
        cost = np.einsum("ijk,kl,ijl->ij", diff, M, diff)
        try:
            import ot
            plan = ot.sinkhorn(a, b, cost, reg=0.05, numItermax=200)
        except Exception:
            plan = np.outer(a, b)
        moment = np.einsum("ij,ijk,ijl->kl", plan, diff, diff)
        trace = float(np.trace(moment)) / max(d, 1)
        grad = moment - trace * np.eye(d)
        M = project_spd(M - lr * grad)
        M = M / max(np.trace(M) / max(d, 1), 1e-8)
        last_plan = plan
        last_cost = cost
    return M, last_plan, last_cost


def metric_features(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    M = project_spd(M)
    eigvals, eigvecs = np.linalg.eigh(M)
    basis = eigvecs * np.sqrt(np.maximum(eigvals, 1e-8))
    return X @ basis
