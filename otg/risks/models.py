from __future__ import annotations

import numpy as np

from otg.core.schema import NodeBatch, RiskOutput
from otg.risks.base import RiskModel, RiskDataset, make_risk_output
from otg.risks.registry import register_risk
from otg.estimators.sklearn_risk import SklearnRiskRegressor, SklearnFailureClassifier


@register_risk("true")
class TrueRisk(RiskModel):
    mode = "true"

    def estimate(self, node: NodeBatch) -> RiskOutput:
        return make_risk_output(self.mode, node, node.true_risk_a.copy(), node.true_risk_b.copy(), {"source": "ground_truth"})


@register_risk("noisy")
class NoisyRisk(RiskModel):
    mode = "noisy"

    def estimate(self, node: NodeBatch) -> RiskOutput:
        rng = self.rng(node.node)
        sigma = float(self.cfg.get("risk", {}).get("noise_std", 0.1))
        a = node.true_risk_a + rng.normal(0.0, sigma, len(node.true_risk_a))
        b = node.true_risk_b + rng.normal(0.0, sigma, len(node.true_risk_b))
        return make_risk_output(self.mode, node, a, b, {"noise_std": sigma})


@register_risk("rollout")
class RolloutRisk(RiskModel):
    mode = "rollout"

    def estimate(self, node: NodeBatch) -> RiskOutput:
        rng = self.rng(node.node)
        mc = int(self.cfg.get("runtime_values", {}).get("mc_rollouts", self.cfg.get("risk", {}).get("mc_rollouts", 64)))
        threshold = float(self.cfg.get("world", {}).get("failure_threshold", 0.75))
        sigma = float(self.cfg.get("uncertainty", {}).get("rollout_terminal_noise", self.cfg.get("uncertainty", {}).get("terminal_noise", 0.12)))
        ta = node.terminal_a.reshape(len(node.terminal_a), -1)
        tb = node.terminal_b.reshape(len(node.terminal_b), -1)
        if ta.shape[1] >= 4:
            ya = ta[:, None, :] + rng.normal(0.0, sigma, (len(ta), mc, ta.shape[1]))
            yb = tb[:, None, :] + rng.normal(0.0, sigma, (len(tb), mc, tb.shape[1]))
            def terminal_failure_prob(Y):
                adequacy = Y[..., 0]
                coverage = Y[..., 1]
                glare = Y[..., 2]
                uncertainty = Y[..., 3]
                score = 3.2 * (0.58 - adequacy) + 2.5 * (0.55 - coverage) + 1.7 * glare + 1.2 * uncertainty
                return 1.0 / (1.0 + np.exp(-score))
            a = terminal_failure_prob(ya).mean(axis=1)
            b = terminal_failure_prob(yb).mean(axis=1)
        else:
            ya = ta[:, :1] + rng.normal(0.0, sigma, (len(ta), mc))
            yb = tb[:, :1] + rng.normal(0.0, sigma, (len(tb), mc))
            a = (ya > threshold).mean(axis=1)
            b = (yb > threshold).mean(axis=1)
        out = make_risk_output(self.mode, node, a, b, {"mc_rollouts": mc, "rollout_terminal_noise": sigma})
        out.uncertainty_a = np.sqrt(np.maximum(a * (1.0 - a), 0.0) / max(mc, 1))
        out.uncertainty_b = np.sqrt(np.maximum(b * (1.0 - b), 0.0) / max(mc, 1))
        return out


@register_risk("misspecified")
class MisspecifiedRisk(RiskModel):
    mode = "misspecified"

    def estimate(self, node: NodeBatch) -> RiskOutput:
        # Deliberately wrong: ignores terminal behavior and operational coordinate, uses nuisance magnitude.
        na = np.linalg.norm(node.nuisance_coords_a, axis=1) if node.nuisance_coords_a.size else np.zeros(len(node.z_a))
        nb = np.linalg.norm(node.nuisance_coords_b, axis=1) if node.nuisance_coords_b.size else np.zeros(len(node.z_b))
        center = np.median(np.r_[na, nb])
        scale = np.std(np.r_[na, nb]) + 1e-8
        a = 1.0 / (1.0 + np.exp(-3.0 * (na - center) / scale))
        b = 1.0 / (1.0 + np.exp(-3.0 * (nb - center) / scale))
        return make_risk_output(self.mode, node, a, b, {"misspecification": "nuisance_norm_proxy"})


class _LearnedRiskBase(RiskModel):
    estimator_name: str = "base"

    def features(self, node: NodeBatch) -> np.ndarray:
        feature_mode = self.cfg.get("risk", {}).get("features", "state")
        if feature_mode == "state":
            return np.vstack([node.z_a, node.z_b])
        if feature_mode == "operational":
            return np.vstack([node.operational_coords_a, node.operational_coords_b])
        if feature_mode == "nuisance":
            return np.vstack([node.nuisance_coords_a, node.nuisance_coords_b])
        if feature_mode == "descriptor":
            return np.vstack([node.descriptor_coords_a, node.descriptor_coords_b])
        if feature_mode == "state_descriptor":
            return np.vstack([
                np.c_[node.z_a, node.descriptor_coords_a],
                np.c_[node.z_b, node.descriptor_coords_b],
            ])
        raise ValueError(f"Unknown risk feature mode: {feature_mode}")

    def split_features(self, node: NodeBatch, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return X[: len(node.z_a)], X[len(node.z_a):]


@register_risk("learned_regression")
class LearnedRegressionRisk(_LearnedRiskBase):
    mode = "learned_regression"

    def estimate(self, node: NodeBatch) -> RiskOutput:
        rng = self.rng(node.node)
        X = self.features(node)
        data = RiskDataset.from_node(node)
        data = RiskDataset(X, data.risk, data.failure, data.domain)
        train_idx, test_idx = self.split_dataset(data, node.node)
        kind = self.cfg.get("risk", {}).get("sklearn_regressor", "ridge")
        model = SklearnRiskRegressor(kind=kind, seed=int(rng.integers(0, 2**31 - 1)), alpha=float(self.cfg.get("risk", {}).get("ridge_alpha", 1e-2)))
        model.fit(data.x[train_idx], data.risk[train_idx])
        pred_all = model.predict(data.x)
        Xa, Xb = self.split_features(node, X)
        pa = model.predict(Xa)
        pb = model.predict(Xb)
        return make_risk_output(
            self.mode,
            node,
            pa,
            pb,
            {"estimator": kind, "features": self.cfg.get("risk", {}).get("features", "state"), "train_size": int(len(train_idx)), "test_size": int(len(test_idx))},
            train_pred=pred_all[train_idx],
            train_true=data.risk[train_idx],
        )


@register_risk("learned_classifier")
class LearnedClassifierRisk(_LearnedRiskBase):
    mode = "learned_classifier"

    def estimate(self, node: NodeBatch) -> RiskOutput:
        rng = self.rng(node.node)
        X = self.features(node)
        data0 = RiskDataset.from_node(node)
        data = RiskDataset(X, data0.risk, data0.failure, data0.domain)
        train_idx, test_idx = self.split_dataset(data, node.node)
        kind = self.cfg.get("risk", {}).get("sklearn_classifier", "logistic")

        if len(np.unique(data.failure[train_idx])) < 2:
            base = float(np.mean(data.failure[train_idx]))
            pa = np.full(len(node.z_a), base)
            pb = np.full(len(node.z_b), base)
            meta = {"estimator": "constant_classifier", "reason": "single_class_train_split"}
            return make_risk_output(self.mode, node, pa, pb, meta)

        model = SklearnFailureClassifier(kind=kind, seed=int(rng.integers(0, 2**31 - 1)), max_iter=int(self.cfg.get("risk", {}).get("logistic_max_iter", 500)))
        model.fit(data.x[train_idx], data.failure[train_idx])
        pred_all = model.predict_risk(data.x)
        Xa, Xb = self.split_features(node, X)
        pa = model.predict_risk(Xa)
        pb = model.predict_risk(Xb)
        return make_risk_output(
            self.mode,
            node,
            pa,
            pb,
            {"estimator": kind, "features": self.cfg.get("risk", {}).get("features", "state"), "train_size": int(len(train_idx)), "test_size": int(len(test_idx))},
            train_pred=pred_all[train_idx],
            train_true=data.risk[train_idx],
        )


@register_risk("learned_mlp")
class LearnedMLPRisk(_LearnedRiskBase):
    mode = "learned_mlp"

    def estimate(self, node: NodeBatch) -> RiskOutput:
        rng = self.rng(node.node)
        X = self.features(node)
        data0 = RiskDataset.from_node(node)
        data = RiskDataset(X, data0.risk, data0.failure, data0.domain)
        train_idx, test_idx = self.split_dataset(data, node.node)

        backend = self.cfg.get("risk", {}).get("mlp_backend", "sklearn")
        hidden = tuple(int(x) for x in self.cfg.get("risk", {}).get("mlp_hidden", [64, 64]))
        seed = int(rng.integers(0, 2**31 - 1))

        if backend == "torch":
            try:
                from otg.estimators.torch_risk import TorchRiskMLP
                model = TorchRiskMLP(
                    input_dim=data.x.shape[1],
                    hidden=hidden,
                    epochs=int(self.cfg.get("risk", {}).get("torch_epochs", 250)),
                    lr=float(self.cfg.get("risk", {}).get("torch_lr", 1e-3)),
                    seed=seed,
                    task="regression",
                )
                model.fit(data.x[train_idx], data.risk[train_idx])
                pred_all = model.predict(data.x)
            except Exception as exc:
                # Fallback to sklearn MLP to keep CPU-only runs reliable.
                from sklearn.neural_network import MLPRegressor
                model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=int(self.cfg.get("risk", {}).get("mlp_max_iter", 400)), random_state=seed)
                model.fit(data.x[train_idx], data.risk[train_idx])
                pred_all = np.clip(model.predict(data.x), 0, 1)
                backend = f"sklearn_fallback_after_torch_error:{type(exc).__name__}"
        else:
            from sklearn.neural_network import MLPRegressor
            model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=int(self.cfg.get("risk", {}).get("mlp_max_iter", 400)), random_state=seed)
            model.fit(data.x[train_idx], data.risk[train_idx])
            pred_all = np.clip(model.predict(data.x), 0, 1)

        pa = pred_all[: len(node.z_a)]
        pb = pred_all[len(node.z_a):]
        return make_risk_output(
            self.mode,
            node,
            pa,
            pb,
            {"estimator": "mlp", "backend": backend, "features": self.cfg.get("risk", {}).get("features", "state"), "train_size": int(len(train_idx)), "test_size": int(len(test_idx))},
            train_pred=pred_all[train_idx],
            train_true=data.risk[train_idx],
        )
