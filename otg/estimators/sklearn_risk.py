from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class SklearnRiskRegressor:
    def __init__(self, kind: str = "ridge", seed: int = 0, **kwargs):
        if kind == "ridge":
            model = Ridge(alpha=float(kwargs.get("alpha", 1e-2)))
        elif kind == "random_forest":
            model = RandomForestRegressor(
                n_estimators=int(kwargs.get("n_estimators", 100)),
                max_depth=kwargs.get("max_depth", None),
                random_state=seed,
            )
        else:
            raise ValueError(f"Unknown sklearn regressor kind: {kind}")
        self.model = make_pipeline(StandardScaler(), model) if kind == "ridge" else model

    def fit(self, x: np.ndarray, risk: np.ndarray) -> "SklearnRiskRegressor":
        self.model.fit(x, risk)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict(x), 0.0, 1.0)


class SklearnFailureClassifier:
    def __init__(self, kind: str = "logistic", seed: int = 0, **kwargs):
        if kind == "logistic":
            model = LogisticRegression(max_iter=int(kwargs.get("max_iter", 500)))
            self.model = make_pipeline(StandardScaler(), model)
        elif kind == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=int(kwargs.get("n_estimators", 100)),
                max_depth=kwargs.get("max_depth", None),
                random_state=seed,
            )
        else:
            raise ValueError(f"Unknown sklearn classifier kind: {kind}")

    def fit(self, x: np.ndarray, failure: np.ndarray) -> "SklearnFailureClassifier":
        self.model.fit(x, failure.astype(int))
        return self

    def predict_risk(self, x: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)[:, 1]
        return self.model.predict_proba(x)[:, 1]
