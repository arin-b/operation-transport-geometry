from __future__ import annotations

import numpy as np


class TorchUnavailableError(RuntimeError):
    pass


class TorchRiskMLP:
    def __init__(
        self,
        input_dim: int,
        hidden: tuple[int, ...] = (64, 64),
        epochs: int = 250,
        lr: float = 1e-3,
        seed: int = 0,
        task: str = "regression",
    ):
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:
            raise TorchUnavailableError("PyTorch is not installed. Install with `pip install -e .[torch]`.") from exc

        self.torch = torch
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.task = task
        torch.manual_seed(int(seed))

        layers = []
        last = input_dim
        for h in hidden:
            layers.append(nn.Linear(last, int(h)))
            layers.append(nn.ReLU())
            last = int(h)
        layers.append(nn.Linear(last, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchRiskMLP":
        torch = self.torch
        X = torch.tensor(x, dtype=torch.float32)
        Y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.BCELoss() if self.task == "classification" else torch.nn.MSELoss()
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = self.model(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            opt.step()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            X = torch.tensor(x, dtype=torch.float32)
            return self.model(X).cpu().numpy().reshape(-1)
