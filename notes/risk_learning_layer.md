# Pass 3: risk and learning layer

Pass 3 separates the risk layer from world generation.

The world always generates true downstream operational risk. The algorithm may then receive one of several risk views:

1. `true`: exact ground-truth risk.
2. `noisy`: ground-truth risk plus controlled additive noise.
3. `rollout`: Monte Carlo estimate of threshold-crossing probability under terminal perturbations.
4. `learned_regression`: supervised sklearn risk regression.
5. `learned_classifier`: supervised sklearn failure-probability classifier.
6. `learned_mlp`: MLP risk estimator using sklearn by default, with optional PyTorch backend.
7. `misspecified`: deliberately wrong risk proxy based on nuisance magnitude.

Each risk estimator returns a common `RiskOutput` containing used risks, MAE against true risk, calibration error, failure accuracy, train error where applicable, and estimator metadata.

This layer is synthetic-first but dataset-API compatible: all learned models train through `RiskDataset`, which can later be replaced by external data without changing the OT/invariance/transport pipeline.
