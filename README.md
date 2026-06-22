# operation-transport-geometry

## Final algorithm status

The default implementation is now **adaptive node-sensitive graph OTG**. It operates on a deployed DAG with multiple deployment domains and node-wise laws, uses pairwise OT only as an internal numerical primitive, applies node/domain-pair adaptive operational costs, activates unbalanced transport for terminally induced dangerous unmatched mass, and aggregates node discrepancies with terminal-sensitivity weights.

Final validation: `33 passed` tests; graph validation `332 checks, 0 errors, 0 warnings`; final 5-seed graph-tensor benchmark ranks OTG first on the primary claim average with `0.979`.


This repository now implements **Operational Transport Geometry over a deployed finite DAG**. The primary pipeline is graph-level: multiple deployment domains induce node-wise laws on multiple internal nodes; pairwise OT problems are derived only as an internal compatibility layer for costs, admissibility, and solvers.

The default world is `synthetic_dag`, with domains `clear`, `viewpoint_shift`, `glare`, and `occlusion`; nodes `input`, `detector`, `representation`, `measurement`, and terminal `audit_report`; and selected internal nodes `detector`, `representation`, and `measurement`.

Key outputs include `D_op_matrix.csv`, `node_pair_metrics.csv`, graph-level reports, domain-pair heatmaps, node-by-domain-pair heatmaps, admissibility masks, transport plans, risk distributions, and dangerous-unmatched-mass diagnostics.

See `notes/proposal_implementation_mapping.md` for the implementation mapping to the proposal notation.

## Graph-domain-node benchmark update

The literature benchmark now evaluates the full tensor `W[v,i,j]` and system matrix `D_op[i,j]`. Pairwise OT remains only the numerical primitive. The benchmark now writes claim-specific scores, per-method domain matrices, dangerous-unmatched-mass matrices, and an OTG cost-weight inspection report. See `notes/graph_tensor_benchmark.md`.

## Domain-pair adaptive OTG cost

The default OTG cost mode is now `domain_pair_adaptive`. For every derived `(node, domain_i, domain_j)` transport problem, OTG computes a terminally induced gate from risk shift, terminal-output shift, and failure-rate shift. Harmless nuisance-dominated comparisons receive relaxed operational/risk/invariance weights; harmful comparisons revert toward the conservative OTG weights. The gate is not domain-name-specific. See `notes/domain_pair_adaptive_cost.md`.


## Quick start

```bash
pip install -e .
otg sanity
otg validate --preset fast
otg run-default --preset fast
python -m benchmarks.cli run --config configs/suites/literature_benchmark.yaml --out runs/literature_benchmark
```

---

# otg-testbed

`otg-testbed` is a mathematical testbed for Operational Transport Geometry.

Pass 2 fully builds out the controlled operational worlds. Each world now has an explicit mathematical specification, a richer generative law, a ground-truth operational coordinate decomposition, configurable nuisance and hidden variables, expected diagnostic behavior, and world-level metadata used by reports and assumption checks.

The repository is built around the pipeline:

`domain law -> DAG/node law -> terminal behavior -> true risk -> estimated risk -> admissible invariance -> operational cost -> balanced/unbalanced OT -> diagnostics`

## Pass 2 additions

Pass 2 adds eight first-class controlled operational worlds:

1. `harmless_nuisance`: large geometric shift in nuisance coordinates, low operational shift.
2. `harmful_boundary`: small geometric shift across an operational failure boundary.
3. `invariance_misspecification`: intentionally wrong invariance descriptors create false collapse or false separation.
4. `risk_degradation`: compares true, noisy, rollout, learned, and misspecified risk behavior.
5. `sample_complexity`: repeats the same world across sample-size regimes.
6. `high_dimensional`: repeats boundary stress with nuisance-heavy high-dimensional state.
7. `admissibility_stress`: compares hard, soft, hybrid, and adaptive admissibility.
8. `unbalanced_dangerous_mass`: injects harmful target-domain mass with no valid counterpart.

Three core paper examples are marked in metadata: `harmless_nuisance`, `harmful_boundary`, and `unbalanced_dangerous_mass`.

## Commands

```bash
pip install -e .
otg sanity
otg run-default --preset fast
otg run-suite --preset fast
otg run --world harmless_nuisance --preset fast
```

## Outputs

Each run saves:

- resolved config
- seeds
- JSON metrics
- Markdown report
- LaTeX table snippet
- point-cloud and risk plots
- cost, admissibility, and transport-plan matrices
- world specification metadata
- assumption report

## Design note

The term “semantic compatibility” has been replaced by admissible invariance. Descriptors and anchors are allowed, but they are only one concrete way to express which latent variations may be collapsed. The main object is invariance under downstream operational behavior.


## Pass 3 additions

Pass 3 builds the risk and learning layer fully. Risk is now an interchangeable module with a common output contract. The supported modes are `true`, `noisy`, `rollout`, `learned_regression`, `learned_classifier`, `learned_mlp`, and `misspecified`.

The learned layer is synthetic-first but uses a clean `RiskDataset` API. It supports sklearn regression/classification and an optional PyTorch MLP backend. Reports now include risk MAE, train error where available, calibration error, and failure accuracy.


## Pass 4 additions

Pass 4 fully builds the admissible-invariance and transport layer. The supported admissibility modes are `hard`, `soft`, `hybrid`, and `adaptive`. The supported solvers are `lp`, `sinkhorn`, `masked_sinkhorn`, and `unbalanced`.

The implementation now reports feasibility, forbidden transport mass, marginal errors, plan mass, and an unmatched-mass decomposition into finite-sample mismatch, operational non-substitutability, and dangerous unmatched failure mass.


## Pass 5 additions

Pass 5 builds the diagnostics and reporting layer. Each run now produces JSON summaries, CSV metrics, Markdown reports, compact interpretations, LaTeX table snippets, per-node artifacts, and diagnostic figures. Suite runs now produce `suite_report.md` and `suite_metrics.csv`.

The diagnostic layer includes false collapse, false separation, harmless-shift collapse, harmful-shift detection, transport forbidden mass, marginal errors, feasibility, risk calibration, failure accuracy, and unmatched-mass decomposition.


## Pass 6 additions

Pass 6 finalizes validation and polish. The package now includes `otg doctor`, `otg list-worlds`, and `otg validate`; same-seed reproducibility checks; runtime preset application; compatibility wrappers; final validation documentation; and examples for minimal use and custom-world extension.

Recommended final check:

```bash
pip install -e .
otg doctor
otg sanity
otg validate --preset fast
pytest -q
```
