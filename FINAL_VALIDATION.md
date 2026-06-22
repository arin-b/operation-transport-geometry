# Final validation: adaptive node-sensitive graph OTG

This repository has been realigned around the proposal-level OTG object:

```text
GraphWorldBatch = (G, D, V_sel, {mu_v^d}, {phi_v^d}, {r_v^d})
```

Pairwise OT is retained only as the numerical primitive used to populate the full graph tensor `W[v,i,j]` and system matrix `D_op[i,j]`.

## Final algorithmic choices

The final default OTG implementation is `adaptive_node_pair`:

1. Shared-space projection `phi_v^d` is used by default through `aligned_samples`.
2. Operational risk is terminally induced and stored as true `r_v`; the algorithm consumes estimated/used `rhat_v`.
3. The ground cost is node-sensitive and domain-pair adaptive.
4. Hybrid admissibility is used: hard exclusions for severe operational/risk incompatibility and soft penalties for uncertain descriptor mismatch.
5. Solver policy is `auto`: masked Sinkhorn for ordinary admissible balanced transport, unbalanced OT when terminally induced dangerous mass is unmatched, and LP for very small audit cases.
6. System aggregation is terminal-sensitivity weighted.
7. Localization uses null-corrected excess node discrepancy rather than raw transport magnitude.
8. The benchmark reports claim-specific graph-tensor scores instead of a single one-node composite score.

## Validation commands run

```bash
python -m pytest -q
python -m otg.cli validate --preset fast --out /mnt/data/otg_final_validation2
python -m benchmarks.cli run --config /mnt/data/otg_final_exact_5seed.yaml --out /mnt/data/otg_final_exact_benchmark_5seed --zip /mnt/data/otg_final_exact_benchmark_5seed_results.zip
```

## Validation status

- Unit/integration tests: `33 passed`
- Graph validation: `332 checks, 0 errors, 0 warnings`
- Final 5-seed graph-tensor benchmark completed successfully.

## Final 5-seed benchmark summary

Same benchmark family as the previous runs: synthetic graph world, 7 methods, 5 seeds, fast preset with `n=40`, `mc_rollouts=16`, claim-specific graph-tensor scoring.

| Rank | Method | Primary mean | Harmless collapse | Harmful separation | Dangerous unmatched | Localization |
|---:|---|---:|---:|---:|---:|---:|
| 1 | OTG | 0.979 | 0.938 | 0.935 | 1.000 | 1.000 |
| 2 | SPO | 0.859 | 0.617 | 0.601 | 1.000 | 0.933 |
| 3 | GroupDRO | 0.824 | 0.617 | 0.591 | 1.000 | 0.733 |
| 4 | SRW | 0.798 | 0.543 | 0.510 | 1.000 | 0.733 |
| 5 | IRM | 0.795 | 0.579 | 0.528 | 1.000 | 0.667 |
| 6 | ToAlign | 0.788 | 0.551 | 0.509 | 1.000 | 0.667 |
| 7 | MLOT | 0.771 | 0.510 | 0.452 | 1.000 | 0.667 |

## Final OTG domain matrix

```text
clear              viewpoint_shift   glare      occlusion
clear              0.000             0.369      7.571      5.077
viewpoint_shift    0.369             0.000      7.524      5.190
glare              7.571             7.524      0.000      6.666
occlusion          5.077             5.190      6.666      0.000
```

Interpretation: harmless viewpoint shift now collapses strongly, while glare and occlusion remain separated. Occlusion also exposes dangerous unmatched mass under adaptive unbalanced transport.
