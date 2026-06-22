# Graph-domain-node benchmark realignment

The literature benchmark no longer treats a single domain pair as the benchmark object. The benchmark object is now the graph-induced tensor

\[
W_{vij}=W_{v,op}(\mu_v^{d_i},\mu_v^{d_j}),
\]

with shape `|V_sel| x |D| x |D|`, together with the aggregated system matrix

\[
D_{op,ij}=A_v(W_{vij}).
\]

Pairwise OT is still used because optimal transport compares two measures, but it is now only the numerical primitive that fills entries of the full tensor.

## Dangerous unmatched mass

The benchmark explicitly activates unbalanced OT for comparisons involving the configured dangerous target domain, default `occlusion`. This tests whether dangerous target mass is exposed as unmatched mass instead of being hidden inside a forced balanced match. The outputs include:

- `dangerous_unmatched_matrix.csv` per run;
- `method_domain_matrices/<method>_dangerous_unmatched_mean.csv` across seeds;
- `dangerous_unmatched_exposed` as a primary claim score.

## Claim-specific scoring

The benchmark no longer reports one composite number as the scientific conclusion. It writes separate claim scores for:

- harmless operational collapse;
- harmful operational separation;
- admissibility respected;
- false collapse avoided;
- dangerous unmatched mass exposed;
- system-level localization.

The field `primary_claim_average_diagnostic` is retained only as a compact diagnostic average for sorting tables. It is not a substitute for claim-level interpretation.

## Method-by-method matrices

For every method and seed, the run directory contains full domain matrices. Across seeds, mean matrices are written under `method_domain_matrices/`. These are the objects to inspect before concluding that OTG wins, loses, or needs algorithmic changes.

## OTG cost-weight inspection

The benchmark writes `otg_cost_weight_inspection.csv` and `otg_cost_weight_inspection.md`. These report the configured OTG weights and the observed geometric, terminal, risk, invariance, and total operational contributions on the transport plans. The inspection is diagnostic only; it does not tune weights to make OTG win.

## Adaptive OTG cost profile

The benchmark can set `benchmark.otg_cost_mode=domain_pair_adaptive`. In that mode OTG keeps the same graph-tensor benchmark object, but each derived `(v,d_i,d_j)` cost uses a gate induced by risk, terminal-output, and failure-rate shifts. The gate relaxes operational/risk/invariance penalties when the comparison is terminally harmless and restores the conservative weights when the downstream behavior moves.

This is a low-level OTG choice, not a benchmark-scoring trick. The gate is computed before the transport solve from the deployed system's terminal behavior and does not inspect method rankings.
