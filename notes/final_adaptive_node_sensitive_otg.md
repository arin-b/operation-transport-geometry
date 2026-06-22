# Final adaptive node-sensitive graph OTG

This note records the final algorithmic choices used in the implementation.

The benchmark and default pipeline operate on the graph-level object
`GraphWorldBatch = (G, D, V_sel, {mu_v^d}, {phi_v^d}, {r_v^d})`. Pairwise OT is only the numerical primitive used to populate `W[v,i,j]` and `D_op[i,j]`.

## Final choices

1. Shared-space projection: costs are computed on `aligned_samples` by default; raw samples are retained for projection diagnostics.
2. Risk: the algorithm consumes `rhat_v`; the synthetic testbed stores true terminally induced `r_v` for audit.
3. Cost: `adaptive_node_pair`, a node-sensitive/domain-pair adaptive cost profile.
4. Admissibility: hybrid hard/soft admissibility.
5. Transport: `auto` solver policy. Balanced masked Sinkhorn is used by default; unbalanced OT activates when terminally induced high-risk/dangerous mass is unmatched.
6. Aggregation: terminal-sensitivity-weighted node aggregation.
7. Localization: null-corrected localization scores, not raw node transport values.
8. Benchmark: claim-specific graph tensor scores, not a single one-node composite leaderboard.

## Implemented formula

For node `v` and domain pair `(i,j)`:

```math
c^{ij}_{v,op}
= \lambda^{ij}_v c_{geom}
+ \alpha^{ij}_v c_{risk}
+ \beta^{ij}_v c_{term}
+ \eta^{ij}_v c_{adm}
+ \gamma^{ij}_v c_{node}
+ \xi^{ij}_v c_{nuis}.
```

The adaptive gate is computed from risk, terminal-output, and failure-rate shifts:

```math
g_{vij}=\sigma\left(b+a_r\frac{\Delta r_{vij}-\tau_r}{s_r}+a_y\frac{\Delta y_{ij}-\tau_y}{s_y}+a_f\frac{\Delta f_{ij}-\tau_f}{s_f}\right).
```

Weights interpolate between node-specific minimum and maximum values:

```math
w^{ij}_{v,k}=w^{\min}_{v,k}+g_{vij}(w^{\max}_{v,k}-w^{\min}_{v,k}).
```

System aggregation uses terminal sensitivity:

```math
D_{op}(d_i,d_j)=\sum_{v\in V_{sel}}\omega^{ij}_vW_{v,op}(d_i,d_j),
\quad
\omega^{ij}_v\propto s_v(\Delta r_{vij}+\Delta y_{ij}+\Delta f_{ij}+\epsilon).
```

Localization uses null-corrected excess:

```math
L_{vij}=s_ve_{vij}[W_{v,op}(d_i,d_j)-W^{null}_{vij}]_+.
```
