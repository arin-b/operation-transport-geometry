# Domain-pair adaptive OTG cost

The default OTG cost profile is now `domain_pair_adaptive`.

The cost still has the proposal form

```text
c_{v,op} = geometry + terminal + risk + invariance + operational-coordinate + nuisance
```

but the low-level weights are no longer globally fixed for every node/domain comparison. For each derived `(v,d_i,d_j)` transport problem, the implementation computes a terminally induced activation gate from:

- mean estimated risk shift,
- mean terminal-output shift,
- failure-rate shift.

When these quantities are small, the pair is treated as likely harmless/nuisance-dominated and the risk, terminal, operational-coordinate, and invariance weights are relaxed toward configured minima. When they are large, the weights revert toward the conservative OTG weights.

This is intentionally not domain-name-specific. The gate does not check whether a domain is called `viewpoint_shift`, `glare`, or `occlusion`. It uses only downstream operational evidence induced by the deployed system.

The resulting rule is meant to reduce false separation on harmless nuisance shifts while preserving harmful separation and dangerous unmatched mass.

Relevant config block:

```yaml
cost:
  mode: domain_pair_adaptive
  weights:
    geometry: 1.0
    terminal: 1.0
    risk: 2.0
    invariance: 3.0
    operational_coordinate: 1.5
    nuisance: 0.15
  adaptive:
    risk_threshold: 0.06
    terminal_threshold: 0.10
    failure_threshold: 0.05
    min_weights:
      geometry: 0.35
      terminal: 0.10
      risk: 0.20
      invariance: 0.15
      operational_coordinate: 0.25
      nuisance: 0.02
```

Benchmark outputs include `cost_weight_adaptive_gate`, effective component weights, and shift diagnostics in `otg_cost_weight_inspection.csv`.
