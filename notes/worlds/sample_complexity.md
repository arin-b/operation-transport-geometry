# sample_complexity

Type: Stress test

Mathematical definition: Same world under smaller empirical laws.

Expected behavior: Transport should stabilize with larger sample size.

Diagnostics to inspect: `transport_value`, `risk_estimation_mae`, `false_collapse_mass`, `false_separation_score`, `allowed_pair_fraction`, `unmatched_mass_b`, and `dangerous_unmatched_mass_b` where applicable.

Interpretation: this is a controlled operational world, not an application simulation. Its purpose is to isolate one failure or success mode of Operational Transport Geometry.
