# invariance_misspecification

Type: Stress test

Mathematical definition: Descriptor and anchor structure are intentionally wrong in one domain.

Expected behavior: False collapse/separation should rise.

Diagnostics to inspect: `transport_value`, `risk_estimation_mae`, `false_collapse_mass`, `false_separation_score`, `allowed_pair_fraction`, `unmatched_mass_b`, and `dangerous_unmatched_mass_b` where applicable.

Interpretation: this is a controlled operational world, not an application simulation. Its purpose is to isolate one failure or success mode of Operational Transport Geometry.
