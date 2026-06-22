# risk_degradation

Type: Stress test

Mathematical definition: Same boundary setting under degraded risk estimates.

Expected behavior: Risk MAE should explain transport degradation.

Diagnostics to inspect: `transport_value`, `risk_estimation_mae`, `false_collapse_mass`, `false_separation_score`, `allowed_pair_fraction`, `unmatched_mass_b`, and `dangerous_unmatched_mass_b` where applicable.

Interpretation: this is a controlled operational world, not an application simulation. Its purpose is to isolate one failure or success mode of Operational Transport Geometry.
