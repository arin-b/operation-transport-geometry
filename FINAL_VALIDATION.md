# Final validation notes

The final repository was validated with four layers:

1. Unit tests for costs, DAG execution, transport solvers, risk estimators, and reporting artifacts.
2. Pipeline tests for the main controlled operational worlds.
3. CLI tests for `doctor`, `list-worlds`, and `validate`.
4. A final validation command: `python -m otg.cli validate --preset fast`.

The final pass also fixes runtime preset application. `fast`, `default`, and `heavy` now fill `runtime_values` unless exact sample sizes are already supplied.

Plain `sinkhorn` remains a deliberate baseline that ignores admissibility masks. Use `masked_sinkhorn` for admissible entropic OT.
