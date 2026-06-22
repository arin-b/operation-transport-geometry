# Pass 6: validation and polish

Pass 6 adds the final repository layer: validation commands, reproducibility checks, CLI stability, preset handling, examples, and compatibility wrappers.

New commands:

- `otg doctor`: dependency and registry report.
- `otg list-worlds`: registered controlled operational worlds.
- `otg validate`: final validation suite and reproducibility check.

The final validation suite checks finite transport values, solver status, forbidden mass, admissibility feasibility, true-risk consistency, and same-seed reproducibility.
