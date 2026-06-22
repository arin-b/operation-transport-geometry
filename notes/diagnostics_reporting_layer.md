# Pass 5: diagnostics and reporting layer

Pass 5 turns each run into a reusable experiment artifact.

Each run now emits:

- `summary.json`
- `assumptions.json`
- `metrics.csv`
- `scorecard.csv`
- `report.md`
- `interpretation.md`
- `table_snippet.tex`
- `assumption_table_snippet.tex`
- per-node diagnostics JSON
- per-node transport metadata JSON
- NumPy arrays for costs, masks, transport plans, risks, and samples
- figures for point clouds, risk distributions, calibration, cost matrices, masks, transport plans, transport arrows, diagnostics, and unmatched mass

The scorecard intentionally separates raw metrics from interpretation. A world-specific expectation is checked for the main stress cases, but the raw diagnostics remain available for paper analysis.
