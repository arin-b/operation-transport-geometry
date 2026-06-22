# Pass 1 architecture hardening

This pass builds the repository as a set of explicit contracts.

The current implementation is intentionally modular. Worlds generate empirical node laws. Risk models replace the risk field used by the algorithm while preserving true risk for diagnostics. Invariance builders compute allowed transport pairs and penalties. Cost builders combine geometry, terminal output, risk, and invariance. Transport solvers consume a single `TransportProblem`. Diagnostics measure whether the pipeline collapses harmless variation, separates harmful variation, and exposes failures.

Pass 1 does not claim that the eight worlds are final mathematical examples. They are runnable controlled operational worlds that validate the architecture. Pass 2 should refine their definitions.
