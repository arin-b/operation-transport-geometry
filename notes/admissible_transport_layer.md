# Pass 4: admissible invariance and transport layer

Pass 4 implements the admissible-transport layer fully.

## Invariance modes

`hard`: only pairs satisfying all compatibility components are allowed.  
`soft`: all pairs are allowed, but incompatibility becomes a penalty.  
`hybrid`: severe operational or risk violations are forbidden, moderate violations are penalized.  
`adaptive`: thresholds relax as risk error, calibration error, and sample uncertainty increase.

The compatibility object is not semantic compatibility. It is admissible invariance. Semantic descriptors and anchors are merely instruments used to express whether variation is collapsible.

## Transport solvers

`lp`: exact balanced linear-program OT with hard support constraints.  
`sinkhorn`: plain entropic OT, intentionally ignores the mask. This is useful as a baseline.  
`masked_sinkhorn`: entropic OT on the admissible support, with forbidden entries zeroed in the kernel to avoid numerical leakage.  
`unbalanced`: risk-aware unbalanced OT using POT when available, with a trimmed-Sinkhorn fallback.

## Unmatched mass interpretation

Unbalanced mass is decomposed into:

1. finite-sample support mismatch,
2. operationally non-substitutable mass,
3. dangerous failure-mode mass.

This prevents “unmatched mass” from being treated as a single uninterpretable number.
