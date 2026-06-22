# Usage

Install locally:

```bash
pip install -e .
```

Run a smoke test:

```bash
otg sanity
```

Inspect the environment:

```bash
otg doctor
```

List worlds:

```bash
otg list-worlds --with-spec
```

Run final validation:

```bash
otg validate --preset fast --out runs/validation
```

Run one custom experiment:

```bash
otg run --world harmful_boundary --risk-mode true --solver masked_sinkhorn --admissibility hybrid --preset fast --out runs/harmful_boundary
```
