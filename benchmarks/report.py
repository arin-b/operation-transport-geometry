from __future__ import annotations

from pathlib import Path


def write_method_summary(out_dir: str | Path, summary: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text(
        f"{summary['method']} / {summary['world']} / seed={summary['seed']} / score={summary['composite_score']:.3f}\n",
        encoding="utf-8",
    )
