from __future__ import annotations

import argparse

from benchmarks.runner import DEFAULT_BENCHMARK_CFG, create_bundle_zip, run_benchmark_suite
from otg.utils.config import deep_merge, load_config


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="otg-benchmark", description="OTG literature challenge benchmark")
    s = p.add_subparsers(dest="cmd", required=True)

    r = s.add_parser("run", help="Run the benchmark suite.")
    r.add_argument("--config", default="configs/suites/literature_benchmark.yaml")
    r.add_argument("--out", default="runs/literature_benchmark")
    r.add_argument("--zip", default=None)

    return p


def main() -> None:
    args = parser().parse_args()
    if args.cmd == "run":
        cfg = load_config(args.config)
        cfg = deep_merge(DEFAULT_BENCHMARK_CFG, cfg or {})
        result = run_benchmark_suite(cfg, args.out)
        if args.zip:
            create_bundle_zip(args.out, args.zip)
        print(f"Benchmark complete: {args.out}")
        print("Diagnostic primary-claim order:", ", ".join(result["aggregate"].get("diagnostic_ranking", result["aggregate"].get("ranking", []))))


if __name__ == "__main__":
    main()
