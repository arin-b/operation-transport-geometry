from __future__ import annotations

import argparse
from pathlib import Path

from otg.core.pipeline import run_pipeline
from otg.utils.config import load_config, deep_merge
from otg.utils.io import ensure_dir
from otg.reporting.suite import write_suite_outputs
from otg.validation.runner import run_validation
from otg.validation.doctor import collect_doctor_report, format_doctor_report
from otg.worlds.registry import available_worlds, make_world
from otg.utils.seeding import SeedBank


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="otg", description="OTG mathematical testbed")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sanity", help="Run a small end-to-end sanity case.")
    s.add_argument("--out", default="runs/sanity")

    s = sub.add_parser("doctor", help="Check installed dependencies, registries, and optional backends.")

    s = sub.add_parser("list-worlds", help="List registered controlled operational worlds.")
    s.add_argument("--with-spec", action="store_true")

    s = sub.add_parser("validate", help="Run final validation cases and reproducibility checks.")
    s.add_argument("--out", default="runs/validation")
    s.add_argument("--preset", default="fast", choices=["fast", "default", "heavy"])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--strict", action="store_true")

    s = sub.add_parser("run-default", help="Run the default experiment.")
    s.add_argument("--out", default="runs/default")
    s.add_argument("--preset", default="fast", choices=["fast", "default", "heavy"])
    s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("run", help="Run one controlled operational world.")
    s.add_argument("--config", default="configs/default.yaml")
    s.add_argument("--world", default=None)
    s.add_argument("--solver", default=None, choices=["lp", "sinkhorn", "masked_sinkhorn", "unbalanced"])
    s.add_argument("--risk-mode", default=None, choices=["true", "noisy", "rollout", "misspecified", "learned_regression", "learned_classifier", "learned_mlp"])
    s.add_argument("--admissibility", default=None, choices=["hard", "soft", "hybrid", "adaptive"])
    s.add_argument("--cost-mode", default=None, choices=["geometry", "geometry_risk", "geometry_output_risk", "operational_only", "full"])
    s.add_argument("--preset", default=None, choices=["fast", "default", "heavy"])
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--out", default="runs/custom")

    s = sub.add_parser("run-suite", help="Run a YAML suite of experiments.")
    s.add_argument("--config", default="configs/suites/full_suite.yaml")
    s.add_argument("--out", default="runs/suite")
    s.add_argument("--preset", default="fast", choices=["fast", "default", "heavy"])
    s.add_argument("--seed", type=int, default=0)

    return p


def main() -> None:
    args = parser().parse_args()

    if args.cmd == "doctor":
        print(format_doctor_report(collect_doctor_report()))
        return

    if args.cmd == "list-worlds":
        names = available_worlds()
        if not args.with_spec:
            print("\n".join(names))
            return
        cfg = load_config("configs/default.yaml")
        for name in names:
            local = deep_merge(cfg, {"world": {"name": name}, "runtime_values": {"n": 4, "mc_rollouts": 4}})
            world = make_world(name, local, SeedBank(0))
            spec = world.spec() if hasattr(world, "spec") else {}
            print(f"{name}: {spec.get('expected_behavior', 'no specification available')}")
        return

    if args.cmd == "validate":
        summary = run_validation(args.out, preset=args.preset, seed=args.seed, strict=args.strict)
        print(f"Validation {summary['status']}: {summary['num_checks']} checks, {summary['num_errors']} errors, {summary['num_warnings']} warnings. Output: {args.out}")
        if summary["status"] != "pass":
            raise SystemExit(1)
        return

    if args.cmd == "sanity":
        cfg = load_config("configs/default.yaml")
        cfg = deep_merge(cfg, {
            "world": {"name": "harmless_nuisance"},
            "runtime": {"preset": "fast"},
            "runtime_values": {"n": 36, "mc_rollouts": 16},
            "transport": {"solver": "masked_sinkhorn"},
            "risk": {"mode": "true"},
            "seed": {"master": 123},
        })
        run_pipeline(cfg, args.out)
        print(f"Sanity complete: {args.out}")
        return

    if args.cmd == "run-default":
        cfg = load_config("configs/default.yaml")
        cfg = deep_merge(cfg, {"runtime": {"preset": args.preset}, "seed": {"master": args.seed}})
        run_pipeline(cfg, args.out)
        print(f"Default run complete: {args.out}")
        return

    if args.cmd == "run":
        cfg = load_config(args.config)
        ov = {}
        if args.world:
            ov.setdefault("world", {})["name"] = args.world
        if args.solver:
            ov.setdefault("transport", {})["solver"] = args.solver
        if args.risk_mode:
            ov.setdefault("risk", {})["mode"] = args.risk_mode
        if args.admissibility:
            ov.setdefault("admissibility", {})["mode"] = args.admissibility
        if args.cost_mode:
            ov.setdefault("cost", {})["mode"] = args.cost_mode
        if args.preset:
            ov.setdefault("runtime", {})["preset"] = args.preset
        if args.seed is not None:
            ov.setdefault("seed", {})["master"] = args.seed
        cfg = deep_merge(cfg, ov)
        run_pipeline(cfg, args.out)
        print(f"Run complete: {args.out}")
        return

    if args.cmd == "run-suite":
        suite = load_config(args.config)
        base = load_config(suite.get("base_config", "configs/default.yaml"))
        out = ensure_dir(args.out)
        results = []
        for i, item in enumerate(suite.get("worlds", [])):
            name = item["name"] if isinstance(item, dict) else str(item)
            overrides = item.get("overrides", {}) if isinstance(item, dict) else {}
            cfg = deep_merge(base, overrides)
            cfg = deep_merge(cfg, {
                "world": {"name": name},
                "runtime": {"preset": args.preset},
                "seed": {"master": args.seed + 1000 * i},
            })
            run_dir = out / f"{i:02d}_{name}"
            artifact = run_pipeline(cfg, run_dir)
            results.append({"world": name, "system_score": artifact.system_score, "out": str(run_dir), "summary": _artifact_summary_for_suite(artifact)})
        write_suite_outputs(out, results)
        print(f"Suite complete: {args.out}")
        return


def _artifact_summary_for_suite(artifact):
    return {
        "world": artifact.world_name,
        "system_score": artifact.system_score,
        "nodes": {
            k: {
                "transport_value": float(v.transport.value),
                "solver": v.transport.solver,
                "status": v.transport.status,
                "diagnostics": v.diagnostics,
            }
            for k, v in artifact.node_results.items()
        },
    }


if __name__ == "__main__":
    main()
