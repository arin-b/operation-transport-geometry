from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import platform
import sys


def _package_status(import_name: str, dist_name: str | None = None, *, import_module: bool = False) -> dict:
    dist_name = dist_name or import_name
    if importlib.util.find_spec(import_name) is None:
        return {"available": False, "error": "not_found"}
    if import_module:
        try:
            mod = importlib.import_module(import_name)
            return {"available": True, "version": getattr(mod, "__version__", "unknown")}
        except Exception as exc:
            return {"available": False, "error": type(exc).__name__}
    try:
        version = importlib.metadata.version(dist_name)
    except Exception:
        version = "available"
    return {"available": True, "version": version}


def collect_doctor_report() -> dict:
    # Avoid importing heavyweight optional packages such as torch in the doctor path.
    packages = {
        "numpy": _package_status("numpy"),
        "scipy": _package_status("scipy"),
        "matplotlib": _package_status("matplotlib"),
        "yaml": _package_status("yaml", "PyYAML"),
        "sklearn": _package_status("sklearn", "scikit-learn"),
        "ot": _package_status("ot", "POT"),
        "torch": _package_status("torch"),
    }

    from otg.worlds.registry import available_worlds
    from otg.risks.registry import RISK_REGISTRY
    from otg.invariance.registry import INVARIANCE_REGISTRY
    from otg.transport.registry import TRANSPORT_REGISTRY
    import otg.risks.models  # noqa: F401
    import otg.invariance.builders  # noqa: F401
    import otg.transport.backends  # noqa: F401

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "worlds": available_worlds(),
        "risk_modes": RISK_REGISTRY.names(),
        "invariance_modes": INVARIANCE_REGISTRY.names(),
        "transport_solvers": TRANSPORT_REGISTRY.names(),
    }


def format_doctor_report(report: dict) -> str:
    lines = ["OTG doctor report", ""]
    lines.append(f"Python: {report['python']}")
    lines.append(f"Platform: {report['platform']}")
    lines.append("")
    lines.append("Packages:")
    for name, info in report["packages"].items():
        if info["available"]:
            lines.append(f"  - {name}: ok ({info['version']})")
        else:
            lines.append(f"  - {name}: missing/optional ({info.get('error')})")
    lines.append("")
    lines.append("Worlds: " + ", ".join(report["worlds"]))
    lines.append("Risk modes: " + ", ".join(report["risk_modes"]))
    lines.append("Invariance modes: " + ", ".join(report["invariance_modes"]))
    lines.append("Transport solvers: " + ", ".join(report["transport_solvers"]))
    return "\n".join(lines)
