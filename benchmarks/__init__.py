from .registry import available_methods, get_method, register_method
from .runner import run_benchmark, run_benchmark_suite
from . import papers  # noqa: F401

__all__ = [
    "available_methods",
    "get_method",
    "register_method",
    "run_benchmark",
    "run_benchmark_suite",
]
