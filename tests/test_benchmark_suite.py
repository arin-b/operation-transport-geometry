from benchmarks import available_methods
from benchmarks.runner import run_benchmark_suite


def test_benchmark_methods_registered():
    methods = available_methods()
    assert "otg" in methods
    assert "toalign" in methods
    assert "irm" in methods


def test_benchmark_suite_smoke(tmp_path):
    cfg = {
        "worlds": ["synthetic_dag"],
        "methods": ["otg", "irm"],
        "seed_count_main": 1,
        "seed_count_stress": 0,
        "shared_seed_stride": 1000,
        "solver": "masked_sinkhorn",
        "runtime": {"preset": "fast"},
        "runtime_values": {"n": 16, "mc_rollouts": 4},
        "seed": {"master": 7},
        "benchmark": {
            "cost_mode": "full",
            "dangerous_unmatched": {"enabled": True, "target_domain": "occlusion", "solver": "unbalanced"},
        },
    }
    result = run_benchmark_suite(cfg, tmp_path / "benchmark")
    assert result["aggregate"]["diagnostic_ranking"]
    assert "claim_rankings" in result["aggregate"]
    assert (tmp_path / "benchmark" / "benchmark_report.md").exists()
    assert (tmp_path / "benchmark" / "benchmark_results.csv").exists()
    assert (tmp_path / "benchmark" / "benchmark_claim_scores.csv").exists()
    assert (tmp_path / "benchmark" / "benchmark_results.json").exists()
    assert (tmp_path / "benchmark" / "figures" / "claim_scores_heatmap.png").exists()
    assert (tmp_path / "benchmark" / "figures" / "method_claim_matrix.png").exists()
    assert (tmp_path / "benchmark" / "method_domain_matrices" / "otg_D_op_mean.csv").exists()
    assert (tmp_path / "benchmark" / "method_domain_matrices" / "otg_dangerous_unmatched_mean.csv").exists()


def test_benchmark_dangerous_unmatched_is_graph_claim(tmp_path):
    cfg = {
        "worlds": ["synthetic_dag"],
        "methods": ["otg"],
        "seed_count_main": 1,
        "seed_count_stress": 0,
        "runtime": {"preset": "fast"},
        "runtime_values": {"n": 24, "mc_rollouts": 4},
        "seed": {"master": 11},
        "benchmark": {
            "dangerous_unmatched": {"enabled": True, "target_domain": "occlusion", "solver": "unbalanced"},
        },
    }
    result = run_benchmark_suite(cfg, tmp_path / "benchmark")
    run = result["results"][0]
    assert run["pipeline"] == "graph_domain_node_tensor"
    assert run["claim_scores"]["dangerous_unmatched_exposed"] > 0.0
    assert run["claim_scores"]["dangerous_unmatched_raw"] > 0.0
    assert len(run["domain_order"]) >= 4
    assert len(run["selected_nodes"]) >= 3
