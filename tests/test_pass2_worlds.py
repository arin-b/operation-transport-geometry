from otg.worlds.registry import available_worlds, make_world
from otg.utils.config import load_config, deep_merge
from otg.utils.seeding import SeedBank
from otg.core.schema import GraphWorldBatch


def test_synthetic_graph_world_generates_node_laws():
    cfg = load_config("configs/default.yaml")
    cfg = deep_merge(cfg, {"runtime_values": {"n": 20, "mc_rollouts": 8}})
    assert "synthetic_dag" in set(available_worlds())
    world = make_world("synthetic_dag", cfg, SeedBank(123))
    data = world.generate()
    assert isinstance(data, GraphWorldBatch)
    assert len(data.domains) >= 4
    assert set(data.selected_nodes) >= {"detector", "representation", "measurement"}
    for node in data.selected_nodes:
        for domain in data.domains:
            law = data.law(node, domain)
            assert law.operational_coords.shape[0] == law.samples.shape[0]
            assert law.aligned_samples.shape[0] == law.samples.shape[0]
            assert law.projection_metadata
