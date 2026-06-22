from otg.worlds.registry import available_worlds, make_world
from otg.utils.config import load_config, deep_merge
from otg.utils.seeding import SeedBank


def test_all_pass2_worlds_generate():
    cfg = load_config("configs/default.yaml")
    expected = {
        "harmless_nuisance",
        "harmful_boundary",
        "invariance_misspecification",
        "risk_degradation",
        "sample_complexity",
        "high_dimensional",
        "admissibility_stress",
        "unbalanced_dangerous_mass",
    }
    assert expected.issubset(set(available_worlds()))
    for name in expected:
        local = deep_merge(cfg, {"world": {"name": name}, "runtime_values": {"n": 20, "mc_rollouts": 8}})
        world = make_world(name, local, SeedBank(123))
        data = world.generate()
        assert "world_spec" in data.metadata
        assert "repr" in data.nodes
        node = data.nodes["repr"]
        assert node.operational_coords_a.shape[0] == node.z_a.shape[0]
        assert node.nuisance_coords_a.shape[0] == node.z_a.shape[0]
