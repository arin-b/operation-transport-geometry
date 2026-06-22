import numpy as np

from otg.core.data import NodeData
from otg.core.costs import build_operational_cost


def test_cost_shape():
    n, m, d = 5, 7, 2
    node = NodeData(
        name="repr",
        z_a=np.zeros((n, d)),
        z_b=np.ones((m, d)),
        y_a=np.zeros(n),
        y_b=np.ones(m),
        true_risk_a=np.zeros(n),
        true_risk_b=np.ones(m),
        used_risk_a=np.zeros(n),
        used_risk_b=np.ones(m),
        descriptors_a=np.zeros((n, 2)),
        descriptors_b=np.ones((m, 2)),
        anchors_a=np.zeros(n),
        anchors_b=np.ones(m),
        mass_a=np.ones(n) / n,
        mass_b=np.ones(m) / m,
        harmful_a=np.zeros(n, dtype=bool),
        harmful_b=np.ones(m, dtype=bool),
        op_coords_a=np.zeros((n, 1)),
        op_coords_b=np.ones((m, 1)),
        nuisance_a=np.zeros((n, 1)),
        nuisance_b=np.ones((m, 1)),
    )
    cfg = {"cost": {"mode": "full", "weights": {}}}
    pack = build_operational_cost(node, cfg)
    assert pack.cost.shape == (n, m)
    assert np.all(np.isfinite(pack.cost))
