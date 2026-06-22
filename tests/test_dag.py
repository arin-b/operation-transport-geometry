import numpy as np
from otg.core.dag import OperationalDAG


def test_dag_runs_in_order():
    dag = OperationalDAG()
    dag.add("a", ["x"], lambda s: s["x"] + 1)
    dag.add("b", ["a"], lambda s: s["a"] * 2)
    out = dag.run({"x": np.array([1, 2])})
    assert np.all(out["b"] == np.array([4, 6]))
