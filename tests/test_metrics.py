import tomllib

import pytest

from evaluation.metrics import MetricEvaluator


@pytest.fixture
def config():
    with open("configs/default.toml", "rb") as f:
        return tomllib.load(f)


def test_metric_evaluator(config):
    evaluator = MetricEvaluator(config)
    logs = [
        {
            "all_reached": True,
            "any_reached": True,
            "steps": 50,
            "total_collisions": 1,
            "total_energy": 12.5,
            "trajectories": {"drone_0": [[0, 0, 0], [1, 1, 1], [2, 2, 2]]},
            "latencies_ms": [1.2, 1.5]
        }
    ]
    res = evaluator.evaluate_trajectories(logs)
    assert res["success_rate"] == 1.0
    assert res["collision_rate"] == 1.0
    assert res["completion_time"] == 50.0
