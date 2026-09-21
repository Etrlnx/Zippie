import tomllib

import pytest

from airsim_interface.env import AirSimEnv
from state_processing.graph_builder import GraphBuilder


@pytest.fixture
def config():
    with open("configs/default.toml", "rb") as f:
        return tomllib.load(f)


def test_graph_builder(config):
    env = AirSimEnv(config)
    builder = GraphBuilder(config)
    obs = env.reset()
    graph = builder.build_graph(obs)

    assert "x" in graph
    assert "mask" in graph
    assert "drone_indices" in graph
    assert graph["x"].shape == (1, config["simulation"]["max_nodes"], config["model"]["node_dim"])
    assert graph["mask"].shape == (1, config["simulation"]["max_nodes"])
    assert len(graph["drone_indices"]) == config["simulation"]["num_agents"]
