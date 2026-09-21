import tomllib

import pytest

from airsim_interface.env import AirSimEnv


@pytest.fixture
def config():
    with open("configs/default.toml", "rb") as f:
        return tomllib.load(f)


def test_env_reset(config):
    env = AirSimEnv(config)
    obs = env.reset()
    assert len(obs["drones"]) == config["simulation"]["num_agents"]
    assert len(obs["obstacles"]) == 10
    assert len(obs["targets"]) == config["simulation"]["num_agents"]


def test_env_step(config):
    env = AirSimEnv(config)
    env.reset()
    actions = {"drone_0": 1, "drone_1": 2, "drone_2": 3}
    obs, rewards, done, info = env.step(actions)
    assert len(rewards) == config["simulation"]["num_agents"]
    assert isinstance(done, bool)
    assert "collisions" in info
    assert "thrust_magnitudes" in info


def test_brake_action_zeroes_velocity(config):
    env = AirSimEnv(config)
    env.reset()
    move_actions = {f"drone_{i}": 1 for i in range(env.num_agents)}
    for _ in range(3):
        env.step(move_actions)
    assert all(float(v_norm) > 0.5 for v_norm in (env.drone_velocities**2).sum(axis=1)**0.5)
    brake_actions = {f"drone_{i}": env.brake_action_idx for i in range(env.num_agents)}
    env.step(brake_actions)
    assert all(float(v_norm) < 1e-4 for v_norm in (env.drone_velocities**2).sum(axis=1)**0.5)
