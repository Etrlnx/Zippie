from typing import Any
import numpy as np
from airsim_interface.sensor_reader import SensorReader

class AirSimEnv:
    def __init__(self, config: dict[str, Any], seed: int | None = None):
        self.config = config
        sim_cfg = config.get("simulation", {})
        env_cfg = config.get("environment", {})
        term_cfg = config.get("termination", {})
        self.num_agents = sim_cfg.get("num_agents", 3)
        self.max_steps = sim_cfg.get("max_steps", 200)
        self.bounds = np.array(sim_cfg.get("world_bounds", [-50.0, 50.0, -50.0, 50.0, 0.0, 20.0]), dtype=np.float32)
        self.backend = sim_cfg.get("backend", "fast_surrogate")
        self.action_space_type = env_cfg.get("action_space", "discrete")
        self.reach_dist = env_cfg.get("reach_distance", 2.0)
        self.drone_radius = env_cfg.get("drone_radius", 0.5)
        self.dt = env_cfg.get("dt", 0.1)
        self.battery_drain_rate = env_cfg.get("battery_drain_rate", 0.0005)
        self.all_targets_reached_term = term_cfg.get("all_targets_reached", True)
        self.timeout_term = term_cfg.get("timeout_terminates", True)
        self.battery_depleted_term = term_cfg.get("battery_depleted_terminates", True)
        self.wind_perturbation = env_cfg.get("wind_perturbation", 0.0)
        self.sensor_reader = SensorReader(config)
        from rl_engine.reward import RewardEngine
        self.reward_engine = RewardEngine(config)
        self.current_step = 0
        self.num_obstacles = env_cfg.get("num_obstacles", 10)
        self.drone_positions = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.drone_velocities = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.drone_orientations = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.drone_batteries = np.ones(self.num_agents, dtype=np.float32)
        self.target_positions = np.zeros((self.num_agents, 3), dtype=np.float32)
        self.obstacles = []
        if seed is not None:
            np.random.seed(seed)
        discrete_actions_cfg = env_cfg.get("discrete_actions", None)
        if discrete_actions_cfg:
            self.discrete_actions = [np.array(a, dtype=np.float32) for a in discrete_actions_cfg]
        else:
            self.discrete_actions = [
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                np.array([2.0, 0.0, 0.0], dtype=np.float32),
                np.array([-2.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 2.0, 0.0], dtype=np.float32),
                np.array([0.0, -2.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0, 1.5], dtype=np.float32),
                np.array([0.0, 0.0, -1.5], dtype=np.float32),
            ]
        self.brake_action_idx = len(self.discrete_actions)
        self.num_discrete_actions = len(self.discrete_actions) + 1

    def _debias_spawn_axes(self, drone_pos: np.ndarray, target_pos: np.ndarray, axis_lo: np.ndarray, axis_hi: np.ndarray) -> np.ndarray:
        gap = np.abs(target_pos - drone_pos)
        dominant = int(np.argmax(gap))
        others = [a for a in range(3) if a != dominant]
        if gap[dominant] <= 1.5 * gap[others].max():
            return target_pos
        for a in others:
            sign = 1.0 if target_pos[a] >= drone_pos[a] else -1.0
            max_room = (axis_hi[a] - drone_pos[a]) if sign > 0 else (drone_pos[a] - axis_lo[a])
            boosted_gap = min(gap[dominant], max_room)
            target_pos[a] = np.clip(drone_pos[a] + sign * boosted_gap, axis_lo[a], axis_hi[a])
        gap = np.abs(target_pos - drone_pos)
        if int(np.argmax(gap)) == dominant:
            runner_up = gap[others].max()
            sign = 1.0 if target_pos[dominant] >= drone_pos[dominant] else -1.0
            target_pos[dominant] = np.clip(drone_pos[dominant] + sign * runner_up, axis_lo[dominant], axis_hi[dominant])
        return target_pos

    def reset(self) -> dict[str, Any]:
        self.current_step = 0
        self.drone_velocities.fill(0.0)
        self.drone_orientations.fill(0.0)
        self.drone_batteries.fill(1.0)
        axis_lo = np.array([self.bounds[0] + 5, self.bounds[2] + 5, self.bounds[4] + 2], dtype=np.float32)
        axis_hi = np.array([self.bounds[1] - 5, self.bounds[3] - 5, self.bounds[5] - 2], dtype=np.float32)
        min_agent_sep = 8.0
        for i in range(self.num_agents):
            for attempt in range(100):
                drone_pos = np.random.uniform(axis_lo, axis_hi).astype(np.float32)
                if all(np.linalg.norm(drone_pos - self.drone_positions[j]) >= min_agent_sep for j in range(i)):
                    break
            target_pos = np.random.uniform(axis_lo, axis_hi).astype(np.float32)
            target_pos = self._debias_spawn_axes(drone_pos, target_pos, axis_lo, axis_hi)
            self.drone_positions[i] = drone_pos
            self.target_positions[i] = target_pos
        self.obstacles = []
        for _ in range(self.num_obstacles):
            obs_pos = np.array([
                np.random.uniform(self.bounds[0] + 15, self.bounds[1] - 15),
                np.random.uniform(self.bounds[2] + 5, self.bounds[3] - 5),
                np.random.uniform(1.0, 15.0)
            ], dtype=np.float32)
            radius = np.random.uniform(1.5, 3.5)
            self.obstacles.append({"pos": obs_pos, "radius": float(radius)})
        return self._get_obs()

    def _get_obs(self) -> dict[str, Any]:
        drones_obs = {}
        for i in range(self.num_agents):
            drone_id = f"drone_{i}"
            sensor_data = self.sensor_reader.read_sensors(
                drone_id=drone_id,
                true_pos=self.drone_positions[i],
                true_vel=self.drone_velocities[i],
                true_orientation=self.drone_orientations[i],
                battery=float(self.drone_batteries[i]),
                obstacles=self.obstacles
            )
            drones_obs[drone_id] = {
                "pos": self.drone_positions[i].copy(),
                "vel": self.drone_velocities[i].copy(),
                "orientation": self.drone_orientations[i].copy(),
                "battery": float(self.drone_batteries[i]),
                "sensor_data": sensor_data,
                "target_pos": self.target_positions[i].copy()
            }
        return {
            "drones": drones_obs,
            "obstacles": self.obstacles,
            "targets": {f"target_{i}": {"pos": self.target_positions[i].copy()} for i in range(self.num_agents)},
            "current_step": self.current_step
        }

    def step(self, actions: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float], bool, dict[str, Any]]:
        self.current_step += 1
        thrust_magnitudes = {}
        prev_distances = {}
        curr_distances = {}
        collisions = {f"drone_{i}": False for i in range(self.num_agents)}

        for i in range(self.num_agents):
            drone_id = f"drone_{i}"
            prev_distances[drone_id] = float(np.linalg.norm(self.drone_positions[i] - self.target_positions[i]))
            act = actions.get(drone_id, 0)
            if self.action_space_type == "discrete":
                idx = int(act) if int(act) < self.num_discrete_actions else 0
                if idx == self.brake_action_idx:
                    target_vel = -4.0 * self.drone_velocities[i]
                else:
                    target_vel = self.discrete_actions[idx]
            else:
                target_vel = np.clip(np.array(act, dtype=np.float32), -3.0, 3.0)
            thrust = float(np.linalg.norm(target_vel))
            thrust_magnitudes[drone_id] = thrust
            self.drone_batteries[i] = max(0.0, self.drone_batteries[i] - self.battery_drain_rate * (1.0 + thrust))
            self.drone_velocities[i] = 0.8 * self.drone_velocities[i] + 0.2 * target_vel
            if self.wind_perturbation > 0.0:
                wind = np.random.normal(0.0, self.wind_perturbation, size=3).astype(np.float32)
                self.drone_velocities[i] += wind
            self.drone_positions[i] += self.drone_velocities[i] * self.dt
            self.drone_positions[i, 0] = np.clip(self.drone_positions[i, 0], self.bounds[0], self.bounds[1])
            self.drone_positions[i, 1] = np.clip(self.drone_positions[i, 1], self.bounds[2], self.bounds[3])
            self.drone_positions[i, 2] = np.clip(self.drone_positions[i, 2], self.bounds[4], self.bounds[5])
            curr_distances[drone_id] = float(np.linalg.norm(self.drone_positions[i] - self.target_positions[i]))

        for i in range(self.num_agents):
            drone_id = f"drone_{i}"
            for j in range(i + 1, self.num_agents):
                if np.linalg.norm(self.drone_positions[i] - self.drone_positions[j]) < (2 * self.drone_radius):
                    collisions[drone_id] = True
                    collisions[f"drone_{j}"] = True
            for obs in self.obstacles:
                if np.linalg.norm(self.drone_positions[i] - obs["pos"]) < (self.drone_radius + obs["radius"]):
                    collisions[drone_id] = True
                    break

        all_reached = all(curr_distances[f"drone_{i}"] < self.reach_dist for i in range(self.num_agents))
        battery_depleted = any(b <= 0.0 for b in self.drone_batteries)
        done = (all_reached and self.all_targets_reached_term) or (self.current_step >= self.max_steps and self.timeout_term) or (battery_depleted and self.battery_depleted_term)

        rewards = self.reward_engine.calculate_rewards(
            prev_distances=prev_distances,
            curr_distances=curr_distances,
            collisions=collisions,
            thrusts=thrust_magnitudes,
            all_reached=all_reached
        )
        obs = self._get_obs()
        info = {
            "all_reached": all_reached,
            "collisions": collisions,
            "distances": curr_distances,
            "current_step": self.current_step,
            "thrust_magnitudes": thrust_magnitudes
        }
        return obs, rewards, done, info
