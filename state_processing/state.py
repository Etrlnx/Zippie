from typing import Any
import numpy as np

class StateProcessor:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.node_dim = config.get("model", {}).get("node_dim", 16)
        world_bounds = config.get("simulation", {}).get("world_bounds", [-50.0, 50.0, -50.0, 50.0, 0.0, 20.0])
        self.pos_scale = max(abs(b) for b in world_bounds)
        self.vel_scale = 5.0
        self.sensor_range = config.get("environment", {}).get("sensor_range", 20.0)
        self.obstacle_sensor = config.get("model", {}).get("obstacle_sensor", False)

    def nearest_obstacle_vector(self, pos: np.ndarray, obstacles: list[dict[str, Any]]) -> np.ndarray:
        best, best_clear = np.zeros(3, dtype=np.float32), self.sensor_range
        for o in obstacles:
            delta = np.asarray(o["pos"], dtype=np.float32) - pos
            d = float(np.linalg.norm(delta))
            clear = d - o["radius"]
            if d > 0 and clear < best_clear:
                best_clear = clear
                best = delta / d * (1.0 - max(clear, 0.0) / self.sensor_range)
        return best

    def process_agent_state(self, drone_data: dict[str, Any], drone_idx: int, obstacles: list[dict[str, Any]] = ()) -> np.ndarray:
        pos = drone_data["pos"] / self.pos_scale
        vel = drone_data["vel"] / self.vel_scale
        ori = self.nearest_obstacle_vector(drone_data["pos"], obstacles) if self.obstacle_sensor else drone_data["orientation"]
        bat = drone_data["battery"]
        target_pos = drone_data.get("target_pos", np.zeros(3, dtype=np.float32)) / self.pos_scale
        rel_target = target_pos - pos
        feat = np.zeros(self.node_dim, dtype=np.float32)
        feat[0:3] = pos
        feat[3:6] = vel
        feat[6:9] = ori
        feat[9] = bat
        feat[10] = 0.0
        feat[11] = float(drone_idx)
        feat[12:15] = rel_target
        feat[15] = float(np.linalg.norm(rel_target))
        return feat

    def process_obstacle_state(self, obs_data: dict[str, Any], obs_idx: int) -> np.ndarray:
        pos = obs_data["pos"] / self.pos_scale
        radius = obs_data.get("radius", 1.0) / self.pos_scale
        feat = np.zeros(self.node_dim, dtype=np.float32)
        feat[0:3] = pos
        feat[9] = radius
        feat[10] = 1.0
        feat[11] = float(obs_idx)
        return feat

    def process_target_state(self, target_data: dict[str, Any], target_idx: int) -> np.ndarray:
        pos = target_data["pos"] / self.pos_scale
        feat = np.zeros(self.node_dim, dtype=np.float32)
        feat[0:3] = pos
        feat[10] = 2.0
        feat[11] = float(target_idx)
        return feat
