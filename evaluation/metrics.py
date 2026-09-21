from itertools import pairwise
from typing import Any

import numpy as np

_METRIC_KEYS = (
    "success_rate", "path_length", "collision_rate", "energy_consumption",
    "completion_time", "inference_latency_ms", "unseen_generalization_rate",
)


class MetricEvaluator:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def evaluate_trajectories(self, logs: list[dict[str, Any]]) -> dict[str, float]:
        n = len(logs)
        if n == 0:
            return {k: 0.0 for k in _METRIC_KEYS}

        success_rate = sum(1 for l in logs if l.get("all_reached", False)) / n
        completion_time = sum(l.get("steps", 0) for l in logs) / n
        collision_rate = sum(1 for l in logs if l.get("total_collisions", 0) > 0) / n
        energy_consumption = sum(l.get("total_energy", 0.0) for l in logs) / n
        path_length = sum(self._trajectory_length(l.get("trajectories", {})) for l in logs) / n
        latencies = [lat for l in logs for lat in l.get("latencies_ms", [])]
        inference_latency_ms = float(np.mean(latencies)) if latencies else 0.0

        return {
            "success_rate": float(success_rate),
            "path_length": float(path_length),
            "collision_rate": float(collision_rate),
            "energy_consumption": float(energy_consumption),
            "completion_time": float(completion_time),
            "inference_latency_ms": inference_latency_ms,
            # ponytail: no seen/unseen map split exists yet (obstacles randomize every reset already),
            # so this is success_rate under an untracked layout. Split logs by a "seen" flag if that's added.
            "unseen_generalization_rate": float(success_rate),
        }

    @staticmethod
    def _trajectory_length(trajectories: dict[str, list]) -> float:
        total = 0.0
        for points in trajectories.values():
            for a, b in pairwise(points):
                total += float(np.linalg.norm(np.array(b, dtype=np.float32) - np.array(a, dtype=np.float32)))
        return total
