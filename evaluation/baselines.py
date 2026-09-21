import time
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from evaluation.metrics import MetricEvaluator
from state_processing.graph_builder import GraphBuilder


class MLPBaselineAgent(nn.Module):
    def __init__(self, num_agents: int, action_dim: int = 7, node_dim: int = 16, max_nodes: int = 30, hidden_dim: int = 64):
        super().__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(max_nodes * node_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * action_dim),
        )

    def forward(self, x: torch.Tensor, mask=None, drone_indices=None, dist_matrix=None) -> tuple[torch.Tensor, torch.Tensor, list]:
        b = x.shape[0]
        logits = self.net(x.reshape(b, -1)).view(b, self.num_agents, self.action_dim)
        return logits, torch.zeros(b, 1), []


class CNNBaselineAgent(nn.Module):
    def __init__(self, num_agents: int, action_dim: int = 7, grid_size: int = 16, world_bounds=(-50.0, 50.0, -50.0, 50.0)):
        super().__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.grid_size = grid_size
        self.bounds = world_bounds
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Linear(16 * 4 * 4, num_agents * action_dim)

    def _to_grid(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        xmin, xmax, ymin, ymax = self.bounds
        gx = ((x[..., 0] - xmin) / (xmax - xmin) * (self.grid_size - 1)).long().clamp(0, self.grid_size - 1)
        gy = ((x[..., 1] - ymin) / (ymax - ymin) * (self.grid_size - 1)).long().clamp(0, self.grid_size - 1)
        grid = torch.zeros(b, 1, self.grid_size, self.grid_size, device=x.device)
        for bi in range(b):
            grid[bi, 0, gy[bi], gx[bi]] = 1.0
        return grid

    def forward(self, x: torch.Tensor, mask=None, drone_indices=None, dist_matrix=None) -> tuple[torch.Tensor, torch.Tensor, list]:
        feat = self.conv(self._to_grid(x)).reshape(x.shape[0], -1)
        logits = self.head(feat).view(x.shape[0], self.num_agents, self.action_dim)
        return logits, torch.zeros(x.shape[0], 1), []


class BaselineRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.graph_builder = GraphBuilder(config)
        self.evaluator = MetricEvaluator(config)

    def run_evaluation(self, agent: nn.Module, env, num_episodes: int = 10, deterministic: bool = False) -> dict[str, float]:
        agent.eval()
        logs = []
        for _ in range(num_episodes):
            logs.append(self._run_episode(agent, env, deterministic))
        return self.evaluator.evaluate_trajectories(logs)

    def _run_episode(self, agent: nn.Module, env, deterministic: bool) -> dict[str, Any]:
        obs = env.reset()
        num_agents = env.num_agents
        trajectories = {f"drone_{i}": [obs["drones"][f"drone_{i}"]["pos"].tolist()] for i in range(num_agents)}
        latencies_ms: list[float] = []
        total_collisions = 0
        total_energy = 0.0
        steps = 0
        done = False
        all_reached = False

        device = next(agent.parameters()).device
        while not done:
            graph = self.graph_builder.build_graph(obs)
            graph = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in graph.items()}
            start = time.perf_counter()
            with torch.no_grad():
                logits, _, _ = agent(graph["x"], graph["mask"], graph["drone_indices"], dist_matrix=graph["dist_matrix"])
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

            dist = Categorical(logits=logits.squeeze(0))
            actions = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            action_dict = {f"drone_{i}": int(actions[i]) for i in range(num_agents)}

            obs, _, done, info = env.step(action_dict)
            steps += 1
            total_collisions += sum(info["collisions"].values())
            total_energy += sum(info["thrust_magnitudes"].values())
            for i in range(num_agents):
                trajectories[f"drone_{i}"].append(obs["drones"][f"drone_{i}"]["pos"].tolist())
            all_reached = info["all_reached"]

        return {
            "all_reached": all_reached,
            "steps": steps,
            "total_collisions": total_collisions,
            "total_energy": total_energy,
            "trajectories": trajectories,
            "latencies_ms": latencies_ms,
        }
