import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from state_processing.graph_builder import GraphBuilder


class RolloutBuffer:
    def __init__(self, n_steps: int, num_agents: int):
        self.n_steps = n_steps
        self.num_agents = num_agents
        self.graphs: list[dict[str, Any]] = []
        self.actions = torch.zeros(n_steps, num_agents, dtype=torch.long)
        self.log_probs = torch.zeros(n_steps, num_agents)
        self.values = torch.zeros(n_steps)
        self.rewards = torch.zeros(n_steps, num_agents)
        self.dones = torch.zeros(n_steps)
        self.ptr = 0

    def add(self, graph: dict[str, Any], actions: torch.Tensor, log_probs: torch.Tensor, value: torch.Tensor, rewards: torch.Tensor, done: bool):
        self.graphs.append(graph)
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.values[self.ptr] = value
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = float(done)
        self.ptr += 1


class MAPPOTrainer:
    def __init__(self, config: dict[str, Any], env, agent):
        self.config = config
        self.env = env
        self.agent = agent
        rl_cfg = config.get("rl", {})
        exp_cfg = config.get("experiment", {})

        self.n_steps = rl_cfg.get("n_steps", 2048)
        self.batch_size = rl_cfg.get("batch_size", 64)
        self.n_epochs = rl_cfg.get("n_epochs", 10)
        self.gamma = rl_cfg.get("gamma", 0.99)
        self.gae_lambda = rl_cfg.get("gae_lambda", 0.95)
        self.clip_range = rl_cfg.get("clip_range", 0.2)
        self.learning_rate = rl_cfg.get("learning_rate", 3e-4)
        self.entropy_coef = rl_cfg.get("entropy_coef", 0.01)
        self.entropy_coef_final = rl_cfg.get("entropy_coef_final", 0.001)
        self.max_grad_norm = rl_cfg.get("max_grad_norm", 0.5)
        self.target_kl = rl_cfg.get("target_kl", 0.01)
        self.total_timesteps = rl_cfg.get("total_timesteps", 1_000_000)

        self.checkpoint_dir = Path(exp_cfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir = exp_cfg.get("log_dir", "logs")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        device_cfg = exp_cfg.get("device", "auto")
        if device_cfg == "auto":
            device_cfg = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_cfg)
        self.agent.to(self.device)

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.learning_rate)
        self.graph_builder = GraphBuilder(config)
        self.global_step = 0
        self.best_mean_reward = float("-inf")
        self._obs = self.env.reset()

    def _to_device(self, graph: dict[str, Any]) -> dict[str, Any]:
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in graph.items()}

    @torch.no_grad()
    def _act(self, graph: dict[str, Any]):
        g = self._to_device(graph)
        logits, value, _ = self.agent(g["x"], g["mask"], g["drone_indices"], dist_matrix=g["dist_matrix"])
        dist = Categorical(logits=logits.squeeze(0))
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions.cpu(), log_probs.cpu(), value.squeeze().cpu()

    def collect_rollout(self) -> RolloutBuffer:
        buf = RolloutBuffer(self.n_steps, self.env.num_agents)
        for _ in range(self.n_steps):
            graph = self.graph_builder.build_graph(self._obs)
            actions, log_probs, value = self._act(graph)
            action_dict = {f"drone_{i}": int(actions[i]) for i in range(self.env.num_agents)}
            next_obs, rewards, done, _ = self.env.step(action_dict)
            per_agent_rewards = torch.tensor([rewards[f"drone_{i}"] for i in range(self.env.num_agents)], dtype=torch.float32)
            buf.add(graph, actions, log_probs, value, per_agent_rewards, done)
            self._obs = self.env.reset() if done else next_obs
            self.global_step += 1
        return buf

    def _bootstrap_value(self) -> torch.Tensor:
        graph = self.graph_builder.build_graph(self._obs)
        with torch.no_grad():
            g = self._to_device(graph)
            _, value, _ = self.agent(g["x"], g["mask"], g["drone_indices"], dist_matrix=g["dist_matrix"])
        return value.squeeze().cpu()

    def _compute_gae(self, buf: RolloutBuffer):
        next_value = self._bootstrap_value()
        values = torch.cat([buf.values, next_value.unsqueeze(0)])
        advantages = torch.zeros(buf.n_steps, buf.num_agents)
        last_gae = torch.zeros(buf.num_agents)
        for t in reversed(range(buf.n_steps)):
            mask = 1.0 - buf.dones[t]
            delta = buf.rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            advantages[t] = last_gae
        returns = advantages.mean(dim=1) + buf.values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def _update(self, buf: RolloutBuffer, advantages: torch.Tensor, returns: torch.Tensor) -> dict[str, float]:
        progress = min(self.global_step / self.total_timesteps, 1.0)
        lr = self.learning_rate * (1.0 - progress)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        entropy_coef = self.entropy_coef + (self.entropy_coef_final - self.entropy_coef) * progress

        indices = np.arange(buf.n_steps)
        stats: dict[str, float] = {}
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            approx_kls = []
            for start in range(0, buf.n_steps, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                xs = torch.cat([buf.graphs[i]["x"] for i in batch_idx], dim=0).to(self.device)
                masks = torch.cat([buf.graphs[i]["mask"] for i in batch_idx], dim=0).to(self.device)
                dists = torch.cat([buf.graphs[i]["dist_matrix"] for i in batch_idx], dim=0).to(self.device)
                drone_indices = buf.graphs[batch_idx[0]]["drone_indices"]

                logits_b, values_b, _ = self.agent(xs, masks, drone_indices, dist_matrix=dists)
                values_b = values_b.squeeze(-1)

                actions_b = buf.actions[batch_idx].to(self.device)
                old_log_probs_b = buf.log_probs[batch_idx].to(self.device)
                adv_b = advantages[batch_idx].to(self.device)
                ret_b = returns[batch_idx].to(self.device)

                dist = Categorical(logits=logits_b)
                new_log_probs = dist.log_prob(actions_b)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values_b, ret_b)
                loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                approx_kls.append((old_log_probs_b - new_log_probs).mean().item())
                stats = {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(), "entropy": entropy.item(), "lr": lr}
            if approx_kls and np.mean(approx_kls) > self.target_kl:
                break
        return stats

    def train_with_progress(self, max_iterations: int, progress_callback: Any | None = None):
        start = time.time()
        for iteration in range(1, max_iterations + 1):
            buf = self.collect_rollout()
            advantages, returns = self._compute_gae(buf)
            stats = self._update(buf, advantages, returns)
            mean_reward = buf.rewards.mean().item()

            self.writer.add_scalar("reward/mean", mean_reward, iteration)
            for k, v in stats.items():
                self.writer.add_scalar(f"train/{k}", v, iteration)

            torch.save(self.agent.state_dict(), self.checkpoint_dir / f"model_iter_{iteration}.pt")
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                torch.save(self.agent.state_dict(), self.checkpoint_dir / "best_model.pt")

            if progress_callback is not None:
                progress_callback(iteration, max_iterations, self.global_step, self.total_timesteps, mean_reward, time.time() - start)

        self.writer.close()

    def train(self, max_iterations: int):
        self.train_with_progress(max_iterations, progress_callback=None)
