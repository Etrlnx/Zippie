import argparse
import tomllib

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from airsim_interface.env import AirSimEnv
from rl_engine.agent import MAPPOAgent
from state_processing.graph_builder import GraphBuilder


def straight_line_action(env: AirSimEnv, obs: dict, i: int, brake_threshold: float) -> int:
    delta = obs["targets"][f"target_{i}"]["pos"] - obs["drones"][f"drone_{i}"]["pos"]
    if float(np.linalg.norm(delta)) <= brake_threshold:
        return env.brake_action_idx
    return int(np.argmax([np.dot(a, delta) for a in env.discrete_actions]))


def oracle_action(env: AirSimEnv, obs: dict, i: int, brake_threshold: float, horizon: int = 15, margin: float = 1.0) -> int:
    drone = obs["drones"][f"drone_{i}"]
    pos, vel = drone["pos"], drone["vel"]
    target_pos = obs["targets"][f"target_{i}"]["pos"]
    if float(np.linalg.norm(target_pos - pos)) <= brake_threshold:
        return env.brake_action_idx
    obs_pos = np.array([o["pos"] for o in obs["obstacles"]]).reshape(-1, 3)
    obs_rad = np.array([o["radius"] for o in obs["obstacles"]]) + env.drone_radius
    others = [obs["drones"][f"drone_{j}"] for j in range(env.num_agents) if j != i]
    best_idx, best_score = env.brake_action_idx, -1e9
    for idx, action_vec in enumerate(env.discrete_actions):
        p, v = pos.copy(), vel.copy()
        clearance = np.inf
        for k in range(1, horizon + 1):
            v = 0.8 * v + 0.2 * action_vec
            p = p + v * env.dt
            if len(obs_pos):
                clearance = min(clearance, float((np.linalg.norm(obs_pos - p, axis=1) - obs_rad).min()))
            for o in others:
                clearance = min(clearance, float(np.linalg.norm(o["pos"] + o["vel"] * env.dt * k - p)) - 2 * env.drone_radius)
        score = -float(np.linalg.norm(target_pos - p)) - 100.0 * max(0.0, margin - clearance)
        if score > best_score:
            best_score, best_idx = score, idx
    return best_idx


def collect_demonstrations(env: AirSimEnv, gb: GraphBuilder, episodes: int, brake_threshold: float, student: MAPPOAgent | None = None):
    graphs, actions, avoid = [], [], []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            graph = gb.build_graph(obs)
            labels = [oracle_action(env, obs, i, brake_threshold) for i in range(env.num_agents)]
            graphs.append(graph)
            actions.append(labels)
            avoid.append([labels[i] != straight_line_action(env, obs, i, brake_threshold) for i in range(env.num_agents)])
            if student is None:
                taken = labels
            else:
                with torch.no_grad():
                    logits, _, _ = student(graph["x"], graph["mask"], graph["drone_indices"], dist_matrix=graph["dist_matrix"])
                taken = Categorical(logits=logits.squeeze(0)).sample().tolist()
            obs, _, done, _ = env.step({f"drone_{i}": int(taken[i]) for i in range(env.num_agents)})
        if (ep + 1) % 10 == 0:
            print(f"  collected {ep + 1}/{episodes} episodes ({len(graphs)} samples so far)", flush=True)
    return graphs, torch.tensor(actions, dtype=torch.long), torch.tensor(avoid, dtype=torch.bool)


def fit(agent, optimizer, graphs, actions, avoid, epochs, batch_size, avoid_weight):
    agent.train()
    action_dim = agent.actor[-1].out_features
    weights = 1.0 + (avoid_weight - 1.0) * avoid.float()
    n = len(graphs)
    indices = np.arange(n)
    for epoch in range(epochs):
        np.random.shuffle(indices)
        total_loss, total_correct, total_count, avoid_correct, avoid_count = 0.0, 0, 0, 0, 0
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            xs = torch.cat([graphs[i]["x"] for i in batch_idx], dim=0)
            masks = torch.cat([graphs[i]["mask"] for i in batch_idx], dim=0)
            dists = torch.cat([graphs[i]["dist_matrix"] for i in batch_idx], dim=0)
            drone_indices = graphs[batch_idx[0]]["drone_indices"]
            targets = actions[batch_idx]
            w = weights[batch_idx]

            logits, _, _ = agent(xs, masks, drone_indices, dist_matrix=dists)
            per_sample = nn.functional.cross_entropy(logits.reshape(-1, action_dim), targets.reshape(-1), reduction="none")
            loss = (per_sample * w.reshape(-1)).sum() / w.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct = logits.argmax(dim=-1) == targets
            total_loss += loss.item() * len(batch_idx)
            total_correct += correct.sum().item()
            total_count += targets.numel()
            avoid_correct += correct[avoid[batch_idx]].sum().item()
            avoid_count += avoid[batch_idx].sum().item()

        print(f"epoch {epoch + 1}/{epochs}  loss={total_loss / n:.4f}  action_accuracy={total_correct / total_count:.1%}  "
              f"avoid_accuracy={avoid_correct / max(avoid_count, 1):.1%} ({avoid_count} avoid labels)", flush=True)
    agent.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--brake_threshold_mult", type=float, default=1.0)
    parser.add_argument("--init", type=str, default="", help="start from these weights instead of a fresh agent")
    parser.add_argument("--dagger_rounds", type=int, default=0, help="rounds of student rollouts relabeled by the teacher")
    parser.add_argument("--dagger_episodes", type=int, default=30)
    parser.add_argument("--dagger_epochs", type=int, default=3)
    parser.add_argument("--avoid_weight", type=float, default=1.0, help="loss weight on states where the teacher deviates from the straight-line move")
    parser.add_argument("--out", type=str, default="checkpoints/bc_pretrained.pt")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    env = AirSimEnv(config, seed=config.get("experiment", {}).get("seed", 42))
    gb = GraphBuilder(config)
    brake_threshold = args.brake_threshold_mult * env.reach_dist

    print(f"Collecting {args.episodes} demonstration episodes (brake_threshold={brake_threshold:.2f})...", flush=True)
    graphs, actions, avoid = collect_demonstrations(env, gb, args.episodes, brake_threshold)
    print(f"Collected {len(graphs)} (state, action) samples.", flush=True)

    agent = MAPPOAgent(config, action_dim=env.num_discrete_actions)
    if args.init:
        agent.load_state_dict(torch.load(args.init, map_location="cpu"))
        print(f"Initialized from {args.init}", flush=True)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    fit(agent, optimizer, graphs, actions, avoid, args.epochs, args.batch_size, args.avoid_weight)

    for r in range(args.dagger_rounds):
        print(f"DAgger round {r + 1}/{args.dagger_rounds}: student flies, teacher labels", flush=True)
        g2, a2, v2 = collect_demonstrations(env, gb, args.dagger_episodes, brake_threshold, student=agent)
        graphs += g2
        actions, avoid = torch.cat([actions, a2]), torch.cat([avoid, v2])
        fit(agent, optimizer, graphs, actions, avoid, args.dagger_epochs, args.batch_size, args.avoid_weight)
        torch.save(agent.state_dict(), args.out)

    torch.save(agent.state_dict(), args.out)
    print(f"Saved BC-pretrained weights to {args.out}", flush=True)


if __name__ == "__main__":
    main()
