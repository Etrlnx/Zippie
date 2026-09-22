import argparse
import tomllib

import numpy as np
import torch
import torch.nn as nn

from airsim_interface.env import AirSimEnv
from rl_engine.agent import MAPPOAgent
from state_processing.graph_builder import GraphBuilder


def oracle_action(env: AirSimEnv, pos: np.ndarray, target_pos: np.ndarray, brake_threshold: float) -> int:
    delta = target_pos - pos
    dist = float(np.linalg.norm(delta))
    if dist <= brake_threshold:
        return env.brake_action_idx
    best_idx, best_score = 0, -1e9
    for idx, action_vec in enumerate(env.discrete_actions):
        score = float(np.dot(action_vec, delta))
        if score > best_score:
            best_score, best_idx = score, idx
    return best_idx


def collect_demonstrations(env: AirSimEnv, gb: GraphBuilder, episodes: int, brake_threshold: float):
    graphs, actions = [], []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            graph = gb.build_graph(obs)
            ep_actions = []
            action_dict = {}
            for i in range(env.num_agents):
                drone_id = f"drone_{i}"
                pos = obs["drones"][drone_id]["pos"]
                target_pos = obs["targets"][f"target_{i}"]["pos"]
                a = oracle_action(env, pos, target_pos, brake_threshold)
                ep_actions.append(a)
                action_dict[drone_id] = a
            graphs.append(graph)
            actions.append(ep_actions)
            obs, _, done, _ = env.step(action_dict)
        if (ep + 1) % 50 == 0:
            print(f"  collected {ep + 1}/{episodes} episodes ({len(graphs)} samples so far)", flush=True)
    return graphs, torch.tensor(actions, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--brake_threshold_mult", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="checkpoints/bc_pretrained.pt")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    env = AirSimEnv(config, seed=config.get("experiment", {}).get("seed", 42))
    gb = GraphBuilder(config)
    brake_threshold = args.brake_threshold_mult * env.reach_dist

    print(f"Collecting {args.episodes} demonstration episodes (brake_threshold={brake_threshold:.2f})...", flush=True)
    graphs, actions = collect_demonstrations(env, gb, args.episodes, brake_threshold)
    print(f"Collected {len(graphs)} (state, action) samples.", flush=True)

    action_dim = env.num_discrete_actions
    agent = MAPPOAgent(config, action_dim=action_dim)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    n = len(graphs)
    indices = np.arange(n)
    for epoch in range(args.epochs):
        np.random.shuffle(indices)
        total_loss, total_correct, total_count = 0.0, 0, 0
        for start in range(0, n, args.batch_size):
            batch_idx = indices[start:start + args.batch_size]
            xs = torch.cat([graphs[i]["x"] for i in batch_idx], dim=0)
            masks = torch.cat([graphs[i]["mask"] for i in batch_idx], dim=0)
            dists = torch.cat([graphs[i]["dist_matrix"] for i in batch_idx], dim=0)
            drone_indices = graphs[batch_idx[0]]["drone_indices"]
            targets = actions[batch_idx]

            logits, _, _ = agent(xs, masks, drone_indices, dist_matrix=dists)
            loss = criterion(logits.reshape(-1, action_dim), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_idx)
            total_correct += (logits.argmax(dim=-1) == targets).sum().item()
            total_count += targets.numel()

        print(f"epoch {epoch + 1}/{args.epochs}  loss={total_loss / n:.4f}  action_accuracy={total_correct / total_count:.1%}", flush=True)

    torch.save(agent.state_dict(), args.out)
    print(f"Saved BC-pretrained weights to {args.out}", flush=True)


if __name__ == "__main__":
    main()
