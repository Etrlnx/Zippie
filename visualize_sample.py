import argparse
import json
import tomllib
from pathlib import Path

import torch
from torch.distributions import Categorical

from airsim_interface.env import AirSimEnv
from explainability.attention_viz import AttentionVisualizer
from rl_engine.agent import MAPPOAgent
from state_processing.graph_builder import GraphBuilder


def main():
    parser = argparse.ArgumentParser(description="Run one episode with a trained checkpoint and save it as a demo sample.")
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_bc_finetune/best_model.pt")
    parser.add_argument("--seed", type=int, default=3, help="seed=3 with stochastic sampling is a verified success (all 3 drones reach target by step 389)")
    parser.add_argument("--deterministic", action="store_true", help="greedy argmax instead of stochastic sampling")
    parser.add_argument("--out", type=str, default="samples/bc_rl_demo")
    parser.add_argument("--frames", type=int, default=4, help="number of spatial-attention snapshots to save across the episode")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    env = AirSimEnv(config, seed=args.seed)
    gb = GraphBuilder(config)
    agent = MAPPOAgent(config, action_dim=env.num_discrete_actions)
    agent.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    agent.eval()

    viz = AttentionVisualizer(config)

    obs = env.reset()
    trajectories = {f"drone_{i}": [obs["drones"][f"drone_{i}"]["pos"].tolist()] for i in range(env.num_agents)}
    obstacles = [{"pos": o["pos"].tolist(), "radius": o["radius"]} for o in obs["obstacles"]]
    targets = {k: v["pos"].tolist() for k, v in obs["targets"].items()}

    done = False
    step = 0
    snapshots = []
    while not done:
        graph = gb.build_graph(obs)
        with torch.no_grad():
            logits, _, attn_weights = agent(graph["x"], graph["mask"], graph["drone_indices"], dist_matrix=graph["dist_matrix"])
        dist = Categorical(logits=logits.squeeze(0))
        actions = dist.probs.argmax(dim=-1) if args.deterministic else dist.sample()
        action_dict = {f"drone_{i}": int(actions[i]) for i in range(env.num_agents)}

        rollout = viz.extract_attention_rollout(attn_weights)
        snapshots.append((step, obs, rollout))

        obs, _, done, info = env.step(action_dict)
        step += 1
        for i in range(env.num_agents):
            trajectories[f"drone_{i}"].append(obs["drones"][f"drone_{i}"]["pos"].tolist())

    all_reached = info["all_reached"]
    mode = "deterministic" if args.deterministic else "stochastic"
    print(f"Episode finished: steps={step}  all_reached={all_reached}  seed={args.seed}  mode={mode}")

    n_snapshots = len(snapshots)
    frame_indices = sorted({round(i * (n_snapshots - 1) / max(args.frames - 1, 1)) for i in range(args.frames)})
    for idx in frame_indices:
        s, s_obs, s_rollout = snapshots[idx]
        viz.plot_spatial_attention(s_obs, graph["drone_indices"], s_rollout, save_path=str(out_dir / f"attention_step{s:04d}.png"))

    with open(out_dir / "episode.json", "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "seed": args.seed,
            "steps": step,
            "all_reached": all_reached,
            "obstacles": obstacles,
            "targets": targets,
            "trajectories": trajectories,
        }, f, indent=2)

    print(f"Saved {len(frame_indices)} attention snapshots + episode.json to {out_dir}/")


if __name__ == "__main__":
    main()
