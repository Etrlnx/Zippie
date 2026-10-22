import argparse
import json
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from torch.distributions import Categorical

from airsim_interface.env import AirSimEnv
from bc_pretrain import oracle_action, straight_line_action
from explainability.attention_viz import AttentionVisualizer
from rl_engine.agent import MAPPOAgent
from state_processing.graph_builder import GraphBuilder

DRONE_COLORS = ["#1f77b4", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
TITLES = {
    "random": "Random moves",
    "straight": "Straight-line rule (ignores obstacles)",
    "teacher": "Teacher: look-ahead planner",
    "trained": "Trained model (Graph Transformer)",
}


def run_episode(name, config, seed, agent=None, deterministic=False, top_k=3):
    torch.manual_seed(seed)
    env = AirSimEnv(config, seed=seed)
    gb = GraphBuilder(config)
    viz = AttentionVisualizer(config)
    rng = np.random.default_rng(seed)
    obs = env.reset()
    episode = {
        "policy": name,
        "obstacles": [{"pos": o["pos"].tolist(), "radius": o["radius"]} for o in obs["obstacles"]],
        "targets": [obs["targets"][f"target_{i}"]["pos"].tolist() for i in range(env.num_agents)],
        "frames": [],
    }
    hit = [False] * env.num_agents
    arrived = [False] * env.num_agents
    done = False
    while True:
        attn = None
        if name == "trained":
            graph = gb.build_graph(obs)
            with torch.no_grad():
                logits, _, attn_weights = agent(graph["x"], graph["mask"], graph["drone_indices"], dist_matrix=graph["dist_matrix"])
            dist = Categorical(logits=logits.squeeze(0))
            actions = (dist.probs.argmax(dim=-1) if deterministic else dist.sample()).tolist()
            rollout = viz.extract_attention_rollout(attn_weights).numpy()
            attn = []
            for i in range(env.num_agents):
                row = rollout[i].copy()
                row[i] = 0.0
                order = np.argsort(-row)[:top_k]
                attn.append([[int(j), round(float(row[j]), 3)] for j in order])
        elif name == "random":
            actions = rng.integers(0, env.num_discrete_actions, env.num_agents).tolist()
        elif name == "straight":
            actions = [straight_line_action(env, obs, i, env.reach_dist) for i in range(env.num_agents)]
        else:
            actions = [oracle_action(env, obs, i, env.reach_dist) for i in range(env.num_agents)]
        episode["frames"].append({
            "pos": [[round(float(c), 2) for c in obs["drones"][f"drone_{i}"]["pos"]] for i in range(env.num_agents)],
            "hit": hit,
            "arrived": arrived,
            **({"attn": attn} if attn is not None else {}),
        })
        if done:
            break
        obs, _, done, info = env.step({f"drone_{i}": int(actions[i]) for i in range(env.num_agents)})
        hit = [bool(info["collisions"][f"drone_{i}"]) for i in range(env.num_agents)]
        arrived = [bool(info["distances"][f"drone_{i}"] < env.reach_dist) for i in range(env.num_agents)]
    episode["all_arrived"] = bool(info["all_reached"])
    return episode


def summarize(ep, upto):
    frames = ep["frames"][: upto + 1]
    n = len(frames[0]["pos"])
    crashes = sum(1 for k in range(1, len(frames)) for i in range(n) if frames[k]["hit"][i] and not frames[k - 1]["hit"][i])
    return sum(frames[-1]["arrived"]), n, crashes


def render_gif(episodes, bounds, path, stride, fps):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12.6))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.04, wspace=0.08, hspace=0.18)
    fig.suptitle("Three drones, each flying to its own star. Same map, four different pilots.", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.925, "Seen from above. Grey circles are obstacles at different heights; a drone can pass over or under one. "
             "Red flash = crash.", ha="center", fontsize=10, color="#444")
    artists = []
    for ax, ep in zip(axes.flat, episodes):
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(TITLES[ep["policy"]], fontsize=12, fontweight="bold")
        for o in ep["obstacles"]:
            shade = 0.35 + 0.45 * (1 - o["pos"][2] / bounds[5])
            ax.add_patch(Circle(o["pos"][:2], o["radius"], color=str(round(1 - shade, 2)), alpha=0.6, zorder=1))
        n = len(ep["targets"])
        for i, t in enumerate(ep["targets"]):
            ax.scatter(t[0], t[1], marker="*", s=420, color=DRONE_COLORS[i], edgecolors="black", zorder=2)
        trails = [ax.plot([], [], color=DRONE_COLORS[i], lw=2, alpha=0.7, zorder=3)[0] for i in range(n)]
        dots = ax.scatter([0] * n, [0] * n, s=140, c=DRONE_COLORS[:n], edgecolors="black", linewidths=1.5, zorder=4)
        status = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=11, zorder=5,
                         bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#999"})
        artists.append((ep, trails, dots, status))

    total = max(len(ep["frames"]) for ep in episodes)
    frame_ids = list(range(0, total, stride)) + [total - 1] * fps * 2

    def update(k):
        out = []
        for ep, trails, dots, status in artists:
            idx = min(k, len(ep["frames"]) - 1)
            hist = np.array([f["pos"] for f in ep["frames"][: idx + 1]])
            frame = ep["frames"][idx]
            for i, line in enumerate(trails):
                line.set_data(hist[:, i, 0], hist[:, i, 1])
            dots.set_offsets(hist[-1, :, :2])
            dots.set_edgecolors(["red" if h else "black" for h in frame["hit"]])
            dots.set_sizes([420 if h else 140 for h in frame["hit"]])
            arrived, n, crashes = summarize(ep, idx)
            finished = " (finished)" if idx == len(ep["frames"]) - 1 and idx < total - 1 else ""
            status.set_text(f"time {idx * 0.1:5.1f}s{finished}   arrived {arrived}/{n}   crashes {crashes}")
            out += [*trails, dots, status]
        return out

    FuncAnimation(fig, update, frames=frame_ids, blit=False).save(path, writer=PillowWriter(fps=fps), dpi=70)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Replay one map with four pilots side by side; writes the GIF plus the episodes.json/summary.json that web/ renders.")
    parser.add_argument("--config", type=str, default="configs/bc_dagger.toml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_bc_dagger/best_model.pt")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--out", type=str, default="samples/replay")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--no_gif", action="store_true")
    parser.add_argument("--summary_episodes", type=int, default=30)
    parser.add_argument("--note", type=str, default="", help="shown under the page intro, e.g. why this map was picked")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = MAPPOAgent(config, action_dim=AirSimEnv(config).num_discrete_actions)
    agent.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    agent.eval()

    episodes = [run_episode(name, config, args.seed, agent, args.deterministic) for name in TITLES]
    for ep in episodes:
        arrived, n, crashes = summarize(ep, len(ep["frames"]) - 1)
        print(f"{ep['policy']:>9}: steps={len(ep['frames']) - 1}  arrived={arrived}/{n}  crashes={crashes}")

    summary = []
    for name in TITLES:
        runs = [run_episode(name, config, 1000 + s, agent, args.deterministic, top_k=0) for s in range(args.summary_episodes)]
        summary.append({
            "policy": name,
            "any": float(np.mean([any(any(f["arrived"]) for f in ep["frames"]) for ep in runs])),
            "all": float(np.mean([ep["all_arrived"] for ep in runs])),
            "crash": float(np.mean([any(any(f["hit"]) for f in ep["frames"]) for ep in runs])),
        })
        print(f"{name:>9} over {args.summary_episodes} maps: {summary[-1]}", flush=True)

    bounds = config["simulation"]["world_bounds"]
    data = {"seed": args.seed, "checkpoint": args.checkpoint, "bounds": bounds, "dt": config["environment"]["dt"],
            "titles": TITLES, "summary_episodes": args.summary_episodes, "note": args.note, "episodes": episodes}
    with open(out_dir / "episodes.json", "w") as f:
        json.dump(data, f, separators=(",", ":"))
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if not args.no_gif:
        render_gif(episodes, bounds, str(out_dir / "replay.gif"), args.stride, args.fps)
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
