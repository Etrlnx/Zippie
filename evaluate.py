import argparse
import tomllib
import torch
from airsim_interface.env import AirSimEnv
from rl_engine.agent import MAPPOAgent
from evaluation.baselines import MLPBaselineAgent, CNNBaselineAgent, BaselineRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.toml")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true", help="Use greedy argmax actions instead of stochastic sampling")
    parser.add_argument("--compare_baselines", action="store_true")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    env = AirSimEnv(config)
    action_dim = env.num_discrete_actions if env.action_space_type == "discrete" else 3
    runner = BaselineRunner(config)

    agent = MAPPOAgent(config, action_dim=action_dim)
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded weights from {args.checkpoint}")

    mode_str = "Deterministic (argmax)" if args.deterministic else "Stochastic (sampled)"
    print(f"--- Evaluating Proposed Graph Transformer MARL Policy [{mode_str}] ---")
    results_gt = runner.run_evaluation(agent, env, num_episodes=args.episodes, deterministic=args.deterministic)
    for k, v in results_gt.items():
        print(f"{k:>28}: {v:.4f}")

    if args.compare_baselines:
        print(f"\n--- Evaluating Baseline: MLP Policy [{mode_str}] ---")
        mlp_agent = MLPBaselineAgent(num_agents=env.num_agents, action_dim=action_dim)
        results_mlp = runner.run_evaluation(mlp_agent, env, num_episodes=args.episodes, deterministic=args.deterministic)
        for k, v in results_mlp.items():
            print(f"{k:>28}: {v:.4f}")

        print(f"\n--- Evaluating Baseline: CNN Policy [{mode_str}] ---")
        cnn_agent = CNNBaselineAgent(num_agents=env.num_agents, action_dim=action_dim)
        results_cnn = runner.run_evaluation(cnn_agent, env, num_episodes=args.episodes, deterministic=args.deterministic)
        for k, v in results_cnn.items():
            print(f"{k:>28}: {v:.4f}")


if __name__ == "__main__":
    main()