import argparse
import random
import tomllib

import numpy as np
import torch

from airsim_interface.env import AirSimEnv
from rl_engine.agent import MAPPOAgent
from rl_engine.trainer import MAPPOTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_progress(iteration: int, total: int, global_step: int, total_timesteps: int, mean_rew: float, elapsed: float):
    pct = iteration / total * 100
    step_pct = global_step / total_timesteps * 100
    bar_len = 40
    filled = int(bar_len * iteration / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    it_per_sec = iteration / elapsed if elapsed > 0 else 0
    eta = (total - iteration) / it_per_sec if it_per_sec > 0 else 0
    print(f"\rIter [{iteration:>4}/{total}] {bar} {pct:5.1f}% | Steps [{global_step:>7}/{total_timesteps}] {step_pct:5.1f}% | Reward: {mean_rew:>7.3f} | {it_per_sec:.2f} it/s | ETA: {eta:.0f}s", end="", flush=True)
    if iteration == total:
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.toml")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    env = AirSimEnv(config, seed=seed)
    action_dim = env.num_discrete_actions if env.action_space_type == "discrete" else 3
    agent = MAPPOAgent(config, action_dim=action_dim)
    trainer = MAPPOTrainer(config, env, agent)
    
    max_iterations = config.get("rl", {}).get("max_iterations", 100)
    total_timesteps = config.get("rl", {}).get("total_timesteps", 1000000)
    
    print(f"Starting training: {max_iterations} iterations, {total_timesteps} total timesteps")
    print(f"n_steps per iteration: {config.get('rl', {}).get('n_steps', 2048)}")
    print("-" * 80)
    
    trainer.train_with_progress(max_iterations, print_progress)


if __name__ == "__main__":
    main()