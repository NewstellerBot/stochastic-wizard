import logging
import torch
import gymnasium as gym
import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import magic_cartpole

from typing import Literal
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import multiprocessing

CONFIDENCE_INTERVAL = 0.95
N_EVAL_EPISODES = 10
STEPS = np.arange(
    0, 50_001, 5_000
)  # [10_000, 15_000, 20_000, 40_000, 100_000, 200_000]
VERBOSE = 0
N_WIZARDS = 8
N_NORMIES = 1


def train_and_evaluate(model, env):
    total_steps = 0

    cis = []
    means = []

    for steps in STEPS:
        print(f"training {int(steps)}/{int(STEPS[-1])}")
        model.learn(total_timesteps=steps - total_steps)
        total_steps = steps
        print("evaluating...")
        mean, std = evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES)

        ci = sts.t.interval(
            CONFIDENCE_INTERVAL,
            N_EVAL_EPISODES - 1,
            loc=mean,
            scale=(std + 1e-7) / np.sqrt(N_EVAL_EPISODES),
        )

        means.append(mean)
        cis.append(ci)

    cis = np.array(cis)
    means = np.array(means)

    return means, cis


def run_experiment(
    *args,
    type: Literal["magic", "norm"],
    device: str = "cpu",
    label: str = None,
    **kwargs,
):
    if type == "magic":
        env = Monitor(gym.make("MagicCartPole", render_mode="rgb_array"))
    else:
        env = Monitor(gym.make("CartPole-v1", render_mode="rgb_array"))

    model = PPO("MlpPolicy", env, verbose=VERBOSE, device=device)
    means, cis = train_and_evaluate(model, env)

    label = label if label is not None else type

    return means, cis, label


def plot_results(results):

    fig, ax = plt.subplots(1, figsize=(15, 8))

    for means, cis, label in results:

        ax.plot(STEPS, means, label=label)
        ax.fill_between(
            STEPS,
            cis[:, 0],
            cis[:, 1],
            alpha=0.1,
            label=f"{int(CONFIDENCE_INTERVAL*100)}% ci",
        )

    ax.set_xlabel("Steps of training")
    ax.set_ylabel("Reward")

    plt.legend()
    fig.tight_layout()
    plt.show()


def run_experiment_wrapper(args):
    fn, kwargs = args
    return fn(**kwargs)


def main():
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # mps seems to work slower than cpu
    logging.info(f"Using {device}")

    num_cpus = multiprocessing.cpu_count() - 2
    logging.info(f"Using {num_cpus} processes")

    with ProcessPoolExecutor(num_cpus) as executor:
        experiment_args = [
            (run_experiment, {"type": "magic", "label": f"Wizard #{ix+1}"})
            for ix in range(N_WIZARDS)
        ] + [
            (run_experiment, {"type": "norm", "label": f"Normie#{ix+1}"})
            for ix in range(N_NORMIES)
        ]

        results = list(tqdm(executor.map(run_experiment_wrapper, experiment_args)))

    plot_results(results)


if __name__ == "__main__":
    main()
