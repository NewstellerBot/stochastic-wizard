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

import magic_cartpole  # type: ignore

from typing import Literal
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import multiprocessing

CONFIDENCE_INTERVAL = 0.95
N_EVAL_EPISODES = 5
STEPS = np.arange(
    0, 200_001, 5_000
)  # [10_000, 15_000, 20_000, 40_000, 100_000, 200_000]
VERBOSE = 0
N_WIZARDS = 10
N_FIGHTERS = 10


def train_and_evaluate(model, env, eval_env: gym.Env = None):
    total_steps = 0

    cis = []
    means = []

    for steps in STEPS:
        model.learn(total_timesteps=steps - total_steps)
        total_steps = steps

        eval_env = eval_env if eval_env is not None else env
        mean, std = evaluate_policy(model, eval_env, n_eval_episodes=N_EVAL_EPISODES)

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
    *env_args,
    type: Literal["magic", "norm"],
    device: str = "cpu",
    label: str = None,
    gravity_mask_eval: bool = True,
    **env_kwargs,
):
    if type == "magic":
        env = Monitor(
            gym.make("MagicCartPole", *env_args, render_mode="rgb_array", **env_kwargs)
        )
        eval_env = Monitor(
            gym.make(
                "MagicCartPole",
                *env_args,
                render_mode="rgb_array",
                gravity_mask=gravity_mask_eval,
                **env_kwargs,
            )
        )
    else:
        env = Monitor(
            gym.make("CartPole-v1", *env_args, render_mode="rgb_array", **env_kwargs)
        )
        eval_env = None

    model = PPO("MlpPolicy", env, verbose=VERBOSE, device=device)
    means, cis = train_and_evaluate(model, env, eval_env)

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
        args = [
            (run_experiment, {"type": "magic", "gravity_mask_eval": True})
            for _ in range(N_WIZARDS)
        ] + [(run_experiment, {"type": "norm"}) for _ in range(N_FIGHTERS)]

        results = list(tqdm(executor.map(run_experiment_wrapper, args)))

    magic_means = np.array([mean for mean, _, _ in results[:N_WIZARDS]])
    fighter_means = np.array([mean for mean, _, _ in results[N_WIZARDS:]])

    magic_cis = np.array(
        [
            sts.t.interval(
                CONFIDENCE_INTERVAL,
                N_WIZARDS - 1,
                loc=mean,
                scale=(std + 1e-7) / np.sqrt(N_WIZARDS),
            )
            for mean, std in zip(magic_means.mean(0), magic_means.std(0))
        ]
    )

    fighter_cis = np.array(
        [
            sts.t.interval(
                CONFIDENCE_INTERVAL,
                N_FIGHTERS - 1,
                loc=mean,
                scale=(std + 1e-7) / np.sqrt(N_FIGHTERS),
            )
            for mean, std in zip(fighter_means.mean(0), fighter_means.std(0))
        ]
    )

    plt.plot(STEPS, magic_means.mean(0), label="Magic")
    plt.fill_between(STEPS, magic_cis[:, 0], magic_cis[:, 1], alpha=0.1)
    plt.plot(STEPS, fighter_means.mean(0), label="Fighter")
    plt.fill_between(STEPS, fighter_cis[:, 0], fighter_cis[:, 1], alpha=0.1)

    plt.xlabel("Steps of training")
    plt.ylabel("Reward")
    plt.title(
        f"Fighter v wizard with earth gravity (g = 9.81)\n ({N_WIZARDS} wizards, {N_FIGHTERS} fighters)"
    )

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
