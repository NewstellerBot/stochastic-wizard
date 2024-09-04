import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import magic_env as _

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from typing import Literal

from train.cartpole.both import STEPS, N_FIGHTERS, N_WIZARDS, ALGO
from utils import calculate_confidence_interval

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

CONFIDENCE_INTERVAL = 0.95


def test_agent(
    n: int,
    kind: Literal["wizard", "fighter"],
):

    gravity_perturbation = lambda: 0 + np.random.normal(loc=0, scale=1)
    gravity_init = lambda: 9.81 + np.random.normal(loc=0, scale=1)

    length_init = lambda: 0.5
    length_perturbations = lambda: np.random.normal(loc=0, scale=0.1)

    env = Monitor(
        gym.make(
            "MagicCartPole/both",
            render_mode="rgb_array",
            # Gravity mask
            gravity_mask=True,
            gravity_perturbation=gravity_perturbation,
            gravity_init=gravity_init,
            # Lenght mask
            length_mask=True,
            length_init=length_init,
            length_perturbation=length_perturbations,
        )
    )

    res = []

    for steps in STEPS:
        agent = ALGO.load(f"./checkpoints/both/{kind}/{n}/{steps}")
        mean, _ = evaluate_policy(agent, env, 1)

        res.append(mean)

    return res


def wrapper(args):
    fn, *args = args
    return fn(*args)


if __name__ == "__main__":
    args = [(test_agent, n, "wizard") for n in range(N_WIZARDS)] + [
        (test_agent, n, "fighter") for n in range(N_FIGHTERS)
    ]

    n_cpu = multiprocessing.cpu_count() - 1
    with ProcessPoolExecutor(n_cpu) as executor:
        results = list(tqdm(executor.map(wrapper, args)))

    wizard = np.array(results[:N_WIZARDS])
    fighter = np.array(results[N_WIZARDS:])

    wizard_ci = calculate_confidence_interval(wizard, CONFIDENCE_INTERVAL, N_WIZARDS)
    fighter_ci = calculate_confidence_interval(fighter, CONFIDENCE_INTERVAL, N_FIGHTERS)

    wizard_ci = np.array(wizard_ci)
    fighter_ci = np.array(fighter_ci)

    plt.plot(STEPS, wizard.mean(0), label="wizard")
    plt.fill_between(STEPS, wizard_ci[:, 0], wizard_ci[:, 1], alpha=0.1)

    plt.plot(STEPS, fighter.mean(0), label="fighter")
    plt.fill_between(STEPS, fighter_ci[:, 0], fighter_ci[:, 1], alpha=0.1)

    plt.xlabel("Steps of training")
    plt.ylabel("Reward")

    plt.legend()
    plt.show()
