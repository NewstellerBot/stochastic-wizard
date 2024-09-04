import numpy as np

from stable_baselines3 import PPO

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import gymnasium as gym
import magic_cartpole

from typing import Literal

from tqdm import tqdm

STEPS = np.arange(0, 250_001, 5_000)
N_FIGHTERS = 10
N_WIZARDS = 10


def train_agent(n, type: Literal["wizard", "fighter"], device):

    length_perturbation = lambda: 0  # + np.random.normal(loc=0, scale=10)
    length_init = lambda: 0.5  # -9.81 + np.random.normal(loc=0, scale=1)

    if type == "wizard":
        env = gym.make(
            "MagicCartPole/length",
            render_mode="rgb_array",
            length_init=length_init,
            length_perturbation=length_perturbation,
        )
    else:
        env = gym.make(
            "MagicCartPole/length",
            render_mode="rgb_array",
            length_mask=True,
            length_init=length_init,
            length_perturbation=length_perturbation,
        )

    agent = PPO("MlpPolicy", env, device=device)
    last = 0

    for steps in STEPS:
        agent.learn(steps - last)
        last = steps
        agent.save(f"./checkpoints/length/{type}/{n}/{steps}")


def wrapper(args):
    fn, *args = args
    return fn(*args)


if __name__ == "__main__":
    print("training wizrd...")

    device = "cpu"
    n_cpu = multiprocessing.cpu_count() - 2

    with ProcessPoolExecutor(n_cpu) as executor:

        print("training wizard...")
        # wizard
        args = [(train_agent, n, "wizard", device) for n in range(N_WIZARDS)]
        res = list(executor.map(wrapper, args))

        print("training fighter...")
        # fighter
        args = [(train_agent, n, "fighter", device) for n in range(N_FIGHTERS)]
        executor.map(wrapper, args)
