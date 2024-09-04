import numpy as np

from stable_baselines3 import PPO

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import gymnasium as gym

import magic_env as _

from typing import Literal, Union

STEPS = np.arange(0, 400_001, 25_000)
N_FIGHTERS = 5
N_WIZARDS = 5
ALGO = PPO


def CHECKPOINTS_PATH(type, n, steps):
    return f"./checkpoints/mujoco/ant/{type}/{n}/{steps}"


def make_env(magic_mask: bool = False):
    return gym.make("magic_env/Ant", render_mode="rgb_array", magic_mask=magic_mask)


def train_agent(
    n: int,
    type: Literal["wizard", "fighter"],
    device: str,
    mask_steps: Union[int, None] = None,
):
    """
    Trains agents on the magic cartpole environment.
    ---
    n: Agent number for bookkeeping purposes and saving checkpoints
    type: Agent type--wizard or fighter
    device: Torch device on which to run the training
    mask_steps: Number of steps after which to start applying masks on wizard
    """

    if type == "wizard":
        env = make_env()
    else:
        env = make_env(True)

    agent = ALGO("MlpPolicy", env, device=device)
    last = 0
    masked = False

    for ix, steps in enumerate(STEPS):
        if (
            mask_steps is not None
            and steps >= mask_steps
            and type == "wizard"
            and not masked
        ):
            env = make_env(magic_mask=True)
            agent = ALGO.load(CHECKPOINTS_PATH(type, n, STEPS[ix - 1]), env)

        agent.learn(steps - last)
        last = steps
        agent.save(CHECKPOINTS_PATH(type, n, steps))


def wrapper(args):
    fn, *args = args
    return fn(*args)


if __name__ == "__main__":
    print("training wizrd...")

    device = "cpu"
    n_cpu = multiprocessing.cpu_count() - 2
    mask_steps = 80_000

    with ProcessPoolExecutor(n_cpu) as executor:
        print("start training...")
        # wizard
        wizard_args = [
            (train_agent, n, "wizard", device, mask_steps) for n in range(N_WIZARDS)
        ]

        # fighter
        fighter_args = [(train_agent, n, "fighter", device) for n in range(N_FIGHTERS)]

        args = wizard_args + fighter_args
        res = list(executor.map(wrapper, args))
