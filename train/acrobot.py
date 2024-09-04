import numpy as np

from stable_baselines3 import PPO

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import gymnasium as gym
import magic_env as _


from typing import Literal, Union

STEPS = np.arange(0, 50_001, 5_000)
N_FIGHTERS = 10
N_WIZARDS = 10
ALGO = PPO
CHECKPOINTS_PATH = lambda type, n, steps: f"./checkpoints/acrobot/{type}/{n}/{steps}"

torque_multiplier_perturbation = lambda: 0  # + np.random.normal(loc=0, scale=10)
torque_multiplier_init = lambda: 1

gravity_perturbation = lambda: 0  # + np.random.normal(loc=0, scale=10)
gravity_init = lambda: 9.8

# stochastic mask --> push_mask (2 actions), gravity_mask (3 actions), force_mask (3 actions)
wizard_stochastic_mask = [1 / 9] * 9
fighter_stochastic_mask = [1 / 9] * 9  # + [0.0] * 6


def make_env(torque_multiplier_mask=False, gravity_mask=False):
    return gym.make(
        "magic_env/MagicAcrobot",
        render_mode="rgb_array",
        # torque_multiplier
        torque_multiplier_init=torque_multiplier_init,
        torque_multiplier_perturbation=torque_multiplier_perturbation,
        torque_multiplier_mask=torque_multiplier_mask,
        # Gravity
        gravity_init=gravity_init,
        gravity_perturbation=gravity_perturbation,
        gravity_mask=gravity_mask,
        # Stochastic mask
        stochastic_mask=wizard_stochastic_mask,
    )


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
        env = make_env(True, True)

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
            env = make_env(True, True)
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
    mask_steps = 20_000

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
