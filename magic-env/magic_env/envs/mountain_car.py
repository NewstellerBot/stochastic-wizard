"""
http://incompleteideas.net/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math
from typing import Optional, Callable

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


FORCE_ACTION_SCALING = 10 ** (-4)
GRAVITY_ACTION_SCALING = 10 ** (-4)

# original force = 0.001
# original gravity = 0.0025
FORCE_BOUNDRIES = (0.0, 0.05)
GRAVITY_BOUNDRIES = (0.0, 0.01)


class MountainCarEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        goal_velocity=0,
        # Force
        force_mask: bool = False,
        force_init: Callable = None,
        force_perturbation: Callable = None,
        # Gravity
        gravity_mask: bool = False,
        gravity_init: Callable = None,
        gravity_perturbation: Callable = None,
        # stochastic mask
        stochastic_mask: Optional[list] = None,
    ):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity

        self.force_init = force_init if force_init is not None else lambda: 0.001
        self.force_mask = force_mask
        self.force_perturbation = (
            force_perturbation if force_perturbation is not None else lambda: 0
        )

        self.gravity_init = gravity_init if gravity_init is not None else lambda: 0.0025
        self.gravity_mask = gravity_mask
        self.gravity_perturbation = (
            gravity_perturbation if gravity_perturbation is not None else lambda: 0
        )

        self.force = self.force_init()
        self.gravity = self.gravity_init()

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        self.action_space = spaces.MultiDiscrete([3, 3, 3])
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.stochastic_mask = stochastic_mask if stochastic_mask else None
        if self.stochastic_mask is not None:
            # Make sure stochastic mask has appropriate length
            assert len(self.stochastic_mask) == sum(
                self.action_space.nvec
            ), "Stochastic mask has to cover whole action space!"

    def step(self, action: int):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        action_mask = False
        masks = [action_mask, self.force_mask, self.gravity_mask]

        # Applying stochastic mask
        if self.stochastic_mask is not None:
            mask = np.random.multinomial(1, self.stochastic_mask)
            action_to_mask = np.where(mask == 1)[0][0]
            bins = np.cumsum(self.action_space.nvec)
            action_group_to_mask = np.digitize(action_to_mask, bins)
            action_to_mask = max(0, action_to_mask - bins[action_group_to_mask - 1])
            if action[action_group_to_mask] == action_to_mask:
                masks[action_group_to_mask] = True

        action_mask, force_mask, gravity_mask = masks
        action, force_action, gravity_action = action

        force_change = 0 if force_mask else force_action * FORCE_ACTION_SCALING
        self.force += force_change + self.force_perturbation()
        self.force = np.clip(self.force, *FORCE_BOUNDRIES)

        gravity_change = 0 if gravity_mask else gravity_action * GRAVITY_ACTION_SCALING
        self.gravity += gravity_change + self.gravity_perturbation()
        self.gravity = np.clip(self.gravity, *GRAVITY_BOUNDRIES)

        position, velocity = self.state
        velocity += (
            (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        ) * int(not action_mask)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        terminated = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )
        reward = -1.0

        self.state = (position, velocity)
        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        self.state = np.array([self.np_random.uniform(low=low, high=high), 0])

        # reset the force and gravity to default values
        self.force = self.force_init()
        self.gravity = self.gravity_init()

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.max_position - self.min_position
        scale = self.screen_width / world_width
        carwidth = 40
        carheight = 20

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        pos = self.state[0]

        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale))

        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

        clearance = 10

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + self._height(pos) * scale,
                )
            )

        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            wheel = (
                int(c[0] + (pos - self.min_position) * scale),
                int(c[1] + clearance + self._height(pos) * scale),
            )

            gfxdraw.aacircle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )
            gfxdraw.filled_circle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )

        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

        gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def get_keys_to_action(self):
        # Control with left and right arrow keys.
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
