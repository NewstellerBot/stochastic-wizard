__credits__ = ["Kallinteris-Andreas"]

from typing import Dict, Tuple, Union

import numpy as np

from gymnasium import utils

# from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from magic_env.envs.mujoco import MujocoEnv


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.8925)),
    "elevation": -20.0,
}

INERTIA_BOUND = 1e-15
MASS_BOUND = 1e-15


class MagicHumanoidStandupEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "humanoidstandup.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        uph_cost_weight: float = 1,
        ctrl_cost_weight: float = 0.1,
        impact_cost_weight: float = 0.5e-6,
        impact_cost_range: Tuple[float, float] = (-np.inf, 10.0),
        reset_noise_scale: float = 1e-2,
        exclude_current_positions_from_observation: bool = True,
        include_cinert_in_observation: bool = True,
        include_cvel_in_observation: bool = True,
        include_qfrc_actuator_in_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
        magic_mask: bool = False,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            uph_cost_weight,
            ctrl_cost_weight,
            impact_cost_weight,
            impact_cost_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            include_cinert_in_observation,
            include_cvel_in_observation,
            include_qfrc_actuator_in_observation,
            include_cfrc_ext_in_observation,
            **kwargs,
        )

        self.magic_mask = magic_mask
        self._uph_cost_weight = uph_cost_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._impact_cost_weight = impact_cost_weight
        self._impact_cost_range = impact_cost_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self._include_cinert_in_observation = include_cinert_in_observation
        self._include_cvel_in_observation = include_cvel_in_observation
        self._include_qfrc_actuator_in_observation = (
            include_qfrc_actuator_in_observation
        )
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

        obs_size = 47
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += 130 * include_cinert_in_observation
        obs_size += 78 * include_cvel_in_observation
        obs_size += 17 * include_qfrc_actuator_in_observation
        obs_size += 78 * include_cfrc_ext_in_observation

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += self.data.cinert[1:].size * include_cinert_in_observation
        obs_size += self.data.cvel[1:].size * include_cvel_in_observation
        obs_size += (self.data.qvel.size - 6) * include_qfrc_actuator_in_observation
        obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
            "cinert": self.data.cinert[1:].size * include_cinert_in_observation,
            "cvel": self.data.cvel[1:].size * include_cvel_in_observation,
            "qfrc_actuator": (self.data.qvel.size - 6)
            * include_qfrc_actuator_in_observation,
            "cfrc_ext": self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation,
            "ten_length": 0,
            "ten_velocity": 0,
        }

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._include_cinert_in_observation is True:
            com_inertia = self.data.cinert[1:].flatten()
        else:
            com_inertia = np.array([])
        if self._include_cvel_in_observation is True:
            com_velocity = self.data.cvel[1:].flatten()
        else:
            com_velocity = np.array([])

        if self._include_qfrc_actuator_in_observation is True:
            actuator_forces = self.data.qfrc_actuator[6:].flatten()
        else:
            actuator_forces = np.array([])
        if self._include_cfrc_ext_in_observation is True:
            external_contact_forces = self.data.cfrc_ext[1:].flatten()
        else:
            external_contact_forces = np.array([])

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )

    def step(self, action):
        # Get magic action from action
        *action, multiplier = action

        # When mask is true, multiply by identity
        if self.magic_mask:
            multiplier = 1

        # Modify gravity
        self.model.body_mass = np.maximum(multiplier * self.model.body_mass, MASS_BOUND)
        # Modify innertia
        idx, *_ = self.model.body_inertia.shape
        self.model.body_inertia[range(idx), :] = np.maximum(
            multiplier * self.model.body_inertia[range(idx), :], INERTIA_BOUND
        )

        self.do_simulation(action, self.frame_skip)
        pos_after = self.data.qpos[2]

        reward, reward_info = self._get_rew(pos_after, action)
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "z_distance_from_origin": self.data.qpos[2] - self.init_qpos[2],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(), reward, False, False, info

    def _get_rew(self, pos_after: float, action):
        uph_cost = (pos_after - 0) / self.model.opt.timestep

        quad_ctrl_cost = self._ctrl_cost_weight * np.square(self.data.ctrl).sum()

        quad_impact_cost = (
            self._impact_cost_weight * np.square(self.data.cfrc_ext).sum()
        )
        min_impact_cost, max_impact_cost = self._impact_cost_range
        quad_impact_cost = np.clip(quad_impact_cost, min_impact_cost, max_impact_cost)

        reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1

        reward_info = {
            "reward_linup": uph_cost,
            "reward_quadctrl": -quad_ctrl_cost,
            "reward_impact": -quad_impact_cost,
        }

        return reward, reward_info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "z_distance_from_origin": self.data.qpos[2] - self.init_qpos[2],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
        }
