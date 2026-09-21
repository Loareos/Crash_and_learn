import gymnasium as gym
import numpy as np
from gymnasium import spaces

from framework.simulation.env_simulation import (
    apply_action,
    close,
    get_obs,
    get_space_info,
    get_step_info,
    reset,
    simulation_step,
)


class CrashLearnEnv(gym.Env):
    """Gymnasium environment wrapping the Crash & Learn simulator."""

    metadata = {"render_modes": []}

    def __init__(self, num_cars: int = 1):
        super().__init__()

        self.num_cars = num_cars

        # We control only car 0 for now.
        self.car_id = 0

        space_info = get_space_info()

        # Action:
        #   [target_speed, steering]
        self.action_space = spaces.Box(
            low=np.array([
                space_info["action"]["speed"]["min"],
                space_info["action"]["steering"]["min"],
            ], dtype=np.float32),
            high=np.array([
                space_info["action"]["speed"]["max"],
                space_info["action"]["steering"]["max"],
            ], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation:
        # lidar + speed + progress
        lidar_rays = space_info["lidar"]["shape"][0]

        self.observation_space = spaces.Box(
            low=np.full(lidar_rays + 2, -np.inf, dtype=np.float32),
            high=np.full(lidar_rays + 2, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    def _get_observation(self):
        obs = get_obs(self.car_id)

        lidar = np.asarray(obs["lidar"], dtype=np.float32)

        speed = np.array(
            [float(obs.get("speed", 0.0))],
            dtype=np.float32,
        )

        progress = np.array(
            [float(obs.get("progress", 0.0))],
            dtype=np.float32,
        )

        return np.concatenate([
            lidar,
            speed,
            progress,
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        reset(self.num_cars)

        observation = self._get_observation()

        return observation, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        target_speed = float(action[0])
        steering = float(action[1])

        apply_action(
            self.car_id,
            target_speed,
            steering,
        )

        simulation_step()

        observation = self._get_observation()
        info = get_step_info()

        reward = self._calculate_reward(info)

        terminated = self._is_terminated(info)
        truncated = False

        return observation, reward, terminated, truncated, {}

    def _calculate_reward(self, info):
        progress_delta = float(
            info["progress_delta"][self.car_id]
        )

        reward = progress_delta * 100.0

        if info["collisions"]["wall"][self.car_id]:
            reward -= 10.0

        if info["collisions"]["vehicle"][self.car_id]:
            reward -= 10.0

        return reward

    def _is_terminated(self, info):
        return info["agent_status"][self.car_id] == 0

    def close(self):
        close()