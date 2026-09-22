import gymnasium as gym
import numpy as np
from framework.simulation.env_simulation import (
    reset,
    get_obs,
    apply_action,
    simulation_step,
    get_step_info,
    close
)

class CrashLearnEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(
            low=np.array([-2.0, -0.4189], dtype=np.float32),
            high=np.array([10.0, 0.4189], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({
            # lidar contient 100 nombres, chacun compris entre 0.1 et 15
            "lidar": gym.spaces.Box(low=0.1, high=15.0, shape=(100,), dtype=np.float32),
            "velocity": gym.spaces.Box(low=-2.0,high=10.0,shape=(1,),dtype=np.float32),
            "steering": gym.spaces.Box(low=-0.4189,high=0.4189,shape=(1,),dtype=np.float32),
            "progress": gym.spaces.Box(low=0.0,high=1.0,shape=(1,),dtype=np.float32),
            "lap_count": gym.spaces.Box(low=0,high=np.inf,shape=(1,),dtype=np.float32),
            # "rank": gym.spaces.Box(low=,high=,shape=,dtype=np.float32),
            # "opponents": gym.spaces.,
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        reset()
        observation = self._format_observation(get_obs(0))
        return observation, {}

    def step(self,action):
        speed, steering = action
        apply_action(0, speed, steering)
        simulation_step()
        
        observation = self._format_observation(get_obs(0))
        step_info = get_step_info()

        reward = float(step_info["progress_delta"][0])
        collision = (
            step_info["collisions"]["wall"][0] 
            or step_info["collisions"]["vehicle"][0])
        lap_complete = step_info["lap_complete"][0]

        if collision:
            reward -= 10
        if lap_complete:
            reward += 100

        terminated = False
        truncated = False
        return observation, reward, terminated, truncated, {}

    def close(self):
        close()

    def _format_observation(self, obs):
        return {
            "lidar": np.asarray(obs["lidar"], dtype=np.float32),
            "velocity": np.asarray([obs["velocity"]], dtype=np.float32),
            "steering": np.asarray([obs["steering"]], dtype=np.float32),
            "progress": np.asarray([obs["progress"]], dtype=np.float32),
            "lap_count": np.asarray([obs["lap_count"]], dtype=np.float32)
        }