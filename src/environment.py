import gymnasium as gym
import numpy as np
from framework.env_simulation import (
    reset,
    get_obs,
    apply_action,
    simulation_step,
    get_step_info,
    close,
    set_map,
)
from src.features import (
    format_observation,
    calculate_reward
)

metadata = {"render_mode": []}

class CrashLearnEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
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
        self.car_id = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        reset()
        raw_obs = get_obs(self.car_id)
        observation = format_observation(raw_obs)
        return observation, {}

    def step(self,action):

        action = np.asarray(action, dtype=np.float32)
        speed = float(action[0])
        steering = float(action[1])

        apply_action(self.car_id, speed, steering)
        simulation_step()
        
        raw_obs = get_obs(self.car_id)
        observation = format_observation(raw_obs)
        step_info = get_step_info()

        reward = calculate_reward(observation,step_info,speed)

        terminated = (
            step_info["collisions"]["wall"][self.car_id] 
            or step_info["collisions"]["vehicle"][self.car_id]
        )

        truncated = False

        return observation, float(reward), terminated, truncated, {
            "reward": float(reward),
            "progress_delta": float(step_info["progress_delta"][self.car_id])
        }
    
    def render(self):
        return None

    def close(self):
        close()