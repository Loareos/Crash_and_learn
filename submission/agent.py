import numpy as np
from stable_baselines3 import PPO

class Agent:
    def __init__(self):
        self.model = PPO.load("models/crashlearn_ppo")

    def predict(self,obs,info):
        obs =  {
            "lidar": np.asarray(obs["lidar"], dtype=np.float32),
            "velocity": np.asarray([obs["velocity"]], dtype=np.float32),
            "steering": np.asarray([obs["steering"]], dtype=np.float32),
            "progress": np.asarray([obs["progress"]], dtype=np.float32),
            "lap_count": np.asarray([obs["lap_count"]], dtype=np.float32)
        }

        action, _ = self.model.predict(obs, deterministic=True)
        speed = float(action[0])
        steering = float(action[1])
        return speed, steering