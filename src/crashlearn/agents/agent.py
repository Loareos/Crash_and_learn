from stable_baselines3 import PPO

class Agent:
    def __init__(self):
        self.model = PPO.load("models/crashlearn_ppo")

    def predict(self,obs,info):
        action, _ = self.model.predict(obs, deterministic=True)
        speed = float(action[0])
        steering = float(action[1])
        return speed, steering