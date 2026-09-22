from stable_baselines3 import PPO
from src.crashlearn.environment.environment import CrashLearnEnv
import gym

if not hasattr(gym, "__version__"):
    gym.__version__ = "0.26.2"

def main():
    env = CrashLearnEnv()
    model = PPO("MultiInputPolicy",env,verbose=1)
    model.learn(total_timesteps=1_000)
    model.save("models/crashlearn_ppo")
    env.close()

if __name__ == "__main__":
    main()