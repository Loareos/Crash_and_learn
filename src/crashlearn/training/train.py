from stable_baselines3 import PPO
from src.crashlearn.environment.environment import CrashLearnEnv
import gym

if not hasattr(gym, "__version__"):
    gym.__version__ = "0.26.2"

def main():
    env = CrashLearnEnv()
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
        )
    model.learn(total_timesteps=100_000)
    model.save("models/crashlearn_ppo")
    env.close()

if __name__ == "__main__":
    main()