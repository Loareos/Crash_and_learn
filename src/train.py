from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment import CrashLearnEnv
import gym

if not hasattr(gym, "__version__"):
    gym.__version__ = "0.26.2"

def main():
    env = Monitor(CrashLearnEnv())
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="logs",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05
        )
    model.learn(total_timesteps=500_000)
    model.save("models/crashlearn_ppo")
    env.close()

if __name__ == "__main__":
    main()