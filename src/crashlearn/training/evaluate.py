from stable_baselines3 import PPO
from src.crashlearn.environment.environment import CrashLearnEnv

def main():
    env = CrashLearnEnv()
    model = PPO.load("models/crashlearn_ppo")
    observation, info = env.reset()
    for step in range(1000):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        print(
            f"step={step}"
            f"action={action}"
            f"reward={reward}"
        )
        if terminated or truncated:
            print("FINI")
            break
    env.close()

if __name__ == "__main__":
    main()