import numpy as np

def format_observation(self, obs):
    return {
        "lidar": np.asarray(obs["lidar"], dtype=np.float32),
        "velocity": np.asarray([obs["velocity"]], dtype=np.float32),
        "steering": np.asarray([obs["steering"]], dtype=np.float32),
        "progress": np.asarray([obs["progress"]], dtype=np.float32),
        "lap_count": np.asarray([obs["lap_count"]], dtype=np.float32)
    }
    
def calculate_reward(self, obs, infos, speed):
    collision = (
        infos["collisions"]["wall"][0]
        or infos["collisions"]["vehicle"][0]
    )

    lap_complete = infos["lap_complete"][0]
    progress = float(infos["progress_delta"][0])

    reward = progress * 100.0

    if collision:
        reward -= 50.0
    else:
        reward += 0.1

    if lap_complete:
        reward += 200.0

    return reward
    # def _calculate_reward(self,obs,infos,speed):
    #     collision = (
    #         infos["collisions"]["wall"][0] 
    #         or infos["collisions"]["vehicle"][0]
    #     )

    #     lap_complete = infos["lap_complete"][0]
    #     progress = float(infos["progress_delta"][0])

    #     reward = progress * 100

    #     front_distance = float(np.min(obs["lidar"][40:60]))

    #     if front_distance < 1.0:
    #         target_speed = 2.0
    #     elif front_distance < 2.0:
    #         target_speed = 4.0
    #     elif front_distance < 4.0:
    #         target_speed = 6.0
    #     else:
    #         target_speed = 8.0

    #     speed_error = abs(float(speed) - target_speed)
    #     reward += max(0.0, 1.0 - speed_error / 8.0)

    #     if collision:
    #         reward -= 50
    #     else:
    #         reward += 0.1

    #     if lap_complete:
    #         reward += 200
    #     return reward