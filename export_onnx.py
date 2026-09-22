import torch
import torch.nn as nn
from stable_baselines3 import PPO


class PPOWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.features_extractor = model.policy.features_extractor
        self.mlp_extractor = model.policy.mlp_extractor
        self.action_net = model.policy.action_net

    def forward(self, lidar, velocity, steering, progress, lap_count):
        observations = {
            "lidar": lidar,
            "velocity": velocity,
            "steering": steering,
            "progress": progress,
            "lap_count": lap_count,
        }

        features = self.features_extractor(observations)

        latent_pi, _ = self.mlp_extractor(features)

        action = self.action_net(latent_pi)

        return action


model = PPO.load("models/crashlearn_ppo")
model.policy.eval()

wrapper = PPOWrapper(model)
wrapper.eval()

dummy_lidar = torch.randn(1, 100)
dummy_velocity = torch.randn(1, 1)
dummy_steering = torch.randn(1, 1)
dummy_progress = torch.randn(1, 1)
dummy_lap_count = torch.randn(1, 1)

torch.onnx.export(
    wrapper,
    (
        dummy_lidar,
        dummy_velocity,
        dummy_steering,
        dummy_progress,
        dummy_lap_count,
    ),
    "submission/model.onnx",
    input_names=[
        "lidar",
        "velocity",
        "steering",
        "progress",
        "lap_count",
    ],
    output_names=["action"],
    dynamic_axes={
        "lidar": {0: "batch"},
        "velocity": {0: "batch"},
        "steering": {0: "batch"},
        "progress": {0: "batch"},
        "lap_count": {0: "batch"},
        "action": {0: "batch"},
    },
    opset_version=17,
)

print("ONNX model exported to submission/model.onnx")