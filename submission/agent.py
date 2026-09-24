import os
import numpy as np
import onnxruntime as ort

class Agent:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "model.onnx")
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

    def predict(self, obs, info):
        inputs = {
            "lidar": np.asarray(obs["lidar"], dtype=np.float32).reshape(1, 100),
            "velocity": np.asarray([obs["velocity"]], dtype=np.float32).reshape(1, 1),
            "steering": np.asarray([obs["steering"]], dtype=np.float32).reshape(1, 1),
            "progress": np.asarray([obs["progress"]], dtype=np.float32).reshape(1, 1),
            "lap_count": np.asarray([obs["lap_count"]], dtype=np.float32).reshape(1, 1),
        }

        action = self.session.run(["action"], inputs)[0][0]

        return float(action[0]), float(action[1])