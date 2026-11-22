"""Read-only PhysMani HDF5 observations, separated from privileged labels."""
from pathlib import Path

import h5py
import numpy as np

from .contracts import Pose
from .perception import CameraCalibration


CAMERAS = ("front", "8", "16", "36")
DT = 0.05


def decode_depth(rgb_uint8, near, far):
    value = np.asarray(rgb_uint8, dtype=np.uint32)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("PhysMani depth must be 24-bit RGB encoded")
    normalized = ((value[..., 0] << 16) | (value[..., 1] << 8) | value[..., 2]).astype(np.float64)/(2**24-1)
    return float(near)+normalized*(float(far)-float(near))


class RecordedEpisode:
    def __init__(self, path):
        self.path = Path(path)
        self.file = None

    def __enter__(self):
        self.file = h5py.File(self.path, "r")
        return self

    def __exit__(self, *args):
        self.file.close()
        self.file = None

    def __len__(self):
        return len(self.file["low_dim_obs/gripper_pose"])

    def observation(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        images, depths, calibrations = {}, {}, {}
        for camera in CAMERAS:
            params = self.file[f"cam_params/{camera}"]
            transform = params["extrinsics"][...]
            if not np.allclose(transform[3], [0., 0., 0., 1.]):
                raise ValueError("unsupported camera extrinsic")
            calibration = CameraCalibration(params["intrinsics"][...], Pose(transform[:3, 3], transform[:3, :3]))
            images[camera] = self.file[f"observations/{camera}/rgb"][index]
            depths[camera] = decode_depth(self.file[f"observations/{camera}/depth"][index], params["near"][()], params["far"][()])
            calibrations[camera] = calibration
        pose = self.file["low_dim_obs/gripper_pose"][index]
        state = np.r_[pose, self.file["low_dim_obs/gripper_open"][index]].astype(np.float32)
        # Deliberately excludes expert targets, future frames, WM predictions.
        return {"timestamp": index*DT, "observation/images": images, "observation/depths": depths,
                "observation/calibrations": calibrations, "observation/state": state,
                "observation/joints": self.file["low_dim_obs/joint_positions"][index]}

    def training_labels(self, index):
        return {"expert_action": self.file["low_dim_obs/expert_trajectory"][index],
                "expert_stage": self.file["low_dim_obs/expert_trajectory_stage"][index],
                "privileged_target_position": self.file["low_dim_obs/expert_trajectory_target_position"][index]}
