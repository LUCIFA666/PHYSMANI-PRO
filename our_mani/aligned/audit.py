"""Public initial conditions for paired protocol acceptance, no future data."""
import hashlib
import numpy as np

def public_initial_conditions(env,observation,robot):
    digest=hashlib.sha256()
    cameras={}
    for name in sorted(observation['observation/images']):
        image=np.asarray(observation['observation/images'][name])
        depth=np.asarray(observation['observation/depths'][name])
        digest.update(name.encode());digest.update(image.tobytes());digest.update(depth.tobytes())
        calibration=observation['observation/calibrations'][name]
        cameras[name]=dict(intrinsics=calibration.intrinsics.tolist(),
                           camera_to_world=calibration.camera_to_world.xyzw().tolist())
    return dict(rgbd_sha256=digest.hexdigest(),robot_joints=robot.joints.tolist(),
                robot_pose=robot.pose.xyzw().tolist(),cameras=cameras,
                static_map=env.map_service.map,information_setting=env.protocol['information_setting'])
