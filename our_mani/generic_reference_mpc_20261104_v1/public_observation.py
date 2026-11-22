"""Strict public policy packet, independent of private robot controller state."""
import numpy as np

CAMERAS = ('front', '8', '16', '36')
KEYS = {'timestamp', 'rgb', 'depth_m', 'K', 'T', 'camera_clips',
        'camera_names', 'robot14', 'robot15', 'robot15_observed_mask', 'input_semantics'}


def pack_public_observation(*, timestamp, rgb, depth_m, K, T, camera_clips, q, tcp_pose):
    if not np.isfinite(timestamp) or timestamp < 0:
        raise ValueError('Actual current time required')
    arrays = dict(rgb=np.asarray(rgb), depth_m=np.asarray(depth_m), K=np.asarray(K), T=np.asarray(T),camera_clips=np.asarray(camera_clips),
        q=np.asarray(q, float), tcp_pose=np.asarray(tcp_pose, float))
    shapes = dict(rgb=(4,256,256,3), depth_m=(4,256,256), K=(4,3,3), T=(4,4,4), camera_clips=(4,2),q=(7,), tcp_pose=(7,))
    if any(arrays[k].shape != shape or not np.isfinite(arrays[k]).all() for k, shape in shapes.items()):
        raise ValueError('Aligned current RGBD/K/T/q/TCP shapes and finite values required')
    if arrays['rgb'].dtype != np.uint8 or arrays['depth_m'].dtype != np.float32 or np.any(arrays['depth_m'] < 0):
        raise ValueError('Original uint8 RGB and metric float32 depth required')
    if np.any(arrays['camera_clips'][:,0]<=0) or np.any(arrays['camera_clips'][:,1]<=arrays['camera_clips'][:,0]):
        raise ValueError('Actual finite camera near/far planes required, no fabricated clipping range')
    if not np.isclose(np.linalg.norm(arrays['tcp_pose'][3:]), 1., atol=1e-5):
        raise ValueError('Current TCP pose quaternion must be unit')
    if (not np.allclose(arrays['K'][:,2], [0,0,1], atol=1e-6, rtol=0)
            or np.any(np.abs(np.linalg.det(arrays['K'])) < 1e-8)
            or not np.allclose(arrays['T'][:,3], [0,0,0,1], atol=1e-6, rtol=0)
            or not np.allclose(arrays['T'][:,:3,:3].transpose(0,2,1) @ arrays['T'][:,:3,:3], np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(arrays['T'][:,:3,:3]), 1., atol=1e-5, rtol=0)):
        raise ValueError('Actual signed calibration matrices required')
    robot14 = np.r_[arrays['q'], arrays['tcp_pose']].astype(np.float32)
    robot15 = np.r_[robot14, np.float32(0)]
    mask = np.ones(15, dtype=bool); mask[-1] = False
    packet = dict(timestamp=float(timestamp), rgb=arrays['rgb'].copy(),
        depth_m=arrays['depth_m'].copy(), K=arrays['K'].copy(), T=arrays['T'].copy(),
        camera_clips=arrays['camera_clips'].copy(),
        camera_names=CAMERAS, robot14=robot14, robot15=robot15, robot15_observed_mask=mask,
        input_semantics='current_RGBD_public_K_T_q_TCP_only; gripper_scalar_masked_not_observed')
    for value in packet.values():
        if isinstance(value, np.ndarray): value.setflags(write=False)
    return packet


def validate_policy_packet(packet):
    if type(packet) is not dict or set(packet) != KEYS:
        raise ValueError('Unknown/private field in public observation; no silent key filtering')
    if packet['input_semantics'] != 'current_RGBD_public_K_T_q_TCP_only; gripper_scalar_masked_not_observed':
        raise ValueError('No Native held/contact/pose/instance masks or HDF gripper scalar in policy')
    if (tuple(packet['camera_names']) != CAMERAS or packet['robot15'][-1] != 0
            or packet['robot15_observed_mask'][-1] != np.False_):
        raise ValueError('Unobserved contaminated gripper channel must remain maskFalse/zero')
    rebuilt = pack_public_observation(timestamp=packet['timestamp'], rgb=packet['rgb'],
        depth_m=packet['depth_m'], K=packet['K'], T=packet['T'],camera_clips=packet['camera_clips'],
        q=packet['robot14'][:7], tcp_pose=packet['robot14'][7:])
    for key in ('robot14','robot15','robot15_observed_mask'):
        if not np.array_equal(packet[key], rebuilt[key]): raise ValueError('Public robot/operation fields disagree')
    return packet
