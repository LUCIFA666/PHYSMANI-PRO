"""Actual public observation prefixes; no imagined frames/calibration or dt."""
import numpy as np
from ..generic_reference_mpc_20261104_v1.public_observation import validate_policy_packet,pack_public_observation


class PublicHistoryV1:
    def __init__(self):self._rows=[]

    def clear(self):self._rows=[]

    def append(self,packet,*,frame_index):
        validate_policy_packet(packet)
        if type(frame_index)is not int or frame_index<0:raise ValueError('Actual public source frame index required')
        if self._rows:
            old_index,old=self._rows[-1]
            if frame_index!=old_index+1 or packet['timestamp']<=old['timestamp']:
                raise ValueError('Unique increasing consecutive source observations required; no invented or duplicated history')
            if any(not np.array_equal(packet[k],old[k])for k in ('K','T','camera_clips')):
                raise ValueError('Frozen encoder assumes fixed history calibration; explicitly clear before a new camera setup')
        copied=pack_public_observation(timestamp=packet['timestamp'],rgb=packet['rgb'],depth_m=packet['depth_m'],
            K=packet['K'],T=packet['T'],camera_clips=packet['camera_clips'],
            q=packet['robot14'][:7],tcp_pose=packet['robot14'][7:])
        self._rows=(self._rows+[(frame_index,copied)])[-5:]

    def arrays(self):
        if not self._rows:raise ValueError('At least one actual public observation required')
        rgbd=np.zeros((1,5,4,4,256,256),np.float32);robot=np.zeros((1,5,15),np.float32)
        indices=np.full((1,5),-1,np.int64)
        for slot,(index,packet)in enumerate(self._rows):
            rgbd[0,slot,:,:3]=np.moveaxis(packet['rgb'].astype(np.float32)/127.5-1.,-1,1)
            rgbd[0,slot,:,3]=packet['depth_m'];robot[0,slot]=packet['robot15'];indices[0,slot]=index
        current=self._rows[-1][1]
        return dict(rgbd=rgbd,robot_history=robot,lengths=np.asarray([len(self._rows)],np.int64),
            K=np.asarray(current['K'],np.float32)[None].copy(),camera_to_world=np.asarray(current['T'],np.float32)[None].copy(),
            near_far=np.asarray(current['camera_clips'],np.float32)[None].copy(),frame_indices=indices)

    def trace(self):
        return dict(source_frame_indices=[i for i,_ in self._rows],actual_public_timestamps=[p['timestamp']for _,p in self._rows],
            observed_frame_dt=[self._rows[i][1]['timestamp']-self._rows[i-1][1]['timestamp']for i in range(1,len(self._rows))],
            model_uses_physical_dt_channel=False,source_physics_clock_independently_verified=False,
            previous_frame_duplicated=False,masked_gripper_input_zero=True)
