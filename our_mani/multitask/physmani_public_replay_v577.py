"""Retain already-supplied public RGBD for offline registration diagnosis.

No extra camera/physics call and no hidden/native body pose is captured. Only
three existing public frames and the unchanged declared TRAIN templates are
saved. Native outcomes remain outside the frontend.
"""
from dataclasses import asdict
from pathlib import Path
from types import MethodType
import pickle
import numpy as np
from PIL import Image
from .physmani_public_trace_v566 import TracedPublicFrontend
from .pipeline import _plain
from .evaluation import atomic_json


class ReplayPublicFrontend(TracedPublicFrontend):
    def __init__(self,actor,*,output):
        super().__init__(actor);self.output=Path(output);self.output.mkdir()
        self.full_metadata=[];self.registrations=[];self.public_frame_count=0
        payload=dict(templates=tuple(actor.tracker.templates.values()),constraints=actor.tracker.constraints,
                     config=actor.tracker.config)
        (self.output/'public_TRAIN_templates.pkl').write_bytes(pickle.dumps(payload))
        previous=actor.tracker._registration
        def registration(tracker,members,clouds,initial):
            event=dict(time=float(tracker._current_timestamp),members=list(members),
                public_point_counts={k:{str(component):len(points) for component,points in clouds[k].items()} for k in members},
                previous_public_pose_supplied=initial is not None)
            result=previous(members,clouds,initial)
            event['registration_returned']=result is not None
            if result is not None:
                event.update(valid=result['valid'],rms_m=result['rms_m'],per_role=result['per_role'],
                    rotation_ambiguous=result['rotation_ambiguous'])
            self.registrations.append(_plain(event));return result
        actor.tracker._registration=MethodType(registration,actor.tracker)

    def observe(self,observation,robot):
        index=self.public_frame_count;self.public_frame_count+=1
        if index in (0,4,20):
            arrays={};cameras=list(observation['observation/images']);calibrations={}
            for name in cameras:
                arrays['rgb_'+name]=observation['observation/images'][name]
                arrays['depth_'+name]=observation['observation/depths'][name]
                if name in observation.get('observation/instance_masks',{}):
                    arrays['mask_'+name]=observation['observation/instance_masks'][name]
                c=observation['observation/calibrations'][name]
                calibrations[name]=dict(intrinsics=c.intrinsics.tolist(),camera_to_world_xyzw=c.camera_to_world.xyzw().tolist(),depth_scale=float(c.depth_scale))
            np.savez_compressed(self.output/f'public_frame_{index:03d}.npz',**arrays)
            atomic_json(self.output/f'public_frame_{index:03d}.json',dict(timestamp=float(robot.timestamp),
                cameras=cameras,calibrations=calibrations,robot_pose_xyzw=robot.pose.xyzw().tolist(),
                input_domain='existing_current_public_RGBD_masks_calibration_proprioception',
                extra_camera_queries=0,extra_physics_ticks=0,native_body_poses_or_outcomes=False))
            if len(cameras)==4:
                rgb=[np.asarray(observation['observation/images'][k],np.uint8) for k in cameras]
                Image.fromarray(np.concatenate([np.concatenate(rgb[:2],axis=1),np.concatenate(rgb[2:],axis=1)],axis=0)).save(self.output/f'public_frame_{index:03d}.png')
        result=super().observe(observation,robot)
        self.full_metadata.append(_plain(dict(time=float(robot.timestamp),
            bodies=self.state.perception_metadata.get('bodies',{}),rigid_groups=self.state.perception_metadata.get('rigid_groups',[]))))
        return result

    def receipt(self):
        return dict(super().receipt(),all_body_diagnostics=self.full_metadata,
            registration_attempts=self.registrations,public_replay_path=str(self.output),
            saved_existing_frame_indices=[i for i in (0,4,20) if i<self.public_frame_count])
