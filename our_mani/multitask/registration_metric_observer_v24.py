"""TRAIN-only registration diagnostics; native labels never reach policy input."""
from copy import deepcopy
from types import MethodType
import numpy as np
from scipy.spatial.transform import Rotation
from ..contracts import Pose
from .native_tasks import resolve_native_roles


class RegistrationMetricObserver:
    def __init__(self,actor,env,*,metric_objects=None):
        self.events=[];self.env=env
        objects=(resolve_native_roles(env.task,env.spec,scene_object_resolver=env.resolve_scene_object)
            if metric_objects is None else metric_objects)
        tracker=actor.tracker;original=tracker._registration
        def registration(instance,members,clouds,previous=None):
            result=original(members,clouds,previous)
            clock=(env.ledger.ticks,env.physics_time(),env.task.t)
            for role,local in members.items():
                if role not in objects:continue
                row=dict(time=float(instance._current_timestamp),role=role,
                    registration_available=result is not None,native_label_is_metric_only=True)
                if result is not None:
                    root=result['pose'];pose=Pose(root.position+root.rotation@local.position,root.rotation@local.rotation)
                    row.update(raw_registration_xyzw=pose.xyzw().tolist(),
                        rotation_ambiguous=bool(result['rotation_ambiguous']),valid=bool(result['valid']),
                        rms_m=float(result['rms_m']),role_quality=deepcopy(result['per_role'].get(role)))
                    try:
                        truth=Pose.from_xyzw(objects[role].get_pose())
                        row.update(native_metric_xyzw=truth.xyzw().tolist(),
                            position_error_m=float(np.linalg.norm(pose.position-truth.position)),
                            orientation_error_rad=float(Rotation.from_matrix(pose.rotation@truth.rotation.T).magnitude()))
                    except Exception as error:row['native_metric_unavailable']=repr(error)
                self.events.append(row)
            if clock!=(env.ledger.ticks,env.physics_time(),env.task.t):
                raise RuntimeError('Metric registration observer advanced physics')
            return result
        tracker._registration=MethodType(registration,tracker)

    def receipt(self):
        return dict(schema='r10_original_registration_current_TRAIN_metric_observer_v24',
            events=deepcopy(self.events),native_labels_supplied_to_actor=False,
            pose_or_ambiguity_result_modified=False,additional_camera_calls=0,additional_physics_ticks=0)
