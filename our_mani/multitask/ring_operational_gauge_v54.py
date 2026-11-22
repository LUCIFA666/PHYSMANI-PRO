"""Choose a continuous certified D4 action representative from public TCP pose.

The observed orientation remains a quotient, never full SO3. Selection is
restricted to the already proved eight TRAIN ring geometry/action orbits.
The first representative minimizes grasp-orientation travel; later frames
minimize changes from the previous public representative.
"""
from pathlib import Path
from copy import deepcopy
from types import MethodType
import numpy as np
from scipy.spatial.transform import Rotation
from ..contracts import Pose
from .evaluation import file_sha
from .protocol import MultitaskProtocolError

SCHEMA='r10_current_public_continuous_D4_operational_gauge_v54'


def choose(pose,group,center,*,previous=None,tcp=None,local_grasp_rotation=None):
    group=np.asarray(group,float);center=np.asarray(center,float)
    if group.shape!=(8,3,3) or center.shape!=(3,):raise ValueError('exact proved D4 group and center required')
    choices=[]
    for index,g in enumerate(group):
        value=Pose(pose.position+pose.rotation@(center-g@center),pose.rotation@g)
        if previous is not None:angle=Rotation.from_matrix(value.rotation@previous.rotation.T).magnitude()
        elif tcp is not None and local_grasp_rotation is not None:angle=Rotation.from_matrix(value.rotation@local_grasp_rotation@tcp.rotation.T).magnitude()
        else:raise ValueError('current public TCP or previous public representative required')
        choices.append((float(angle),index,value))
    angle,index,value=min(choices,key=lambda row:(row[0],row[1]))
    return value,dict(group_index=index,orientation_travel_rad=angle,previous_pose_used=previous is not None,full_orientation_promoted=False)


def binding(root,protocol,task):
    if task!='insert_onto_rotating_peg' or task not in protocol.get('ring_runtime_v25',{}):raise MultitaskProtocolError('full TRAIN ring action proof required before gauge selection')
    return dict(schema=SCHEMA,task=task,source=dict(path=str(Path(__file__).resolve().relative_to(Path(root).resolve())),sha256=file_sha(__file__)),
        full_TRAIN_permission=protocol['ring_affordances'][task],exact_replay=protocol['ring_runtime_v25'][task]['replay_report'],
        allowed_group='existing proved D4 only; equivalent pose around certified CAD center',
        first_frame_rule='minimise orientation between current public TCP and TRAIN approach endpoint across eight proved equivalents',
        later_frame_rule='nearest previous public representative; no frame-dependent task outcome or native labels',
        current_retained_RGBD_hypotheses_revalidated=True,full_orientation_promoted=False,grasp_collision_joint_gates_changed=False)


def install(actor,protocol,root,task):
    expected=binding(root,protocol,task)
    if protocol.get('ring_operational_gauge_v54',{}).get(task)!=expected:raise MultitaskProtocolError('freeze actual ring gauge adapter')
    template=actor.controller.library.for_phase('approach')[0];local_rotation=template.at(1.)[0].rotation
    symmetry=actor.tracker.templates['ring'].finite_rotation_symmetry
    group=symmetry.certificate['group_rotations_body'];center=symmetry.certificate['center_body'];state={};events=[]
    previous_act=actor.act
    def act(instance,observation,robot,remaining_seconds):
        state['tcp']=robot.pose
        return previous_act(observation,robot,remaining_seconds)
    actor.act=MethodType(act,actor)
    previous_symmetry=actor.tracker._symmetry_pose
    def symmetry_pose(instance,role,pose,points,previous):
        original=previous_symmetry(role,pose,points,previous)
        if role!='ring':return original
        selected,row=choose(original,group,center,previous=previous,tcp=state.get('tcp'),local_grasp_rotation=local_rotation)
        events.append(dict(row,time=float(instance._current_timestamp),selected_xyzw=selected.xyzw().tolist()))
        return selected
    actor.tracker._symmetry_pose=MethodType(symmetry_pose,actor.tracker)
    previous_metrics=actor.metrics_report
    def metrics(instance):return dict(previous_metrics(),ring_operational_gauge_v54=dict(binding=expected,events=deepcopy(events)))
    actor.metrics_report=MethodType(metrics,actor)
