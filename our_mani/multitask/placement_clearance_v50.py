"""Stop above the complete conservative bin mesh before releasing a held body.

The TRAIN XY/orientation intention is retained. Its height is a lower bound;
current measured attachment geometry determines any additional clearance.
The resulting explicit goal keeps the original endpoint tolerances, stopped
release rule, complete collisions, joint bounds and episode deadline.
"""
from pathlib import Path
from copy import deepcopy
from types import MethodType
import json
import numpy as np
from ..contracts import Pose
from .evaluation import file_sha
from .protocol import MultitaskProtocolError

SCHEMA='r10_public_held_geometry_conservative_bin_release_height_v50'
TASKS=('put_rubbish_in_moving_bin','put_rubbish_in_moving_bin_high_speed')


def binding(root,protocol,task):
    if task not in TASKS:raise MultitaskProtocolError('clearance release supports exactly the two bin tasks')
    return dict(schema=SCHEMA,task=task,source=dict(path=str(Path(__file__).resolve().relative_to(Path(root).resolve())),sha256=file_sha(__file__)),
        bin_swept_geometry=protocol['axis_swept_assets'][task]['bin'],held_TRAIN_geometry=protocol['learned_artifacts'][task]['geometry'],
        goal_rule='retain TRAIN XY and TCP orientation; raise TCP just enough that every held TRAIN mesh vertex is above every conservative bin vertex, plus16float32ULPs',
        original_TRAIN_height_is_lower_bound=True,endpoint_tolerances_changed=False,stopped_release_required=True,
        original_collision_joint_grasp_gates_changed=False,deadline_extended=False,native_dynamic_pose_used=False,
        source_canonical_TCP_goal_replaced_by_explicit_clearance_goal=True,old_canonical_errors_retained=True,formal_admission=False)


def vertical_shift(bin_pose,bin_vertices,tcp_pose,tool_to_body,held_vertices):
    top=float(bin_pose.apply(bin_vertices)[:,2].max())
    relative=tool_to_body.apply(held_vertices)@tcp_pose.rotation.T
    bottom=float(relative[:,2].min())
    margin=16*float(np.spacing(np.float32(max(abs(top),1.))))
    required=top+margin-bottom
    shift=max(0.,required-float(tcp_pose.position[2]))
    return shift,dict(conservative_bin_top_z=top,held_bottom_relative_to_TCP_z=bottom,numerical_margin_m=margin,
        required_TCP_height_m=required,original_TRAIN_TCP_height_m=float(tcp_pose.position[2]),vertical_shift_m=shift)


class PlacementClearance:
    def __init__(self,actor,protocol,root,task):
        self.actor=actor;self.binding=binding(root,protocol,task);self.events=[]
        if protocol.get('placement_clearance_v50',{}).get(task)!=self.binding:raise MultitaskProtocolError('freeze explicit clearance-goal policy')
        def load(pointer):
            path=Path(root)/pointer['path']
            if file_sha(path)!=pointer['sha256']:raise MultitaskProtocolError('clearance TRAIN geometry changed')
            return json.loads(path.read_text())
        self.bin_vertices=np.asarray(load(self.binding['bin_swept_geometry'])['vertices'],float)
        self.held_vertices=np.asarray(load(self.binding['held_TRAIN_geometry'])['bodies']['rubbish']['collision_vertices_body'],float)

    def adjust(self,state,forecast,template,reference,target,arrival):
        if template.phase!='deposit' or not state.robot.contact:return reference,None
        attachment=self.actor.attachments.get('rubbish')
        if attachment is None or 'rubbish' not in self.actor.map.estimated_attachments:return reference,None
        modes,_=self.actor.collision_port._geometry_modes()
        if modes.get('rubbish')!='primary' or modes.get('bin')!='primary':return reference,None
        bin_pose=forecast.at('bin',arrival)[0]
        shift,row=vertical_shift(bin_pose,self.bin_vertices,target,attachment.tool_to_body,self.held_vertices)
        receipt=dict(row,schema=SCHEMA,selected_at=float(state.robot.timestamp),arrival=float(arrival),
            measured_attachment_at=attachment.observed_at,tool_to_body_xyzw=attachment.tool_to_body.xyzw().tolist(),
            original_reference_xyzw=reference.xyzw().tolist(),target_xyzw_before_clearance=target.xyzw().tolist(),
            goal_is_original_TRAIN_XY_orientation_plus_public_clearance=True,full_original_gates_still_required=True)
        return Pose(reference.position+np.array([0.,0.,shift]),reference.rotation),receipt

    def install_completion(self,controller):
        previous=controller._phase_done
        def phase_done(instance,state,target,**kwargs):
            goal=instance.goal;evidence=getattr(goal,'placement_clearance',None) if goal is not None else None
            if evidence is not None and goal.phase=='deposit' and kwargs.get('public_geometry') is True:
                old_error=float(np.linalg.norm(state.robot.pose.position-target.position))
                target=Pose(target.position+np.array([0.,0.,evidence['vertical_shift_m']]),target.rotation)
                self.events.append(dict(time=float(state.robot.timestamp),original_canonical_TCP_error_m=old_error,
                    explicit_clearance_goal_error_m=float(np.linalg.norm(state.robot.pose.position-target.position)),
                    original_position_tolerance_m=goal.template.position_tolerance_m,goal=evidence,
                    source_canonical_completion_not_claimed=True,original_phase_verifier_called=True))
            return previous(state,target,**kwargs)
        controller._phase_done=MethodType(phase_done,controller)
        previous_metrics=self.actor.metrics_report
        def metrics(instance):return dict(previous_metrics(),placement_clearance_v50=dict(binding=self.binding,events=deepcopy(self.events)))
        self.actor.metrics_report=MethodType(metrics,self.actor)
