"""A control curve reaches the unchanged canonical endpoint with zero local twist.

The fitted geometric curve is smoothly reparameterized; the small terminal
bias is blended out. External-frame velocity remains the forecast's velocity.
This changes the controller's reference policy, not the native success gate.
"""
from dataclasses import dataclass,fields
from types import MethodType
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from ..contracts import Pose
from .interaction import PhaseTemplate,COMMAND_GOAL


@dataclass(frozen=True)
class CanonicalTerminalTemplate(PhaseTemplate):
    control_source: object=None
    physical_endpoint: object=None

    def at(self,progress):
        s=float(np.clip(progress,0.,1.));weight=s*s*(3.-2.*s)
        derivative=6.*s*(1.-s);rate=derivative/self.duration_seconds
        pose,v,w,opening=self.control_source.at(weight)
        last=self.control_source.at(1.)[0];endpoint=self.physical_endpoint
        shift=endpoint.position-last.position
        delta=Rotation.from_matrix(last.rotation.T@endpoint.rotation).as_rotvec()
        rotation=pose.rotation@Rotation.from_rotvec(delta*weight).as_matrix()
        position=pose.position+weight*shift
        velocity=v*derivative+rate*shift
        angular=w*derivative+rate*(pose.rotation@delta)
        return Pose(position,rotation),velocity,angular,self.control_source.at(s)[3]


def terminal_curve(template):
    if template.reference_time_basis!=COMMAND_GOAL:raise ValueError('explicit canonical command curve required')
    endpoint=Pose.from_xyzw(template.completion_geometry_xyzw)
    args={f.name:getattr(template,f.name) for f in fields(PhaseTemplate)}
    return CanonicalTerminalTemplate(**args,control_source=template,physical_endpoint=endpoint)


def install(actor,*,phases,ready=lambda:True):
    c,m=actor.controller,actor.continuous_reference_manager
    if c.goal is not None or c.feedback_step_calls or m.confirmations:raise ValueError('install before execution')
    previous_step=c.step;events=[];active={'converted':False}
    def step(instance,state,forecast,remaining):
        phase=instance.library.phase_order[instance.phase_index] if instance.phase_index<len(instance.library.phase_order) else None
        if phase in phases and instance.goal is None and not active['converted'] and ready():
            converted=[]
            for t in instance.library.templates:
                if t.phase in phases:
                    source_pose,v,w,_=t.at(1.)
                    new=terminal_curve(t);converted.append(new)
                    events.append(dict(time=float(state.robot.timestamp),phase=t.phase,
                        event='canonical_endpoint_zero_local_twist_control_curve',
                        source_endpoint_xyzw=source_pose.xyzw().tolist(),
                        unchanged_canonical_endpoint_xyzw=t.completion_geometry_xyzw.tolist(),
                        old_local_endpoint_velocity=v.tolist(),old_local_endpoint_angular_velocity=w.tolist(),
                        actual_local_endpoint_velocity=[0.,0.,0.],actual_local_endpoint_angular_velocity=[0.,0.,0.],
                        original_duration_and_completion_tolerances_unchanged=True))
                else:converted.append(t)
            instance.library.templates=tuple(converted);active['converted']=True
        return previous_step(state,forecast,remaining)
    c.step=MethodType(step,c)
    previous_metrics=actor.metrics_report
    def metrics(instance):return dict(previous_metrics(),canonical_terminal_curve=dict(
        schema='r10_control_canonical_terminal_zero_relative_twist_v1',phases=list(phases),events=deepcopy(events),
        world_motion_of_external_target_retained=True,original_native_success_not_overridden=True))
    actor.metrics_report=MethodType(metrics,actor)
    return events
