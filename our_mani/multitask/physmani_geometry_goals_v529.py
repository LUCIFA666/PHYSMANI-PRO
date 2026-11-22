"""Public trained skill endpoints for the shared PhysMani pose-action interface.

Pure target construction. It neither advances a phase nor operates a robot.
Current references retain the existing observation/rotation admission, and
each future reference stays inside the current predictor's causal horizon.
"""
import numpy as np
from ..contracts import Pose
from .interaction import COMMAND_GOAL,_reference_at,_world_target


def pose_goals(frontend,phase,*,origin=None,lookahead=.5,committed=None):
    if not np.isfinite(lookahead) or not .05<=lookahead<=1.1:
        raise ValueError('bounded causal pose-action lookahead required')
    a=frontend.actor;c=a.controller;state,forecast=frontend.state,frontend.forecast
    if state is None or forecast is None:
        return []
    now=float(state.robot.timestamp);cutoff=forecast.metadata.get('observed_until')
    if type(cutoff) not in (int,float) or abs(float(cutoff)-now)>1e-8:
        return []
    bindings={**state.role_bindings,**c.committed_roles}
    rows=[]
    for template in c.library.for_phase(phase):
        if committed is not None and template.anchor_id!=committed['template'].anchor_id:
            continue
        references=((template.reference_kind+':'+phase,) if template.reference_kind in ('phase_origin','world')
                    else bindings.get(template.reference_role,()))
        manipulated=bindings.get(template.manipulated_role,())
        for reference_id in references:
            choices=(reference_id,) if template.reference_role==template.manipulated_role else manipulated
            for manipulated_id in choices:
                if committed is not None and (reference_id!=committed['reference_id']
                        or (manipulated_id,)!=committed['manipulated_ids']):
                    continue
                try:
                    current=c.generator.observed_reference(state,reference_id,template,origin)
                    if template.reference_kind not in ('phase_origin','world'):
                        mode=state.perception_metadata.get('bodies',{}).get(reference_id,{}).get('motion_mode')
                        if mode!=template.reference_mode:
                            continue
                    query=now+lookahead+template.lag_seconds
                    if query>forecast.times[-1]+1e-8:
                        continue
                    ref,v,w=_reference_at(forecast,reference_id,template,query,origin)
                    target=_world_target(ref,v,w,template,1.)[0]
                    present=_world_target(current.pose,current.linear_velocity,current.angular_velocity,template,1.)[0]
                    rows.append(dict(phase=phase,template=template,reference_id=reference_id,
                        manipulated_ids=(manipulated_id,),pose=target,present_pose=present,
                        opening=float(template.at(1.)[3]),selected_at=now,
                        predicted_reference_time=query,lookahead_seconds=lookahead,
                        score=-float(np.linalg.norm(target.position-state.robot.pose.position))))
                except (KeyError,ValueError,IndexError):
                    continue
    return sorted(rows,key=lambda r:(-r['score'],r['reference_id'],r['manipulated_ids'],r['template'].anchor_id))


def current_completion_pose(frontend,goal,*,origin=None):
    """Original current public canonical geometry, separate from future intent."""
    c=frontend.actor.controller;template=goal['template']
    current=c.generator.observed_reference(frontend.state,goal['reference_id'],template,origin)
    if template.reference_time_basis==COMMAND_GOAL:
        canonical=Pose.from_xyzw(template.completion_geometry_xyzw)
        return Pose(current.pose.apply(canonical.position),current.pose.rotation@canonical.rotation)
    return _world_target(current.pose,np.zeros(3),np.zeros(3),template,1.)[0]
