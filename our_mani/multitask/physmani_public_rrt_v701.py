"""Use original native RRT/RML with a public collision scene for approach and acquisition.

The author RML profile/waypoint filter/gripper remain intact. Geometry methods
may ask the native RRT to avoid their observed obstacle ghosts, rather than
repeatedly vetoing the same collision-ignoring path. Fresh per-target sweeps
remain authoritative. This static planning snapshot is not a space-time proof.
"""
from copy import deepcopy
from pathlib import Path
from .physmani_skill_executor_v534 import PhysManiSkillExecutor as OriginalSkillExecutor
from .evaluation import file_sha
from .protocol import MultitaskProtocolError,MultitaskInfrastructureError
from ..sensor_alignment.physmani_executor_v6 import InfrastructureExecutionError


def binding(root):
    return dict(schema='r10_public_ghost_approach_acquire_original_native_RRT_v701',
        source_sha256=file_sha(Path(root)/'our_mani/multitask/physmani_public_rrt_v701.py'),
        phases=['approach','acquire'],require_no_public_attachments=True,ignore_collisions=False,
        scene='same_current_public_forecast_at_actual_pose_intent_query_time',
        forecast_scene_time_matches_target_query=True,author_wp0_current_pose_uses_zero_delay=True,
        original_RRT_algorithm_and_search_parameters=True,original_RML_parameters_and_waypoint_filter=True,
        per_RML_target_current_public_geometry_sweep_retained=True,
        original_baseline_algorithm_unchanged=True,actual_motion_bound_not_claimed=True)


class PublicRRTSkillExecutor(OriginalSkillExecutor):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.public_rrt_events=[]

    def execute(self,env,action,*,observer=None):
        service=env.map_service;original=service.plan_path;policy=self.skill_policy
        def plan(position,quaternion,**kwargs):
            f=policy.frontend;chosen=policy.selected
            if chosen is None or chosen['phase'] not in ('approach','acquire') or f.actor.bound_attachments:
                return original(position,quaternion,**kwargs)
            state,forecast=f.state,f.forecast;c=f.actor.controller;port=f.actor.collision_port
            c.generator.bind_candidate_scene(state,forecast,chosen['phase'],chosen['reference_id'],chosen['manipulated_ids'],
                                             float(state.robot.timestamp)+.05)
            now=float(state.robot.timestamp)
            query=now if chosen.get('author_substage')=='wp0' else float(chosen['predicted_reference_time'])
            if not now-1e-8<=query<=float(forecast.times[-1])+1e-8:
                raise ValueError('same current public forecast must support the actual RML target query time')
            poses,_=port._poses(max(0.,query-now));modes,_=port._geometry_modes()
            allowed=set(port.bound['allowed_robot_contact_ids'])
            planning_poses={k:v for k,v in poses.items() if k not in allowed}
            event=dict(time=float(state.robot.timestamp),phase=chosen['phase'],
                public_obstacle_ids=sorted(planning_poses),allowed_robot_contact_ids=sorted(allowed),
                target_query_time=query,obstacle_query_time=query,forecast_offset_seconds=max(0.,query-now),
                forecast_observed_until=forecast.metadata['observed_until'],
                ignore_collisions=False,original_RML_parameters_changed=False,
                live_dynamic_native_poses_read=False,space_time_collision_certificate_claimed=False)
            options=dict(kwargs,ignore_collisions=False);service.path_queries+=1
            try:
                with service.public_scene(planning_poses,include_attachments=False,geometry_modes=modes):
                    path=env.arm.get_path(position,quaternion=quaternion,**options)
                event['path_found']=True
                return path
            except (MultitaskProtocolError,MultitaskInfrastructureError) as error:
                event['infrastructure_error']=type(error).__name__
                raise InfrastructureExecutionError('public native RRT scene/cleanup failed') from error
            except Exception as error:
                event['path_found']=False;event['planner_error']=type(error).__name__
                event['planner_error_message']=str(error)[:500]
                event['planner_cause_type']=None if error.__cause__ is None else type(error.__cause__).__name__
                event['planner_cause_message']=None if error.__cause__ is None else str(error.__cause__)[:500]
                raise
            finally:self.public_rrt_events.append(event)
        service.plan_path=plan
        try:return super().execute(env,action,observer=observer)
        finally:service.plan_path=original

    def geometry_receipt(self):
        return dict(super().geometry_receipt(),public_geometry_RRT_v701=deepcopy(self.public_rrt_events))
