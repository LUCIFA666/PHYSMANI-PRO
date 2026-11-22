"""Observable candidate features and attempt-level outcome accounting.

An attempt succeeds only if the native task succeeds before this candidate is
abandoned or replaced. Initial approach, actual arrival and remaining transport
budget are part of the experiment. No ideal local reset or future truth input.
"""
from collections import Counter
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
from ..contracts import InterceptGoal
from ..intercept_planner import CandidateInterception


def attempt_controller_hash(root,config,predictor):
    import hashlib
    import json
    from pathlib import Path
    from ..aligned.protocol import controller_hash
    digest=hashlib.sha256(controller_hash(root,config,predictor).encode())
    digest.update(Path(__file__).read_bytes())
    for name in ('intercept_planner.py','planning.py','reachability.py'):
        digest.update((Path(root)/'our_mani'/name).read_bytes())
    keys=('dt','simulation_seconds','forecast_seconds','warmup_ticks','global_joint_acceleration',
        'joint_servo_lead','local_servo_lead','local_max_speed','local_max_acceleration',
        'candidate_stride','candidate_regions','region_radii','relative_speed_limit','orientation_limit',
        'preposition_height','replan_interval_seconds','maximum_forecast_deviation')
    digest.update(json.dumps({k:config[k] for k in keys},sort_keys=True).encode())
    return digest.hexdigest()


def candidate_features(forecast,region,robot,goal_pose,arrival,pose,velocity,check,config,predictor):
    now=float(robot.timestamp)
    current,current_v,_=forecast.at(region.object_id,now)
    future,future_v,_=forecast.at(region.object_id,arrival)
    remaining=max(0.,config['simulation_seconds']-now)
    bounds=np.array([[-.03,-.53,.74],[.43,.53,.81]])
    margins=np.r_[future.position[:2]-bounds[0,:2],bounds[1,:2]-future.position[:2]]
    orientation=Rotation.from_matrix(pose.rotation@robot.pose.rotation.T).as_rotvec()
    history=float(forecast.metadata.get('history_frames',1))/20.
    feature=np.r_[current.position,current_v,robot.pose.position-current.position,
        robot.linear_velocity-current_v,robot.angular_velocity,robot.joints,robot.joint_velocities,
        future.position,future_v,pose.position-future.position,velocity-future_v,orientation,
        check.joints-robot.joints,check.joint_velocities,goal_pose.position-pose.position,
        arrival-now,remaining,remaining-(arrival-now),history,margins,
        check.evidence['maximum_speed_fraction'],check.evidence['maximum_acceleration_fraction']]
    context=None
    if predictor=='context':
        context=np.asarray(forecast.metadata.get('context'),dtype=float)
        if context.shape!=(16,) or not forecast.metadata.get('context_encoder_sha256'):
            raise ValueError('candidate probability requires the current causal context')
        feature=np.r_[feature,context]
    if not np.isfinite(feature).all():raise ValueError('nonfinite candidate probability feature')
    return feature.astype(np.float32)


class AttemptSelector:
    def __init__(self,owner,original,model=None,exploration='geometry',seed=0):
        self.owner,self.reachability=owner,original.reachability
        self.relative_velocity_options=original.relative_velocity_options
        self.object_position_bounds=original.object_position_bounds
        self.latency_reserve=original.latency_reserve
        self.model,self.exploration=model,exploration
        self.rng=np.random.default_rng(seed)
        self.next_goal_id=0;self.last_diagnostics={}

    def select(self,forecast,regions,robot,now,*,min_horizon=.15,stride=2):
        candidates=[];rejected=Counter()
        for region in regions:
            if region.object_id not in forecast.object_ids:
                rejected['region_target_missing']+=1;continue
            for arrival in forecast.times[1::stride]:
                duration=arrival-now-self.latency_reserve
                if duration<min_horizon:
                    rejected['insufficient_remaining_horizon']+=1;continue
                obj_pose,obj_v,obj_w=forecast.at(region.object_id,arrival)
                bounds=self.object_position_bounds.get(region.object_id)
                if bounds is not None and (np.any(obj_pose.position<bounds[0]) or np.any(obj_pose.position>bounds[1])):
                    rejected['forecast_outside_declared_target_workspace']+=1;continue
                pose,surface_v=region.in_world(obj_pose,obj_v,obj_w)
                for relative in self.relative_velocity_options:
                    if np.linalg.norm(relative)>region.max_relative_speed:
                        rejected['relative_speed_outside_region']+=1;continue
                    velocity=surface_v+pose.rotation@relative
                    check=self.reachability.check(robot,pose,velocity,obj_w,duration)
                    if not check.feasible:
                        rejected[check.reason]+=1;continue
                    effort=float(np.linalg.norm(check.joints-robot.joints))
                    geometry_score=-.10*(arrival-now)-.02*effort
                    feature=candidate_features(forecast,region,robot,self.owner.goal_pose,arrival,pose,velocity,check,
                        self.owner.config,self.owner.predictor)
                    candidates.append(dict(arrival=float(arrival),region=region,pose=pose,velocity=velocity,
                        angular_velocity=obj_w,check=check,geometry_score=geometry_score,features=feature))
        self.last_diagnostics=dict(feasible_candidates=len(candidates),rejections=dict(rejected))
        if not candidates:return None
        geometry=np.array([c['geometry_score'] for c in candidates])
        probabilities=None;fallback=None;propensity=None
        if self.model is not None:
            predictions=self.model.predict(np.stack([c['features'] for c in candidates]),self.owner.config['_episode_task'])
            supported=predictions['supported']
            probabilities=predictions['probability']
            if supported.any():
                scores=np.where(supported,probabilities,-np.inf)
                # Geometry is only a deterministic tie break for equal scores.
                selected=int(np.argmax(scores+1e-7*geometry))
            else:
                selected=int(np.argmax(geometry));fallback='all_candidates_outside_TRAIN_support_geometry_fallback'
            self.last_diagnostics.update(supported_candidates=int(supported.sum()),fallback=fallback)
        elif self.exploration=='uniform':
            selected=int(self.rng.integers(len(candidates)));propensity=1./len(candidates)
        elif self.exploration=='softmax':
            distribution=np.exp((geometry-geometry.max())/.08);distribution/=distribution.sum()
            selected=int(self.rng.choice(len(candidates),p=distribution));propensity=float(distribution[selected])
        else:
            selected=int(np.argmax(geometry));propensity=1.
        c=candidates[selected];probability=float(probabilities[selected]) if probabilities is not None and not fallback else None
        self.next_goal_id+=1
        evaluated_region=replace(c['region'],value=probability) if probability is not None else c['region']
        goal=InterceptGoal(self.next_goal_id,float(forecast.times[0]),c['arrival'],evaluated_region,
            c['pose'],c['velocity'],c['angular_velocity'],forecast,c['check'],
            probability if probability is not None else c['geometry_score'])
        self.last_diagnostics.update(selected_success_probability=probability,
            score_kind='attempt_success_with_actual_approach_and_remaining_episode_budget' if probability is not None else 'geometry',
            context_in_success_head=self.owner.predictor=='context')
        self.owner.start_attempt(goal,c['features'],robot,propensity,len(candidates),probability)
        return goal


class AttemptInterception(CandidateInterception):
    def __init__(self,dynamics,model,collision_scene,goal_pose,config,*,predictor,ranking,value=None):
        super().__init__(dynamics,model,collision_scene,goal_pose,config,predictor=predictor,ranking=ranking,value=None)
        self.attempts=[];self.active_attempt=None
        self.selector=AttemptSelector(self,self.selector,model=value,
            exploration=config.get('_exploration','geometry'),seed=config.get('_collection_seed',0))
        config.get('_workflow_sink',[]).append(self)

    def close_attempt(self,success,reason,time):
        if self.active_attempt is None:return
        self.active_attempt.update(official_task_success=bool(success),outcome_observed_until=float(time),
            outcome_reason=reason,complete=True)
        self.active_attempt=None

    def start_attempt(self,goal,features,robot,propensity,candidates,probability):
        self.close_attempt(False,'candidate_replaced_before_task_success',robot.timestamp)
        record=dict(attempt_id=len(self.attempts),goal_id=goal.goal_id,features=features.tolist(),
            selected_at=float(robot.timestamp),arrival_time=goal.arrival_time,
            remaining_episode_budget_seconds=max(0.,self.config['simulation_seconds']-robot.timestamp),
            capture_transport_budget_after_arrival=max(0.,self.config['simulation_seconds']-goal.arrival_time),
            context_source='current_past_observations' if self.predictor=='context' else 'CV_no_learned_context',
            context_encoder_sha256=goal.forecast.metadata.get('context_encoder_sha256'),
            region_id=goal.region.region_id,selection_propensity=propensity,feasible_candidates=candidates,
            predicted_probability=probability,complete=False)
        self.attempts.append(record);self.active_attempt=record

    def step(self,robot,obj,remaining_seconds):
        previous_stage=self.stage
        decision=super().step(robot,obj,remaining_seconds)
        if previous_stage=='local' and self.stage!='local':
            self.close_attempt(False,'local_capture_or_transport_failed',robot.timestamp)
        elif self.stage=='failed':
            self.close_attempt(False,'episode_or_attempt_budget_expired',robot.timestamp)
        elif self.active_attempt is not None and self.goal is None and self.stage!='local':
            self.close_attempt(False,'candidate_abandoned',robot.timestamp)
        return decision
