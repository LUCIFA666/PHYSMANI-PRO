"""Causal empirical position-servo compensation for simulation development.

Predicted actual q/v paths and the emitted servo setpoint are distinct. The
former use the original timed joint/sampled-collision gate; the latter obey the
shared actuator's position and velocity-step bounds. Interval samples are a
discrete screen, not a proof over every response gain or physical trajectory.
"""
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType
import numpy as np

from .evaluation import digest, file_sha
from ..reachability import TimedReachability

SCHEMA='r10_online_empirical_servo_response_v1'
ROOT=Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ServoResponseConfig:
    version:str=SCHEMA
    dt:float=.05
    minimum_samples_per_joint:int=8
    window_samples_per_joint:int=32
    minimum_excitation_rad:float=.0001
    gain_floor:float=.2
    gain_ceiling:float=1.2
    gain_interval_margin:float=.05
    maximum_gain_interval_width:float=.5
    maximum_reference_amplification:float=2.
    candidate_scales:tuple=(1.,.9,.75,.5,.25)
    velocity_proxy_consistency_tolerance:float=.0001
    observed_velocity_tolerance:float=.00001
    observed_acceleration_tolerance:float=.001

    def __post_init__(self):
        values=(self.minimum_excitation_rad,self.gain_floor,self.gain_ceiling,
            self.gain_interval_margin,self.maximum_gain_interval_width,
            self.maximum_reference_amplification,self.velocity_proxy_consistency_tolerance,
            self.observed_velocity_tolerance,self.observed_acceleration_tolerance,*self.candidate_scales)
        if (self.version!=SCHEMA or self.dt!=.05 or not np.isfinite(values).all()
                or type(self.minimum_samples_per_joint) is not int
                or type(self.window_samples_per_joint) is not int
                or not 3<=self.minimum_samples_per_joint<=self.window_samples_per_joint<=256
                or self.minimum_excitation_rad<=0 or not 0<self.gain_floor<self.gain_ceiling
                or self.gain_interval_margin<=0 or self.maximum_gain_interval_width<=0
                or self.maximum_reference_amplification<1 or not self.candidate_scales
                or any(not 0<x<=1 for x in self.candidate_scales)
                or tuple(sorted(set(self.candidate_scales),reverse=True))!=tuple(self.candidate_scales)
                or min(self.velocity_proxy_consistency_tolerance,self.observed_velocity_tolerance,
                    self.observed_acceleration_tolerance)<0):
            raise ValueError('invalid declared empirical servo configuration')


def servo_response_binding(config):
    c=config if isinstance(config,ServoResponseConfig) else ServoResponseConfig(**config)
    normalized=asdict(c);normalized['candidate_scales']=list(normalized['candidate_scales'])
    return dict(normalized_config=normalized,configuration_sha256=digest(normalized),
        source=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=file_sha(__file__)))


def actuator_reference_bounds(robot, reference, model, dt):
    """Same clipping expression as shared simulation.command_joints, no query."""
    reference=np.asarray(reference,float)
    if reference.shape!=robot.joints.shape or not np.isfinite(reference).all():
        raise ValueError('invalid servo reference')
    return np.clip(robot.joints+np.clip(reference-robot.joints,
        -model.velocity_limits*dt,model.velocity_limits*dt),model.lower,model.upper)


class OnlineServoResponseCompensator:
    def __init__(self,model,config=None):
        self.model=model
        self.config=config if isinstance(config,ServoResponseConfig) else ServoResponseConfig(**(config or {}))
        self.binding=servo_response_binding(self.config)
        self.samples=[deque(maxlen=self.config.window_samples_per_joint) for _ in model.velocity_limits]
        self.previous_observation=self.previous_issued=None
        self.last_guard=None;self.events=[];self.transitions=[]

    def estimate(self):
        c=self.config;count=[len(s) for s in self.samples]
        if min(count)<c.minimum_samples_per_joint:
            return dict(ready=False,reason='insufficient_causal_excited_samples',counts=count)
        median=[];lower=[];upper=[]
        for samples in self.samples:
            g=np.asarray([s['gain'] for s in samples]);lo,mid,hi=np.quantile(g,[.1,.5,.9])
            median.append(float(mid));lower.append(float(max(c.gain_floor,lo-c.gain_interval_margin)))
            upper.append(float(min(c.gain_ceiling,hi+c.gain_interval_margin)))
        ready=all(c.gain_floor<=m<=c.gain_ceiling and h-l<=c.maximum_gain_interval_width
                  for m,l,h in zip(median,lower,upper))
        return dict(ready=ready,reason='ready_empirical' if ready else 'wide_or_invalid_gain_interval',
            counts=count,median=median,lower=lower,upper=upper,
            observed_until=None if self.previous_observation is None else self.previous_observation.timestamp,
            fit_source='past_issued_reference_and_already_observed_next_q_and_reported_v',
            velocity_model='next reported joint velocity equals backward q difference within declared tolerance',
            confidence_interval_is_empirical_not_statistical_guarantee=True)

    def observe(self,robot):
        c=self.config;previous=self.previous_observation
        if previous is not None and robot.timestamp==previous.timestamp:
            if not (np.array_equal(robot.joints,previous.joints)
                    and np.array_equal(robot.joint_velocities,previous.joint_velocities)):
                raise ValueError('one timestamp cannot identify two different servo observations')
            return deepcopy(self.last_guard)
        if previous is not None and robot.timestamp<previous.timestamp:
            raise ValueError('servo observations must arrive causally in increasing time')
        violations=[]
        if np.any(robot.joints<self.model.lower-1e-7) or np.any(robot.joints>self.model.upper+1e-7):
            violations.append('observed_joint_position_limit')
        if np.any(abs(robot.joint_velocities)>self.model.velocity_limits+c.observed_velocity_tolerance):
            violations.append('observed_joint_velocity_limit')
        if previous is not None and abs(robot.timestamp-previous.timestamp-c.dt)<1e-8:
            acceleration=(robot.joint_velocities-previous.joint_velocities)/c.dt
            if np.any(abs(acceleration)>self.model.acceleration_limits+c.observed_acceleration_tolerance):
                violations.append('observed_joint_acceleration_limit')
        prior_model=self.estimate();issued=self.previous_issued
        if issued is not None:
            if abs(robot.timestamp-issued['timestamp']-c.dt)<1e-8:
                q0=np.asarray(issued['measured_joints']);u=np.asarray(issued['applied_reference_prediction'])
                delta=u-q0;actual_delta=robot.joints-q0
                consistent=np.all(abs(robot.joint_velocities-actual_delta/c.dt)
                    <=c.velocity_proxy_consistency_tolerance)
                row=dict(observed_at=float(robot.timestamp),issued_at=issued['timestamp'],
                    actual_delta=actual_delta.tolist(),issued_delta=delta.tolist(),
                    reported_velocity_matches_backward_difference=bool(consistent),fitted_joint_indices=[],
                    response_interval_mismatch_joint_indices=[])
                if consistent and not violations:
                    for j,d in enumerate(delta):
                        if abs(d)<c.minimum_excitation_rad:continue
                        gain=float(actual_delta[j]/d)
                        if prior_model.get('ready') and not prior_model['lower'][j]<=gain<=prior_model['upper'][j]:
                            row['response_interval_mismatch_joint_indices'].append(j)
                        if not c.gain_floor<=gain<=c.gain_ceiling:continue
                        self.samples[j].append(dict(observed_at=float(robot.timestamp),gain=gain))
                        row['fitted_joint_indices'].append(j)
                if row['response_interval_mismatch_joint_indices']:
                    violations.append('observed_empirical_response_interval_violation')
                if not consistent:
                    for samples in self.samples:samples.clear()
                row['excluded_from_fit_due_observed_violation']=bool(violations)
                self.transitions.append(row)
            self.previous_issued=None
        if violations:
            for samples in self.samples:samples.clear()
        self.previous_observation=robot
        self.last_guard=dict(observed_at=float(robot.timestamp),violations=violations,
            hold_replan=bool(violations),measured_joints=robot.joints.tolist(),
            measured_joint_velocities=robot.joint_velocities.tolist(),
            prior_estimate=prior_model,actual_tracking_guaranteed=False)
        return deepcopy(self.last_guard)

    def record_issued(self,robot,reference,*,kind):
        if self.previous_observation is None or self.previous_observation.timestamp!=robot.timestamp:
            raise ValueError('observe the current state before recording its issued reference')
        if not (np.array_equal(robot.joints,self.previous_observation.joints)
                and np.array_equal(robot.joint_velocities,self.previous_observation.joint_velocities)):
            raise ValueError('issued command state differs from its observed state')
        if self.previous_issued is not None:
            raise ValueError('one issued command per observed counted tick')
        applied=actuator_reference_bounds(robot,reference,self.model,self.config.dt)
        self.previous_issued=dict(timestamp=float(robot.timestamp),measured_joints=robot.joints.tolist(),
            issued_reference=np.asarray(reference).tolist(),applied_reference_prediction=applied.tolist(),kind=kind,
            applied_rule='same current_joint + velocity_limit*dt clip, then joint position clip as shared command_joints',
            extra_native_confirmation_query=False)

    def _gain_samples(self,estimate):
        median=np.asarray(estimate['median']);low=np.asarray(estimate['lower']);high=np.asarray(estimate['upper'])
        samples=[('median',median),('all_lower',low),('all_upper',high)]
        for j in range(len(median)):
            for name,end in (('lower',low),('upper',high)):
                gain=median.copy();gain[j]=end[j];samples.append((f'joint{j}_{name}',gain))
        return samples

    def prepare(self,robot,desired_joints,desired_velocities,reachability):
        guard=self.observe(robot);c=self.config
        q=np.asarray(desired_joints,float);v=np.asarray(desired_velocities,float)
        event=dict(schema=SCHEMA,observed_time=float(robot.timestamp),
            desired_measured_state_joints=q.tolist(),desired_measured_state_velocities=v.tolist(),
            measured_joints=robot.joints.tolist(),measured_joint_velocities=robot.joint_velocities.tolist(),
            observed_guard=guard,actual_tracking_guaranteed=False,
            sampled_interval_approximation=True,full_uncertainty_region_proved=False,
            servo_reference_original_hermite_gate_verified=False,
            additional_sensor_or_physics_queries=False,phase_completion_claimed=False,total_budget_extended=False)
        if guard['hold_replan']:
            event.update(action='hold_replan',reason='observed_motion_cap_or_response_model_violation')
            self.events.append(event);return event
        desired=TimedReachability.check_joints(reachability,robot,q,v,c.dt)
        event['desired_actual_path_original_gate']=dict(feasible=bool(desired.feasible),reason=desired.reason,evidence=deepcopy(desired.evidence))
        if not desired.feasible:
            event.update(action='hold_replan',reason='desired_bounded_path_rejected')
            self.events.append(event);return event
        estimate=self.estimate();event['response_estimate']=estimate
        if not estimate['ready']:
            event.update(action='fallback_original_bounded',reason=estimate['reason'],
                servo_reference=actuator_reference_bounds(robot,q,self.model,c.dt).tolist(),
                servo_reference_independent_bounds_verified=True,predicted_actual_samples=[])
            self.events.append(event);return event
        inverse=(q-robot.joints)/np.asarray(estimate['median'])
        inverse=np.clip(inverse,-c.maximum_reference_amplification*abs(q-robot.joints),
                        c.maximum_reference_amplification*abs(q-robot.joints))
        trials=[]
        for scale in c.candidate_scales:
            reference=actuator_reference_bounds(robot,robot.joints+scale*inverse,self.model,c.dt)
            delta=reference-robot.joints;checks=[];passed=True
            for name,gain in self._gain_samples(estimate):
                actual_q=robot.joints+gain*delta
                actual_v=gain*delta/c.dt
                result=TimedReachability.check_joints(reachability,robot,actual_q,actual_v,c.dt)
                checks.append(dict(sample=name,gain=gain.tolist(),predicted_actual_joints=actual_q.tolist(),
                    predicted_reported_velocity_proxy=actual_v.tolist(),feasible=bool(result.feasible),
                    reason=result.reason,evidence=deepcopy(result.evidence)))
                if not result.feasible:passed=False;break
            trials.append(dict(scale=float(scale),servo_reference=reference.tolist(),samples=checks,passed=passed))
            if passed:
                event.update(action='emit_compensated_reference',reason='sampled_empirical_actual_paths_pass_original_gate',
                    servo_reference=reference.tolist(),servo_reference_independent_bounds_verified=True,
                    selected_scale=float(scale),predicted_actual_samples=checks,trials=trials)
                self.events.append(event);return event
        event.update(action='hold_replan',reason='no_empirical_interval_sampled_path_feasible',trials=trials)
        self.events.append(event);return event

    def metrics(self):
        return dict(schema=SCHEMA,**deepcopy(self.binding),estimate=self.estimate(),
            transitions=deepcopy(self.transitions),command_events=deepcopy(self.events),
            actual_tracking_guaranteed=False,sampled_interval_approximation=True,
            full_uncertainty_region_proved=False,no_future_or_native_truth_used=True,
            phase_completion_claimed=False,total_budget_extended=False)


def install_servo_response_compensator(actor,config):
    """Install only after the frozen feedback controller; wrap instance methods."""
    from .feedback_interaction import ObservedFeedbackController
    if type(actor.controller) is not ObservedFeedbackController or actor.controller.feedback_step_calls:
        raise ValueError('install servo compensation before the feedback actor first acts')
    compensator=OnlineServoResponseCompensator(actor.robot_model,config)
    original_act=actor.act;original_metrics=actor.metrics_report
    class DeferredSegments:
        def __init__(self,segments):self.original=segments
        def __getattr__(self,name):return getattr(self.original,name)
        def issue(self,kind):
            # The public plate mask must follow the final emitted command, not
            # an inner press proposal subsequently rejected by the outer guard.
            return np.asarray([body not in self.original.non_exogenous
                for body in self.original.body_ids],bool)
    def act(instance,observation,robot,remaining_seconds):
        guard=compensator.observe(robot)
        old_opening=getattr(instance,'requested_opening',None)
        segments=getattr(instance,'command_segments',None)
        if segments is not None:instance.command_segments=DeferredSegments(segments)
        try:command=original_act(observation,robot,remaining_seconds)
        finally:
            if segments is not None:instance.command_segments=segments
        decision=None
        if guard['hold_replan']:
            decision=dict(schema=SCHEMA,observed_time=float(robot.timestamp),action='hold_replan',
                reason='observed_motion_cap_or_response_model_violation',observed_guard=guard,
                actual_tracking_guaranteed=False,sampled_interval_approximation=True,
                full_uncertainty_region_proved=False,total_budget_extended=False)
            compensator.events.append(decision)
        elif command.get('measured_feedback_command') and command.get('command')=='joint_trajectory':
            decision=compensator.prepare(robot,command['joints'],command['joint_velocities'],
                instance.controller.original_reachability)
        if decision is not None:
            if decision['action']=='hold_replan':
                controller=instance.controller
                if controller.goal is not None:
                    phase=controller.goal.template.phase
                    controller.attempts.abandon(phase,float(robot.timestamp),'servo_response_'+decision['reason'])
                controller.goal=controller.trajectory=controller.origin=None
                controller._gripper_goal_token=controller._close_edge_issued_at=None
                instance.requested_opening=old_opening
                command=dict(command='hold',phase='servo_response_replan',reason=decision['reason'],public_command_kind='hold')
            else:
                command=dict(command,desired_measured_state_joints=np.asarray(command['joints']).copy(),
                    desired_measured_state_velocities=np.asarray(command['joint_velocities']).copy(),
                    joints=np.asarray(decision['servo_reference']),servo_response_schema=SCHEMA,
                    servo_response_action=decision['action'],servo_reference_independent_bounds_verified=True,
                    servo_reference_original_hermite_gate_verified=False,
                    desired_actual_path_constraints_verified=True,
                    nominal_constraints_verified=False,
                    joints_are_servo_setpoint_not_predicted_measured_state=True,
                    servo_reference_terminal_velocity_declared=False,
                    actual_tracking_guaranteed=False,sampled_interval_approximation=True)
                command.pop('joint_velocities',None)
            current_feedback=[event for event in instance.controller.feedback_events
                if event.get('observed_time')==float(robot.timestamp)]
            current_gripper=[event for event in instance.controller.gripper_timing_events
                if event.get('time')==float(robot.timestamp)]
            decision['inner_feedback_proposals']=deepcopy(current_feedback)
            decision['inner_gripper_proposals']=deepcopy(current_gripper)
            for event in current_feedback:
                original_issued=bool(event.get('command_issued'))
                same_reference=(command['command']=='joint_trajectory'
                    and np.allclose(command['joints'],event['bounded_joints'],rtol=0,atol=1e-12))
                event.update(inner_command_proposed=original_issued,
                    command_issued=bool(original_issued and same_reference),
                    actual_servo_reference_issued=command['command']=='joint_trajectory',
                    outer_command_action=decision['action'],
                    actual_servo_reference=None if command['command']!='joint_trajectory'
                        else np.asarray(command['joints']).tolist())
            for event in current_gripper:
                event.update(inner_opening_proposal=event.get('commanded_opening'),
                    commanded_opening=None if command['command']!='joint_trajectory'
                        else float(command['opening']),
                    actual_opening_request_issued=command['command']=='joint_trajectory',
                    actual_opening_request=None if command['command']!='joint_trajectory'
                        else float(command['opening']),outer_returned_command=command['command'],
                    hold_retains_previous_native_opening_request=command['command']=='hold')
            if current_feedback:
                instance.controller.last_diagnostics['measured_feedback_execution']=deepcopy(current_feedback[-1])
                if isinstance(instance.last_diagnostics.get('controller'),dict):
                    instance.last_diagnostics['controller']['measured_feedback_execution']=deepcopy(current_feedback[-1])
            instance.last_diagnostics['online_servo_response']=deepcopy(decision)
        if segments is not None:segments.issue(command.get('public_command_kind',command['command']))
        events=getattr(instance,'events',None)
        if events and events[-1].get('time')==float(robot.timestamp):
            if decision is not None:decision['inner_public_event']=deepcopy(events[-1])
            events[-1].update(command=command['command'],phase=command.get('phase'),
                public_command_kind=command.get('public_command_kind',command['command']),
                reason=command.get('reason'))
        instance.last_command=command
        reference=command['joints'] if command.get('command')=='joint_trajectory' else robot.joints
        compensator.record_issued(robot,reference,kind=command['command'])
        return command
    def metrics(instance):
        result=dict(original_metrics());result['online_servo_response']=compensator.metrics();return result
    actor.act=MethodType(act,actor);actor.metrics_report=MethodType(metrics,actor)
    actor.servo_response_compensator=compensator
    return actor
