"""Public RGBD/GRU pregrasp-only canary, with no closure or task-success claim.

Robot sensors, current public estimates and control memory are policy inputs.
Privileged object/held/success values are handled only by the inherited separate
evaluation logger. Native static collision queries remain explicit execution
checks and do not qualify a formal public benchmark.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass,asdict
from pathlib import Path
import time
import numpy as np

from .public_shadow import PublicShadowBackend,sanitized_robot
from .public_state import PublicStateAssembler
from .public_motion_bridge import PublicRobotSensors
from .response_factory import build_planner as build_response_planner
from .forecast_control import BoundForecastExecutionModel,ForecastJointGoalCEMPlanner
from ..contracts import ObjectState
from ..local_mpc_20261102.candidates import GeometricCandidates
from ..local_mpc_20261102 import mpc
from ..local_mpc_20261102.records import append,write_new,plain,quantiles


ROOT=Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PublicPregraspConfig:
    position_tolerance_m: float = .02
    angle_tolerance_rad: float = .2
    dwell_seconds: float = .1
    sensor_dt: float = .05
    def __post_init__(self):
        if (not np.isfinite(self.position_tolerance_m) or not 0<self.position_tolerance_m<=.02
                or not np.isfinite(self.angle_tolerance_rad) or not 0<self.angle_tolerance_rad<=.2
                or not np.isfinite(self.dwell_seconds) or self.dwell_seconds<.1 or self.sensor_dt!=.05):
            raise ValueError('pregrasp canary requires <=2cm/<=.2rad for >=.1s on real50ms samples')


class PublicPregraspMetric:
    """Current public geometry, continuous across actual 50ms frames only."""
    def __init__(self,config=None):
        self.config=config or PublicPregraspConfig()
        self.candidate=None;self.last_time=None;self.dwell=0.;self.latest=None
    def set_candidate(self,candidate):
        candidate=mpc.validate_candidate(candidate)
        keys=('object_tcp_position','object_tcp_quaternion','approach_direction','pregrasp_offset')
        same=self.candidate is not None and all(np.allclose(candidate[k],self.candidate[k],atol=1e-10,rtol=0) for k in keys)
        self.candidate=deepcopy(candidate)
        # Different deadline IDs with the same current geometry need not erase
        # observed dwell. Different geometry cannot inherit a prior success.
        if not same:self.last_time=None;self.dwell=0.;self.latest=None
        elif self.latest is not None:self.latest['candidate_id']=candidate['id']
    def observe(self,result,robot):
        now=float(robot.timestamp)
        if self.last_time is not None and abs(now-self.last_time)<=1e-9:return self.latest
        if self.last_time is not None and now<self.last_time-1e-9:raise ValueError('public metric clock moved backwards')
        contiguous=self.last_time is not None and np.isclose(now-self.last_time,self.config.sensor_dt,atol=1e-8,rtol=0)
        old_good=bool(self.latest and self.latest['within_region'])
        available=(self.candidate is not None and isinstance(result,dict) and result.get('mode')=='FREE'
            and result.get('available') is True and result.get('free_forecast_available') is True
            and result.get('Native_inputs_used') is False and result.get('future_observations_used') is False
            and result.get('observations_fabricated') is False and abs(float(result.get('timestamp',np.inf))-now)<=1e-8)
        obj=result.get('object_observation') if isinstance(result,dict) else None
        available=available and isinstance(obj,ObjectState) and obj.visible and abs(obj.timestamp-now)<=1e-8
        distance=angle=None;good=False
        if available:
            offset=self.candidate['object_tcp_position']+self.candidate['approach_direction']*self.candidate['pregrasp_offset']
            target=obj.pose.apply(offset)
            target_q=mpc.quaternion(mpc.quat_multiply(obj.pose.xyzw()[3:],self.candidate['object_tcp_quaternion']))
            distance=float(np.linalg.norm(robot.tcp_position-target))
            angle=float(np.linalg.norm(mpc.quat_log(mpc.quat_multiply(target_q,mpc.quat_inverse(robot.tcp_quaternion)))))
            good=distance<=self.config.position_tolerance_m+1e-12 and angle<=self.config.angle_tolerance_rad+1e-12
        self.dwell=self.dwell+self.config.sensor_dt if good and contiguous and old_good else 0.
        self.last_time=now
        self.latest=dict(timestamp=now,available=bool(available),candidate_id=self.candidate['id'] if self.candidate else None,
            distance_m=distance,angle_rad=angle,within_region=bool(good),dwell_seconds=float(self.dwell),
            public_pregrasp_reached=bool(good and self.dwell>=self.config.dwell_seconds-1e-8),
            Native_object_held_success_used=False,orientation_observed=False)
        return deepcopy(self.latest)
    @property
    def passed(self):return bool(self.latest and self.latest['public_pregrasp_reached'])
    def receipt(self):return dict(criteria=asdict(self.config),latest=deepcopy(self.latest),task_success_claimed=False)


class ObservedPublicBackend(PublicShadowBackend):
    """One real RGBD + whitelisted robot observation per actual physical tick."""
    def __init__(self,backend,output,*,assembler=None,bridge=None,recorder=None,save_frames=True):
        self.assembler=assembler if assembler is not None else PublicStateAssembler()
        self.latest_robot=None
        self.pregrasp_metric=PublicPregraspMetric()
        super().__init__(backend,output,bridge=bridge,recorder=recorder,save_frames=save_frames,
                         controller_uses_public_forecast=True)
    def sample(self,state=None):
        # Full raw state is forwarded only to super's separate evaluation label
        # writer. The policy-side assembler receives this explicit robot type.
        current=self.backend.read_state() if state is None else state
        robot=sanitized_robot(current)
        result=super().sample(current)
        self.assembler.observe_robot(robot)
        self.latest_robot=robot
        self.pregrasp_metric.observe(result,robot)
        return result
    def receipt(self):
        result=super().receipt()
        result.update(public_pregrasp_control_only=True,Native_object_held_success_policy_inputs=False,
                      robot_observation_period_seconds=.05,extra_physical_ticks=0)
        return result


def qualify_forecast_pregrasp_candidates(backend,state,candidates,bound,helper):
    """GRU pose at original fixed tau + approach offset, one K8 DLS batch.

    Expired/out-of-domain candidates keep their identities and deadlines. Their
    finite anchor placeholders are masked infeasible, never sampled in the past
    or advertised as a new goal. Only feasible rows receive the original Native
    static collision probe. This is an explicitly privileged execution oracle.
    """
    started=time.perf_counter();values=[mpc.validate_candidate(c) for c in candidates]
    if len(values)!=8 or len({c['id'] for c in values})!=8:raise ValueError('exactly eight distinct public candidates required')
    poses=[];times=[];valid=[];reasons=[]
    for candidate in values:
        at=candidate['deadline'];ok=bound.anchor_time-1e-9<=at<=bound.forecast.times[-1]+1e-9
        if ok:pose=bound.sample(at)[0]
        else:pose=bound.sample(bound.anchor_time)[0]
        offset=candidate['object_tcp_position']+candidate['approach_direction']*candidate['pregrasp_offset']
        position=pose.apply(offset)
        quaternion=mpc.quaternion(mpc.quat_multiply(pose.xyzw()[3:],candidate['object_tcp_quaternion']))
        poses.append((position,quaternion));times.append(at);valid.append(ok)
        reasons.append('in_current_forecast_domain' if ok else 'expired_or_beyond_current_forecast_domain')
    q=np.asarray(state['q'],float)
    if q.shape!=(7,) or not np.isfinite(q).all():raise ValueError('current public q7 required')
    solved=helper.adapter.solve(np.broadcast_to(q,(8,7)).copy(),np.stack([p[0] for p in poses]),np.stack([p[1] for p in poses]))
    qualified=[];receipts=[]
    for i,candidate in enumerate(values):
        feasible=bool(valid[i] and solved['joint_goal_feasible'][i]);collision=None
        before=time.perf_counter()
        if feasible:collision=bool(backend.probe_joint_collision(solved['joint_goal'][i]))
        value=deepcopy(candidate);value['native_pose_qualified']=bool(feasible and not collision)
        qualified.append(value)
        receipts.append(dict(id=candidate['id'],deadline=times[i],deadline_in_current_forecast=bool(valid[i]),
            deadline_status=reasons[i],target_position=poses[i][0],target_quaternion=poses[i][1],
            finite_placeholder_only=not valid[i],algebraic_IK_available=feasible,
            static_arm_collision=collision,qualified=value['native_pose_qualified'],
            position_error_m=float(solved['ik_position_error'][i]),angle_error_rad=float(solved['ik_angle_error'][i]),
            collision_wall_seconds=time.perf_counter()-before,physical_steps=0,Native_IK_calls=0,
            target_source='current_GRU_fixed_absolute_tau_pregrasp_pose',
            Native_static_collision_oracle_used_for_execution_check=feasible,formal_qualification=False))
    return qualified,dict(cases=receipts,poses=8,batch_DLS_calls=1,wall_seconds=time.perf_counter()-started,
                          candidate_count_preserved=True,deadline_shifted=False,CV_target_used=False)


def _safe_open(executor,robot,reason):
    return executor.execute_joint_target(np.asarray(robot.q).copy(),opening=1.,phase='P3_public_safe_open_'+reason)


def run_public_pregrasp(backend,executor,output,config,*,recorder=None):
    """100ms public decisions; success is only stable current pregrasp geometry.

    The caller wraps backend BEFORE binding it to the executor. GT labels are
    written by the observer separately; neither backend.read_state/terminal nor
    evaluation labels are read here to plan, compute success, or stop success.
    """
    if not isinstance(backend,ObservedPublicBackend) or getattr(executor,'backend',None) is not backend:
        raise ValueError('ObservedPublicBackend must be the executor backend so every actual tick is observed')
    if config.get('closure_permitted',False) is not False:raise ValueError('closure cannot be enabled in a public pregrasp-only canary')
    if float(config.get('control_dt',.1))!=.1 or float(config.get('physics_dt',.05))!=.05:
        raise ValueError('two real50ms ticks per100ms public decision required')
    if backend.samples==0:backend.sample()
    output=Path(output)
    criteria=PublicPregraspConfig(config.get('public_pregrasp_position_tolerance_m',.02),
        config.get('public_pregrasp_angle_tolerance_rad',.2),config.get('public_pregrasp_dwell_seconds',.1))
    backend.pregrasp_metric=PublicPregraspMetric(criteria)
    core,helper,provenance=build_response_planner(ROOT,executor.limits,controller='coupled',action_space='joint',
        joint_goal_config={'lower':tuple(config.get('workspace_lower',[-.1,-.65,.755]))},
        mpc_config={'approach_enter':criteria.position_tolerance_m,'approach_dwell':criteria.dwell_seconds,
                    'angle_enter':criteria.angle_tolerance_rad},
        terminal_config={'enabled':bool(config.get('terminal_goal_cost',True))},reference_frequency=executor.config.reference_frequency)
    provenance=deepcopy(provenance)
    provenance.update(controller_information_setting='public_estimated_pregrasp_diagnostic',
        public_pregrasp_only=True,GRU_used_by_controller=True,Transformer_used=False,TF_controller_used=False,
        Native_object_state_input_used=False,Native_held_input_used=False,Native_success_exit_used=False,Native_goal_input_used=False,
        Native_static_collision_oracle_used_for_execution_checks=True,formal_qualification=False,
        full_grasp_attempted=False,closure_permitted=False,pregrasp_criteria=asdict(criteria),orientation_observed=False,
        canary_thresholds_differ_from_P1=True,actual_phase_guard='approach_until_current_real50ms_public_pregrasp_metric_passes',
        virtual_future_align_and_close_permitted=True)
    candidate_policy=config.get('candidate_generation','legacy_400ms_grace')
    if candidate_policy=='immediate_expiry':
        from .temporal_candidates import CurrentWindowCandidates
        geometry=CurrentWindowCandidates(horizon=config.get('horizon_seconds',.8))
    elif candidate_policy=='legacy_400ms_grace':
        geometry=GeometricCandidates(horizon=config.get('horizon_seconds',.8))
    else:
        raise ValueError('unknown explicit candidate-generation policy')
    provenance['candidate_generation']=candidate_policy
    write_new(output/'public_pregrasp_provenance.json',provenance)
    warm=phase_state=None;last_generation=-1;cycles=0;plans=0;warmups=0
    stop_reason='budget_exhausted';plan_times=[];cycle_times=[];reached=False
    maximum=int(config.get('public_pregrasp_max_decisions',np.ceil(float(config.get('simulation_seconds',6.))/.1)))
    if maximum<=0:raise ValueError('positive explicit public decision budget required')
    while cycles<maximum and backend.remaining_ticks()>=2:
        if any(path.exists() for path in (output/'STOP',output.parent/'STOP',ROOT/'runtime/framework/local_mpc_20261102/STOP')):
            stop_reason='administrative_stop';break
        started=time.perf_counter();robot=backend.latest_robot;public=backend.latest
        if not isinstance(robot,PublicRobotSensors) or abs(robot.timestamp-backend.clock())>1e-8:
            raise ValueError('current real observer robot frame required at decision boundary')
        if backend.pregrasp_metric.passed:
            reached=True;stop_reason='public_pregrasp_reached';break
        actual_h=deepcopy(executor.state())
        # The initial frame has no invented TCP twist. A safe open command also
        # makes an initially nonbinary measured aperture an actual binary target;
        # no controller snapshot is relabeled or fabricated for the assembler.
        if backend.assembler.robot_twist is None or not backend.assembler.robot_twist[2]:
            state=None;unavailable='public_robot_twist_not_available'
        else:
            state=backend.assembler.build(public,robot,actual_h,phase_state=phase_state)
            unavailable='public_state_unavailable' if state is None else None
        if state is not None and (public.get('mode')!='FREE' or public.get('free_forecast_available') is not True):
            state=None;unavailable='current_FREE_GRU_unavailable'
        result=qualification=action_qualification=None;proposed=None;selected=None;reject=None;must_stop=False
        if state is None:
            warm=None;phase_state=None;warmups+=1
            receipt=_safe_open(executor,robot,unavailable)
            executed=dict(joint_goal=robot.q.copy(),opening=1.,source=unavailable)
        else:
            # 100ms update_phase alone can credit .1s at the first good frame.
            # The canary's actual phase therefore remains approach until the
            # separate real50ms public metric has passed. This is an explicit
            # pregrasp policy constraint, not an after-the-fact action override.
            state['phase_state']=dict(phase='approach',dwell=backend.pregrasp_metric.dwell,
                                      lost_hold=0.,updated_at=robot.timestamp)
            phase_state=deepcopy(state['phase_state'])
            if warm and warm.get('sequence'):
                elapsed=max(0,int(round((robot.timestamp-float(warm['time']))/core.config.dt)))
                first=min(elapsed,len(warm['sequence'])-1)
                if int(warm['sequence'][first].get('nominal_kind',-1))!=0:
                    warm=None
            candidates=geometry.update(state)
            if geometry.generation!=last_generation:warm=None;last_generation=geometry.generation
            bound=BoundForecastExecutionModel.from_public_bridge(core.execution_model,public)
            candidates,qualification=qualify_forecast_pregrasp_candidates(backend,state,candidates,bound,helper)
            planner=ForecastJointGoalCEMPlanner(core.config,execution_model=bound,goal_helper=helper,
                constraint_cost=core.planner.constraint_cost,terminal_config=core.terminal_config)
            before=time.perf_counter();result=planner.plan(state,candidates,warm_start=warm);plan_times.append(time.perf_counter()-before);plans+=1
            proposed=result['action'];phase_state=result['phase_state']
            if proposed is None:
                reject='no_feasible_public_forecast_candidate';warm=None
                receipt=_safe_open(executor,robot,reject);executed=dict(joint_goal=robot.q.copy(),opening=1.,source=reject)
            else:
                selected=next(c for c in candidates if c['id']==result['selected_candidate_id'])
                backend.pregrasp_metric.set_candidate(selected)
                backend.pregrasp_metric.observe(public,robot)
                if backend.pregrasp_metric.passed:
                    reached=True;stop_reason='public_pregrasp_reached';break
                opening=float(proposed.get('opening',np.nan))
                if not np.isfinite(opening) or opening!=1.:
                    reject=('closure_not_permitted_in_pregrasp_canary' if np.isfinite(opening) and opening<.5
                            else 'non_open_action_not_permitted_in_pregrasp_canary')
                    stop_reason=reject;must_stop=True;warm=None
                    receipt=_safe_open(executor,robot,reject);executed=dict(joint_goal=robot.q.copy(),opening=1.,source=reject)
                else:
                    goal,reject,action_qualification=helper.qualify_action(backend,state,proposed,require_predicted_joint_goal=True)
                    if reject:
                        warm=None;receipt=_safe_open(executor,robot,reject);executed=dict(joint_goal=robot.q.copy(),opening=1.,source=reject)
                    else:
                        if not np.array_equal(goal,np.asarray(proposed['joint_goal'])):raise ValueError('selected modeled joint goal changed before execution')
                        receipt=executor.execute_joint_target(goal,opening=opening,phase='P3_public_pregrasp_exact_joint_goal')
                        executed=dict(joint_goal=np.asarray(goal).copy(),opening=opening,source='exact_selected_model_joint_goal')
                        warm=result['warm_start']
        after=backend.latest_robot
        append(output/'public_decisions.jsonl',dict(index=cycles,timestamp=robot.timestamp,public_robot=robot,
            public_state=state,actuator_state_before=actual_h,unavailable_reason=unavailable,
            candidate_qualification=qualification,proposed_plan=result,proposed_action=proposed,action_rejected_reason=reject,
            action_qualification=action_qualification,executed_action=executed,executor=receipt,
            public_robot_after=after,actuator_state_after=deepcopy(executor.state()),public_pregrasp=backend.pregrasp_metric.receipt(),
            Native_labels_used_by_policy=False,Native_static_collision_oracle_execution_check=True,
            GRU_used_by_controller=state is not None,Transformer_used=False))
        cycles+=1;cycle_times.append(time.perf_counter()-started)
        if after is None or after.timestamp<=robot.timestamp+1e-8:
            stop_reason='executor_produced_no_real_public_tick';break
        if must_stop:break
        if backend.pregrasp_metric.passed:
            reached=True;stop_reason='public_pregrasp_reached';break
    return dict(stage='P3_public_pregrasp_only',complete=True,public_pregrasp_reached=reached,canary_stop_reason=stop_reason,
        full_grasp_attempted=False,Native_task_success_claimed=False,closure_commands_executed=0,
        controller_information_setting='public_estimated_pregrasp_diagnostic',GRU_used_by_controller=True,
        Transformer_used=False,TF_controller_used=False,Native_object_state_input_used=False,Native_held_input_used=False,
        Native_success_exit_used=False,Native_goal_input_used=False,labels_only_used_for_offline_evaluation=True,
        Native_static_collision_oracle_used_for_execution_checks=True,formal_qualification=False,precision_qualified=False,
        cycles=cycles,planned_decisions=plans,warmup_or_unavailable_decisions=warmups,
        public_pregrasp=backend.pregrasp_metric.receipt(),public_observer=backend.receipt(),
        planner_wall_seconds=quantiles(plan_times),full_cycle_wall_seconds=quantiles(cycle_times),
        real_time_guaranteed=False,controller_provenance=provenance,geometric_candidate_events=geometry.events)
