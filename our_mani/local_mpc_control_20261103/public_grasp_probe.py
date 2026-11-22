"""Bounded public attachment verification, distinct from a held/lift claim.

A probe is admitted only when the unchanged public estimator currently passes
EVERY non-motion gate. The trigger-time measured TCP is its immutable anchor;
the fixed world-z reference is 12mm above it, with a 15mm command/prediction
radius and 0.04rad attitude bound. No object displacement is presumed. The
robot model is rolled out by CEM; selected q/opening are not edited afterwards.
"""
from copy import deepcopy
from dataclasses import dataclass,asdict
import hashlib,json,time
import numpy as np
from ..contracts import ObjectState
from .public_motion_bridge import PublicRobotSensors
from .public_attachment import AttachmentConfig
from .joint_goal_cem import JointGoalCEMPlanner,norm,qlog,qmul,qinv
from ..local_mpc_p2_20261103.action_adapter import matrix_quaternion
from ..local_mpc_p2_20261103.execution_models import validate_step_batch

MOTION_GATES=frozenset(('actual_TCP_motion_witness','visual_object_motion_witness'))
NONMOTION_GATES=frozenset(('real_consecutive_feedback_window','stable_relative_translation',
    'orientation_consistency','target_in_declared_grasp_neighborhood','nonempty_grasp_opening',
    'fingers_consistent','stalled_opening','closure_actually_observed'))


@dataclass(frozen=True)
class VerificationProbeConfig:
    lift_m:float=.012
    maximum_displacement_m:float=.015
    maximum_rotation_rad:float=.04
    position_scale_m:float=.003
    terminal_weight:float=5.
    def __post_init__(self):
        if not all(np.isfinite(v) and v>0 for v in asdict(self).values()):raise ValueError('positive finite probe geometry/cost required')
        if not .010<=self.lift_m<=.015 or not self.lift_m<=self.maximum_displacement_m<=.015:
            raise ValueError('10-15mm probe and <=15mm absolute trigger-anchor radius required')
        if self.maximum_rotation_rad>.04:raise ValueError('probe attitude cannot exceed original unobservable-orientation gate .04rad')


class BoundedVerificationProbe:
    """Once per close attempt; failed gates never create attachment evidence."""
    def __init__(self,config=None):
        self.config=config or VerificationProbeConfig();self.reset()
    def reset(self):
        self.spec=None;self.used=False;self.active=False;self.reason='not_started'
        self.last_feedback_time=None;self.abort_feedback_time=None
    def abort(self,reason):
        if self.active:self.reason=reason
        self.active=False
    def _current(self,result,robot,closed):
        a=result.get('attachment_estimate',{});diag=a.get('diagnostic',{}) if isinstance(a,dict) else {}
        gates=diag.get('gates',{});target=result.get('object_observation')
        current=(isinstance(target,ObjectState) and target.visible and target.confidence>=.5
            and abs(target.timestamp-robot.timestamp)<1e-8 and abs(float(result.get('timestamp',np.inf))-robot.timestamp)<1e-8
            and isinstance(a,dict) and abs(float(a.get('timestamp',np.inf))-robot.timestamp)<1e-8
            and a.get('mode')=='UNRESOLVED' and result.get('mode')=='UNRESOLVED'
            and closed)
        exact_gates=isinstance(gates,dict) and set(gates)==NONMOTION_GATES|MOTION_GATES
        nonmotion=exact_gates and all(gates[key] is True or isinstance(gates[key],np.bool_) and bool(gates[key]) for key in NONMOTION_GATES)
        # Reuse the unchanged estimator's declared thresholds, including actual
        # current fingers rather than treating a historical window as current.
        ac=AttachmentConfig(**a['config']) if isinstance(a,dict) and a.get('config')==asdict(AttachmentConfig()) else None
        actual_fingers=(ac is not None and np.all(robot.finger_open_amounts>=ac.opening_nonempty_min)
            and np.all(robot.finger_open_amounts<=ac.opening_grasp_max)
            and abs(float(robot.finger_open_amounts[0]-robot.finger_open_amounts[1]))<=ac.maximum_finger_asymmetry)
        return bool(current and nonmotion and actual_fingers),diag,gates
    def _actual_inside(self,robot):
        displacement=np.linalg.norm(robot.tcp_position-self.spec['anchor_position'])
        angle=np.linalg.norm(qlog(qmul(robot.tcp_quaternion,qinv(self.spec['anchor_quaternion']))))
        return displacement<=self.config.maximum_displacement_m+1e-8 and angle<=self.config.maximum_rotation_rad+1e-8
    def observe_feedback(self,result,robot,*,close_requested):
        """Latch real50ms failures; never dispatch or pretend to stop midmacro.

        Executor h is updated after this callback, so this monitor reads only
        actual public feedback/intent. Fresh h is required separately at each
        100ms admission/decision. A latch is consumed at the next decision.
        """
        if not self.active:return
        if not isinstance(robot,PublicRobotSensors) or type(close_requested) is not bool:
            raise TypeError('actual public feedback and observed command intent required')
        gap=robot.timestamp-self.last_feedback_time
        self.last_feedback_time=robot.timestamp
        # Evidence becoming confirmed cannot erase a same-frame measurement
        # bound/gap failure. Ordinary held control is a later, separate mode.
        if abs(gap-.05)>1e-6 or not self._actual_inside(robot):
            self.abort_feedback_time=robot.timestamp
            self.abort('real50ms_probe_gap_or_actual_bound_failure_latched_until_next_control_boundary');return
        if result.get('mode')=='ATTACHED_PUBLIC':
            self.abort('current_public_attachment_confirmation_ends_probe');return
        ok,_,_=self._current(result,robot,close_requested)
        if not ok:
            self.abort_feedback_time=robot.timestamp
            self.abort('real50ms_probe_gate_or_bound_failure_latched_until_next_control_boundary')
    def consider(self,result,robot,h):
        if not isinstance(robot,PublicRobotSensors):raise TypeError('actual public robot frame required')
        current,diag,gates=self._current(result,robot,h.get('gripper_target')==0.)
        fresh_h=h.get('reference_clock') is not None and abs(float(h['reference_clock'])-robot.timestamp)<1e-8
        if not(current and fresh_h):
            self.abort('current_visual_robot_or_nonmotion_gate_failed')
            return None
        if self.active:
            if not self._actual_inside(robot):
                self.abort('actual_trigger_anchor_bound_exceeded');return None
            return deepcopy(self.spec)
        if self.used:return None
        if all(bool(gates[key]) for key in MOTION_GATES):return None
        self.used=True;self.active=True;self.reason='all_nonmotion_gates_pass_only_motion_unverified'
        self.last_feedback_time=robot.timestamp
        anchor=robot.tcp_position.copy();quaternion=robot.tcp_quaternion.copy()
        identity=hashlib.sha256(json.dumps(dict(time=robot.timestamp,anchor=anchor.tolist(),quaternion=quaternion.tolist(),config=asdict(self.config)),sort_keys=True).encode()).hexdigest()
        self.spec=dict(identity=identity,trigger_time=robot.timestamp,anchor_position=anchor,
            anchor_quaternion=quaternion,target_position=anchor+np.array([0.,0.,self.config.lift_m]),
            target_quaternion=quaternion.copy(),config=asdict(self.config),
            admission_diagnostic=deepcopy(diag),attachment_asserted=False,held=False,
            anchor_source='actual_TCP_at_first_all_nonmotion_gates_pass_frame',
            contact_response_qualified=False,object_follows_probe_assumed=False)
        return deepcopy(self.spec)
    def receipt(self):return dict(active=self.active,used_this_close_attempt=self.used,reason=self.reason,
        last_real50ms_feedback_time=self.last_feedback_time,abort_feedback_time=self.abort_feedback_time,
        failures_latched_to_next100ms_boundary=True,midmacro_physical_hard_stop_claimed=False,
        bound_scope='command_FK_and_both_discrete_50ms_predicted_actual_FK_plus_100ms_endpoint',
        measured_feedback_scope='real50ms_monitor_latches_failures_for_next100ms_decision',
        second_committed_tick_cancellation_guaranteed=False,continuous_physical_bound_guaranteed=False,
        physical_safety_certificate=False,spec=deepcopy(self.spec))


class VerificationProbePlanner(JointGoalCEMPlanner):
    """Same 256 joint CEM budget, explicit temporary robot verification cost.

    The object response remains UNKNOWN and unheld. In particular predicted
    object lift is never used as a benefit or an attachment assumption here.
    """
    def __init__(self,*args,probe_spec,**kwargs):
        super().__init__(*args,**kwargs);self.probe_spec=deepcopy(probe_spec)
        self.probe_config=VerificationProbeConfig(**probe_spec['config'])
    def _prepare_nominals(self,initial,candidates,old):
        started=time.perf_counter();spec=self.probe_spec
        if initial['held'] or initial['requested_opening']!=0.:raise ValueError('probe must initialize unknown/unheld and actual close request')
        positions=np.broadcast_to(spec['target_position'],(8,3)).copy();quats=np.broadcast_to(spec['target_quaternion'],(8,4)).copy()
        result=self.goal_helper.adapter.solve(np.broadcast_to(initial['q'],(8,7)).copy(),positions,quats)
        self._nominals=dict(joint_goal=np.broadcast_to(result['joint_goal'],(3,8,7)).copy(),
            feasible=np.broadcast_to(result['joint_goal_feasible'],(3,8)).copy(),
            pose=np.broadcast_to(np.c_[positions,quats],(3,8,7)).copy(),time=np.full((3,8),spec['trigger_time']))
        self.nominal_receipt=dict(DLS_calls=1,poses=8,wall_seconds=time.perf_counter()-started,
            basis_version='fixed_public_trigger_TCP_verification_probe_v1',probe=deepcopy(spec),
            original_candidate_ids_and_deadlines_preserved=True,distinct_probe_pose_modes=False)
    def _inside(self,position,quaternion):
        return ((norm(position-self.probe_spec['anchor_position'])<=self.probe_config.maximum_displacement_m+1e-10)
            &(norm(qlog(qmul(quaternion,qinv(self.probe_spec['anchor_quaternion']))))<=self.probe_config.maximum_rotation_rad+1e-10))
    def _rollout_batch(self,initial,candidates,modes,controls,old,committed,fixed_sequence):
        c=self.config;pc=self.probe_config;n=len(modes);s,h=self.execution_model.initialize_batch(initial,n)
        arrays={key:np.empty((n,9)+value.shape[1:],dtype=value.dtype) for key,value in s.items()}
        for key in arrays:arrays[key][:,0]=s[key]
        action_arrays={key:np.empty((n,8,size)) for key,size in (('tcp_position',3),('tcp_quaternion',4),
            ('tcp_velocity_reference',3),('angular_velocity_reference',3),('joint_goal',7),('nominal_joint_goal',7),
            ('nominal_tcp_reference',7),('endpoint_tcp_reference',7))}
        for key in ('opening','nominal_reference_time'):action_arrays[key]=np.empty((n,8))
        for key in ('phase_index','nominal_kind'):action_arrays[key]=np.empty((n,8),dtype=int)
        action_arrays['joint_goal_feasible']=np.empty((n,8),bool)
        qualified=np.asarray([candidates[m].get('native_pose_qualified',False) for m in modes],bool)
        nominal=self._nominals['joint_goal'][1,modes];nominal_valid=self._nominals['feasible'][1,modes]
        warm_mask=np.zeros(n,bool)
        if fixed_sequence is not None:warm_mask[next(i for i,row in enumerate(candidates) if row['id']==committed)*(n//8)]=True
        totals={key:np.zeros(n) for key in ('probe_position','probe_orientation','smoothness','effort','constraint')};cost=np.zeros(n)
        for k in range(8):
            qgoal=nominal+controls[k]
            if np.any(warm_mask):qgoal[warm_mask]=np.asarray(fixed_sequence[k]['joint_goal'],float)
            pose=self.goal_helper.kinematics(qgoal);position=pose[:,:3,3];quaternion=matrix_quaternion(pose[:,:3,:3])
            feasible=nominal_valid&np.all((qgoal>=self.goal_helper.adapter.lower)&(qgoal<=self.goal_helper.adapter.upper),-1)
            feasible&=np.all(np.abs(qgoal-nominal)<=self.joint_config.max_perturbation_rad+1e-12,-1)&self._inside(position,quaternion)
            action=dict(tcp_position=position,tcp_quaternion=quaternion,joint_goal=qgoal,joint_goal_feasible=feasible,
                opening=np.zeros(n),tcp_velocity_reference=np.zeros((n,3)),angular_velocity_reference=np.zeros((n,3)),
                nominal_joint_goal=nominal.copy(),nominal_kind=np.ones(n,int),phase_index=np.full(n,2),
                nominal_tcp_reference=self._nominals['pose'][1,modes].copy(),endpoint_tcp_reference=self._nominals['pose'][1,modes].copy(),
                nominal_reference_time=np.full(n,self.probe_spec['trigger_time']))
            for key in action_arrays:action_arrays[key][:,k]=action[key]
            before=s;nxt,hn,info=self.execution_model.step_batch(deepcopy(s),deepcopy(action),deepcopy(h),c.dt)
            model_feasible=validate_step_batch(before,action,nxt,hn,info,c.dt)&feasible
            if np.any(nxt['held']):raise ValueError('verification model fabricated held from a close command')
            s,h=nxt,hn
            for key in arrays:arrays[key][:,k+1]=s[key]
            terms=dict(probe_position=np.sum(((s['tcp_position']-self.probe_spec['target_position'])/pc.position_scale_m)**2,-1),
                probe_orientation=(norm(qlog(qmul(s['tcp_quaternion'],qinv(self.probe_spec['target_quaternion']))))/pc.maximum_rotation_rad)**2,
                smoothness=np.zeros(n) if k==0 else np.sum(((position-action_arrays['tcp_position'][:,k-1])/c.position_scale)**2,-1),
                effort=np.sum(s['tcp_velocity']**2,-1),constraint=np.zeros(n))
            if self.constraint_cost is not None:
                value=np.asarray(self.constraint_cost.batch(before,action,s,qualified,s['time']),float)
                if value.shape!=(n,) or np.isnan(value).any() or np.any(value<0):raise ValueError('invalid probe batch constraints')
                terms['constraint']=value
            ok=model_feasible&qualified&self._inside(s['tcp_position'],s['tcp_quaternion'])
            micro_q=np.asarray(info.get('predicted_actual_q'))
            if micro_q.shape!=(n,2,7):raise ValueError('both actual-joint 50ms predictions are required for probe bounds')
            micro_pose=self.goal_helper.kinematics(micro_q.reshape(n*2,7))
            micro_ok=self._inside(micro_pose[:,:3,3],matrix_quaternion(micro_pose[:,:3,:3])).reshape(n,2).all(-1)
            ok&=micro_ok
            terms['constraint']=np.where(ok,terms['constraint'],np.inf)
            step=terms['probe_position']+.2*terms['probe_orientation']+c.weights['smoothness']*terms['smoothness']+c.weights['effort']*terms['effort']+terms['constraint']
            for key,value in terms.items():totals[key]+=value
            cost+=step
            self.last_model_diagnostics.append(dict(model=info.get('model'),batch=n,microsteps=info['microsteps'],
                feasible_sequences=int(ok.sum()),joint_prediction=info.get('joint_prediction'),attachment_assumed=False))
        phi=pc.terminal_weight*(terms['probe_position']+.2*terms['probe_orientation'])
        cost+=phi;totals['probe_terminal_phi']=phi
        return cost,totals,arrays,action_arrays
    def plan(self,state,candidates,*,warm_start=None):
        if warm_start and warm_start.get('verification_probe_identity')!=self.probe_spec['identity']:
            raise ValueError('incompatible probe warm must be cleared')
        result=super().plan(state,candidates,warm_start=warm_start)
        if result.get('warm_start') is not None:result['warm_start']['verification_probe_identity']=self.probe_spec['identity']
        result['diagnostics'].update(verification_probe=True,probe=deepcopy(self.probe_spec),held_asserted=False,
            robot_response_model_used=True,probe_cost_is_temporary_explicit_robot_pose_cost=True,
            object_follows_probe_assumed=False,selected_action_postprocessed=False,
            continuous_physical_bound_guaranteed=False,physical_safety_certificate=False)
        return result


def qualify_probe_candidates(backend,state,candidates,helper,spec):
    """Same K8 ledger and original static collision oracle; no Native IK."""
    if len(candidates)!=8:raise ValueError('K8 candidate ledger required')
    result=helper.adapter.solve(np.broadcast_to(state['q'],(8,7)).copy(),
        np.broadcast_to(spec['target_position'],(8,3)).copy(),np.broadcast_to(spec['target_quaternion'],(8,4)).copy())
    values=[];receipts=[]
    for i,candidate in enumerate(candidates):
        representative_defined=('parallel_gripper_symmetry_choice' not in candidate
            or candidate['parallel_gripper_symmetry_choice'] is not None)
        ok=bool(result['joint_goal_feasible'][i] and representative_defined);collision=bool(backend.probe_joint_collision(result['joint_goal'][i])) if ok else None
        value=deepcopy(candidate);value['native_pose_qualified']=ok and not collision;values.append(value)
        receipts.append(dict(id=candidate['id'],deadline=candidate['deadline'],deadline_shifted=False,
            candidate_representation_defined=representative_defined,
            qualified=value['native_pose_qualified'],static_collision=collision,target_position=spec['target_position'].copy(),
            target_quaternion=spec['target_quaternion'].copy(),target_source='bounded_public_verification_probe',Native_IK_calls=0,physical_steps=0))
    return values,dict(cases=receipts,batch_DLS_calls=1,poses=8,formal_qualification=False,probe=deepcopy(spec))
