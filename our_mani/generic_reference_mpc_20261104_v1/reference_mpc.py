"""Robot-only reference-tracking CEM; no fabricated object/held state.

The old arm ridge is reused directly. The old finger NN is evaluated under
both legacy held-feature hypotheses; neither hypothesis is a measured contact.
This is a new, unqualified tracking baseline, not P4 local-object handoff.
"""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

from ..local_mpc_p2_20261103.reference import ReferenceLimits, reference_step
from ..local_mpc_p2_20261103.action_adapter import BatchPoseJointAdapter, DLSConfig, matrix_quaternion
from ..local_mpc_p2_20261103.kinematics import PoEKinematics
from ..local_mpc_control_20261103.coupled_response import FrozenCoupledResponse
from .contracts import WorldTCPReference, vector, retained_gripper_command


@dataclass(frozen=True)
class TrackingConfig:
    horizon: int = 8
    samples: tuple = (88, 88, 80)
    elite_count: int = 8
    joint_offset_bound: float = .30
    initial_std: float = .5
    minimum_std: float = .05
    position_scale_m: float = .03
    rotation_scale_rad: float = .2
    smooth_weight: float = .01
    finger_tracking_weight: float = .1
    seed: int = 261104920

    def __post_init__(self):
        if self.horizon != 8 or self.samples != (88,88,80) or self.elite_count != 8:
            raise ValueError('Explicit fixed256/H8/88-88-80 tracking budget required')
        if min(self.position_scale_m,self.rotation_scale_rad,self.joint_offset_bound,self.initial_std,self.minimum_std) <= 0:
            raise ValueError('Positive declared cost/proposal scales required')


def actual_reference_batch(state, n):
    h = state['actuator_state']; q, dq = state['q'], state['dq']
    valid = bool(h.get('reference_valid', False))
    if valid and (h.get('reference_clock') is None or abs(h['reference_clock'] - state['time']) > 1e-9):
        raise ValueError('Only actual current executor state may initialize a plan')
    def batch(name, default):
        value = h.get(name)
        return np.broadcast_to(vector(default if value is None else value, 7, name), (n,7)).copy()
    return dict(q_ref=batch('reference_positions',q), dq_ref=batch('reference_velocities',dq),
        ddq_ref=batch('reference_accelerations',np.zeros(7)), goal_q=batch('target_joints',q),
        valid=np.full(n,valid), reference_valid=np.full(n,valid),
        initialized=np.full(n,h.get('reference_accelerations') is not None))


class GenericReferenceMPC:
    def __init__(self, joint_response, finger_response, kinematics, limits, config=TrackingConfig()):
        from ..local_mpc_control_20261103.finger_response_fastpath_20261103 import FrozenFingerOnlyResponse
        if (type(joint_response) is not FrozenCoupledResponse or type(finger_response) is not FrozenFingerOnlyResponse
                or type(kinematics) is not PoEKinematics or type(config) is not TrackingConfig):
            raise TypeError('Exact frozen arm/finger/FK types and explicit new tracking config required')
        self.joint_response, self.finger_response, self.kinematics = joint_response, finger_response, kinematics
        self.limits = ReferenceLimits.from_executor_limits(limits); self.config = config
        self.adapter = BatchPoseJointAdapter(kinematics, self.limits, DLSConfig())
        self.rng = np.random.default_rng(config.seed)
        self.evidence = dict(schema='generic_robot_only_reference_CEM_v1', model_qualified=False,
            old_learned_contact_dynamics=False, object_state_supplied=False,
            finger_feature_modes=[0,1], finger_modes_are_observations=False,
            finger_mode_switch_or_continuous_uncertainty_certified=False,
            current_network_task_success_qualified=False, physical_handoff_qualified=False)

    def _rollout(self, state, goals, reference):
        c = self.config; n = len(goals); q = np.broadcast_to(state['q'],(n,7)).copy()
        dq = np.broadcast_to(state['dq'],(n,7)).copy()
        fingers = [np.broadcast_to(state['finger_open_amounts'],(n,2)).copy() for _ in range(2)]
        h = actual_reference_batch(state,n); feasible = np.ones(n,bool); costs = np.zeros(n)
        target = retained_gripper_command(state); target_R = Rotation.from_quat(reference.quaternion).as_matrix()
        previous_goal = np.broadcast_to(state['q'],(n,7)).copy(); qstates=[]; finger_calls=0
        for step in range(c.horizon):
            goal = goals[:,step]
            costs += c.smooth_weight * np.sum((goal - previous_goal)**2,axis=-1)
            for _ in range(2):
                applied = reference_step(q,dq,goal,h,self.limits,dt=.05,frequency=6.)
                feasible &= applied['feasible']; h = applied['next_h']
                refs = dict(q_ref=applied['q_ref'],dq_ref=applied['dq_ref'],ddq_ref=applied['ddq_ref'],
                    gripper_target=np.full(n,target))
                joint = self.joint_response.predict_joint(q,dq,refs['q_ref'],refs['dq_ref'],refs['ddq_ref'],dt=.05)
                for mode in (0,1):
                    # Explicit counterfactual conditioning, NOT a relabelled
                    # feedback 'held=False'. No contact/attachment is inferred.
                    predicted = self.finger_response.predict_execution(
                        dict(q=q,dq=dq,finger_open_amounts=fingers[mode]), refs, None, .05,
                        held_anchor=np.full(n,bool(mode),dtype=bool))
                    fingers[mode] = predicted['finger_open_amounts']; feasible &= predicted['feasible']; finger_calls += 1
                q,dq = joint['q'],joint['dq']; qstates.append(q.copy())
                feasible &= np.isfinite(q).all(-1) & np.isfinite(dq).all(-1)
                feasible &= (q >= self.limits.lower-1e-8).all(-1) & (q <= self.limits.upper+1e-8).all(-1)
            transforms = self.kinematics(q)
            position = transforms[:,:3,3]
            angles = np.linalg.norm(Rotation.from_matrix(target_R @ transforms[:,:3,:3].transpose(0,2,1)).as_rotvec(),axis=-1)
            costs += np.sum((position-reference.position)**2,axis=-1)/c.position_scale_m**2 + angles**2/c.rotation_scale_rad**2
            errors = np.stack([np.sum((f-target)**2,axis=-1) for f in fingers])
            costs += c.finger_tracking_weight * errors.max(0)
            previous_goal = goal
        costs[~feasible | ~np.isfinite(costs)] = np.inf
        return costs, dict(predicted_actual_q=np.stack(qstates,axis=1),
            conditional_fingers=np.stack(fingers,axis=1),finger_model_batch_calls=finger_calls,
            counterfactual_finger_modes_not_contact_measurement=True)

    def plan(self, state, reference):
        if type(reference) is not WorldTCPReference or abs(reference.timestamp-state['time']) > 1e-9:
            raise ValueError('Current caller-selected world-command reference required')
        q, dq = vector(state['q'],7,'actual q'), vector(state['dq'],7,'actual dq')
        fingers = vector(state['finger_open_amounts'],2,'actual finger openings')
        if np.any((fingers<0)|(fingers>1)) or not isinstance(state.get('actuator_state'),dict):
            raise ValueError('Actual robot/actuator feedback required, no object-state surrogate')
        gripper_target = retained_gripper_command(state)
        pose = self.kinematics(q); actual_pose = vector(state['tcp_pose'],7,'actual TCP pose')
        rotation_error = np.linalg.norm(Rotation.from_matrix(Rotation.from_quat(actual_pose[3:]).as_matrix() @ pose[:3,:3].T).as_rotvec())
        # The old FK certificate covers its static snapshot; dynamic measured
        # TCP residual is diagnostic, not a new uncalibrated10um runtime gate.
        current_fk_error = float(np.linalg.norm(pose[:3,3]-actual_pose[:3]))
        solved = self.adapter.solve(q[None],reference.position[None],reference.quaternion[None])
        if not bool(solved['joint_goal_feasible'][0]):
            return dict(action=None,reason='world_reference_algebraic_IK_infeasible',evaluations=0,
                reference_timestamp=reference.timestamp,**self.evidence)
        nominal = solved['joint_goal'][0]; c=self.config
        mean=np.zeros((c.horizon,7));std=np.full_like(mean,c.initial_std);best=None;global_order=0;finger_calls=0
        for iteration,count in enumerate(c.samples):
            noise=self.rng.normal(size=(count,c.horizon,7));controls=np.clip(mean+noise*std,-1.,1.)
            controls[0]=mean
            goals=np.clip(nominal+controls*c.joint_offset_bound,self.limits.lower,self.limits.upper)
            costs,info=self._rollout(state,goals,reference);finger_calls+=info['finger_model_batch_calls']
            order=np.argsort(costs,kind='stable');finite=order[np.isfinite(costs[order])]
            for row in finite:
                key=(float(costs[row]),global_order+int(row))
                if best is None or key<best[0]:
                    best=(key,goals[row].copy(),info['predicted_actual_q'][row].copy())
            if len(finite):
                elites=controls[finite[:c.elite_count]];mean=elites.mean(0);std=np.maximum(elites.std(0),c.minimum_std)
            global_order+=count
        result=dict(evaluations=global_order,finger_model_batch_calls=finger_calls,
            current_measured_TCP_minus_staticFK_m=current_fk_error,
            current_measured_TCP_vs_staticFK_rotation_rad=float(rotation_error),
            staticFK_is_not_full_dynamic_contact_model=True,
            reference_timestamp=reference.timestamp,reference_selected_index=reference.selected_index,
            reference_selection_rule=reference.selection_rule,task_utility_in_tracking_cost=False,
            opening_intent=reference.opening_intent,opening_intent_condition=reference.opening_intent_condition,
            opening_intent_issued=False,gripper_sequencing_status='BLOCKED_unresolved_arm_to_gripper_condition',
            cost_scope='TCPpose_tracking_plus_joint_smoothness_and_two_conditional_finger_errors',
            tracking_scale_is_not_accuracy_guarantee=True,**self.evidence)
        if best is None: return dict(result,action=None,reason='all_tracking_rollouts_infeasible')
        goal=best[1][0];T=self.kinematics(goal)
        return dict(result,action=dict(joint_goal=goal,tcp_position=T[:3,3],tcp_quaternion=matrix_quaternion(T[:3,:3]),
            opening=gripper_target),cost=best[0][0],
            selected_joint_sequence=best[1],predicted_actual_q=best[2],
            reason='model_selected_only_requires_original8Native_sampled_checks')


def qualify_first_action(backend, state, action):
    """Exact selected q, native bounds and endpoint+seven original path samples."""
    goal=vector(action['joint_goal'],7,'selected joint goal');actual=vector(state['q'],7,'actual q')
    if abs(backend.clock()-state['time'])>1e-9 or not np.allclose(backend.get_joint_positions(),actual,atol=1e-6,rtol=0):
        raise ValueError('Action qualification cannot borrow a previous physical state')
    limits=backend.joint_limits();receipts=[]
    if np.any(goal<limits['lower'])or np.any(goal>limits['upper']):
        return False,dict(reason='outside_native_joint_limits',probes=[],physical_steps=0)
    for alpha in (1.,.125,.25,.375,.5,.625,.75,.875):
        before=backend.clock();collision=bool(backend.probe_joint_collision((1-alpha)*actual+alpha*goal))
        if backend.clock()!=before:raise RuntimeError('Native sampled collision query advanced physics')
        receipts.append(dict(alpha=alpha,collision=collision))
        if collision:return False,dict(reason='sampled_arm_scene_collision',probes=receipts,physical_steps=0)
    return True,dict(reason='all_original8_configuration_queries_passed',probes=receipts,physical_steps=0,
        whole_swept_path_or_task_success_certified=False)


def load_frozen_robot_response(*,joint_parameters,joint_sha256,finger_checkpoint,finger_sha256,limits,config=TrackingConfig()):
    """Actual artifact loader, no forward/Native; never invoked by defaultcheck.

    Uses existing completed P2 weights/FK validation without promoting their
    position-only evidence to a new16-task/contact qualification.
    """
    import torch
    from ..local_mpc_p2_20261103_v2.servo_adapter import TorchServoTransition
    from ..local_mpc_control_20261103.finger_response_fastpath_20261103 import FrozenFingerOnlyResponse
    from ..local_mpc_control_20261103.planner_factory import _validated_fk
    torch.set_num_threads(2)
    joint=FrozenCoupledResponse.from_json(joint_parameters,expected_sha256=joint_sha256)
    transition=TorchServoTransition.from_checkpoint(finger_checkpoint,expected_sha256=finger_sha256)
    snapshot,evidence=_validated_fk(transition.evidence)
    planner=GenericReferenceMPC(joint,FrozenFingerOnlyResponse(transition),PoEKinematics(snapshot),limits,config)
    planner.evidence.update(joint_parameters_sha256=joint_sha256,finger_checkpoint_sha256=finger_sha256,
        validated_static_FK=evidence,old_model_generalized_to16tasks_qualified=False,
        frozen_model_loader_ready=True,NN_forward_calls_by_loader=0)
    return planner
