"""PoE DLS target qualification without Native IK or physical stepping.

Algebraic solves are pure NumPy. Optional Native collision callbacks retain
the original no-step point/path sampling checks; they are not a swept-path or
physical reachability certificate. This helper contains no model weights.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import time
import numpy as np

from ..local_mpc_20261102 import mpc
from ..local_mpc_20261102.batch_mpc import norm,qmul,qinv,qlog,unit
from ..local_mpc_p2_20261103.action_adapter import BatchPoseJointAdapter,DLSConfig,matrix_quaternion
from ..local_mpc_p2_20261103.kinematics import PoEKinematics


@dataclass(frozen=True)
class JointGoalConfig:
    position_tolerance_m: float = .002
    angle_tolerance_rad: float = .02
    iterations: int = 24
    damping: float = .01
    rotation_weight: float = .25
    max_joint_step_rad: float = .25
    lower: tuple = (-.1,-.65,.76)
    upper: tuple = (.65,.65,1.65)
    path_samples: tuple = (.125,.25,.375,.5,.625,.75,.875)

    def __post_init__(self):
        DLSConfig(iterations=self.iterations,damping=self.damping,
                  position_tolerance=self.position_tolerance_m,angle_tolerance=self.angle_tolerance_rad,
                  rotation_weight=self.rotation_weight,max_joint_step=self.max_joint_step_rad)
        lower,upper=np.asarray(self.lower,float),np.asarray(self.upper,float)
        if lower.shape!=(3,) or upper.shape!=(3,) or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower>=upper):
            raise ValueError('ordered finite workspace bounds required')
        if tuple(self.path_samples)!=(.125,.25,.375,.5,.625,.75,.875):
            raise ValueError('preserve the original seven intermediate collision samples')


def _joints(value,dof,name):
    value=np.asarray(value,dtype=np.float64)
    if value.shape!=(dof,) or not np.isfinite(value).all():raise ValueError('finite '+name+'[J] required')
    return value.copy()


class PoEJointGoalHelper:
    """Batch K8 solve and exact selected-goal reuse, with explicit receipts.

    Pass a separately validated world PoE chain and actual Native joint limits.
    The same `adapter` object can be injected into a joint-response planner.
    If that planner supplies an action joint_goal, qualification reuses its exact
    values and only verifies FK/bounds; it does NOT resolve onto another branch.
    A Cartesian-only baseline may explicitly allow one post-plan DLS solve,
    recorded as no modeled joint goal rather than a false consistency claim.
    """
    def __init__(self,kinematics,limits,config=None):
        if not isinstance(kinematics,PoEKinematics):raise TypeError('explicit world PoEKinematics required')
        self.kinematics,self.limits=kinematics,limits
        self.config=config if isinstance(config,JointGoalConfig) else JointGoalConfig(**(config or {}))
        c=self.config
        self.adapter=BatchPoseJointAdapter(kinematics,limits,DLSConfig(iterations=c.iterations,damping=c.damping,
            position_tolerance=c.position_tolerance_m,angle_tolerance=c.angle_tolerance_rad,
            rotation_weight=c.rotation_weight,max_joint_step=c.max_joint_step_rad))
        self.metadata=dict(method='pure_numpy_world_PoE_batch_DLS',Native_IK_called=False,physical_steps=0,
            thresholds=asdict(c),prediction_dispatch_rule='reuse_selected_joint_goal_exactly_no_second_IK_branch',
            tolerance_scope='ideal_kinematic_pose_residual_not_dynamic_Native_TCP_accuracy',
            independently_validated_kinematic_snapshot_required=True,learned_weights=False)

    def solve_candidate_targets(self,state,candidates):
        """Exactly eight target poses, once as one NumPy batch.

        Target time and pregrasp selection match the old candidate qualifier.
        Workspace checks remain in rollout/action gates: this function does not
        silently introduce an extra candidate workspace exclusion.
        """
        started=time.perf_counter();actual=mpc.validate_state(state)
        q=_joints(state.get('q'),self.kinematics.dof,'actual q')
        candidates=[mpc.validate_candidate(c) for c in candidates]
        if len(candidates)!=8 or len({c['id'] for c in candidates})!=8:raise ValueError('exactly eight distinct candidate poses required')
        phase=(state.get('phase_state') or {}).get('phase','approach')
        targets=[mpc.contact_target(actual,c,max(actual['time'],c['deadline']),pregrasp=phase=='approach') for c in candidates]
        solved=self.adapter.solve(np.broadcast_to(q,(8,len(q))).copy(),
            np.stack([t['position'] for t in targets]),np.stack([t['quaternion'] for t in targets]))
        receipts=[]
        for i,(candidate,target) in enumerate(zip(candidates,targets)):
            receipts.append(dict(id=candidate['id'],time=actual['time'],target_time=target['time'],
                target_position=target['position'].copy(),target_quaternion=target['quaternion'].copy(),
                joint_goal=solved['joint_goal'][i].copy(),algebraic_IK_available=bool(solved['joint_goal_feasible'][i]),
                position_error_m=float(solved['ik_position_error'][i]),angle_error_rad=float(solved['ik_angle_error'][i]),
                iterations=int(solved['ik_iterations'][i]),seed_valid=bool(solved['ik_seed_valid'][i]),
                Native_IK_called=False,physical_steps=0,physical_reachability_certified=False))
        return dict(candidates=candidates,targets=targets,joint_goals=solved['joint_goal'],
                    feasible=solved['joint_goal_feasible'],receipts=receipts,
                    batch_solves=1,poses=8,wall_seconds=time.perf_counter()-started)

    def qualify_candidates(self,backend,state,candidates):
        """Same K8 ledger; collision only via the original Native probe service."""
        result=self.solve_candidate_targets(state,candidates);qualified=[];receipts=[]
        for i,candidate in enumerate(result['candidates']):
            receipt=deepcopy(result['receipts'][i]);ok=bool(result['feasible'][i]);collision=None
            t=time.perf_counter()
            if ok:collision=bool(backend.probe_joint_collision(result['joint_goals'][i]))
            receipt.update(static_arm_collision=collision,collision_wall_seconds=time.perf_counter()-t,
                           qualified=ok and not collision,IK_available=ok,whole_path_certified=False,
                           scope='PoE_algebraic_IK_plus_original_Native_static_collision_not_Native_IK')
            value=deepcopy(candidate);value['native_pose_qualified']=receipt['qualified']
            value['qualification_joint_goal']=result['joint_goals'][i].copy()
            value['joint_goal_qualification_method']='pure_PoE_DLS_plus_Native_static_probe'
            qualified.append(value);receipts.append(receipt)
        return qualified,dict(cases=receipts,batch_solves=1,poses=8,DLS_wall_seconds=result['wall_seconds'],
                              collision_wall_seconds=sum(r['collision_wall_seconds'] for r in receipts),Native_IK_calls=0,physical_steps=0)

    def solve_action_target(self,state,action,*,require_predicted_joint_goal=True):
        """Exact reuse if a modeled joint goal exists, otherwise explicit DLS.

        Missing modeled goal fails closed by default. A caller opting into a
        Cartesian-only baseline can set require_predicted_joint_goal=False and
        obtains one DLS solve, with model_joint_goal_match=False in the receipt.
        """
        started=time.perf_counter();q=_joints(state.get('q'),self.kinematics.dof,'actual q')
        position=np.asarray(action.get('tcp_position'),float);quaternion=np.asarray(action.get('tcp_quaternion'),float)
        if position.shape!=(3,) or quaternion.shape!=(4,) or not np.isfinite(position).all() or not np.isfinite(quaternion).all() or norm(quaternion)<1e-12:
            raise ValueError('finite selected TCP target pose required')
        quaternion=unit(quaternion)
        predicted=action.get('joint_goal')
        receipt=dict(Native_IK_called=False,physical_steps=0,feedback_time=float(state['time']),
                     position_tolerance_m=self.config.position_tolerance_m,angle_tolerance_rad=self.config.angle_tolerance_rad)
        if predicted is None:
            if require_predicted_joint_goal:
                return None,'missing_selected_model_joint_goal',dict(receipt,model_joint_goal_match=False,DLS_solves=0,wall_seconds=time.perf_counter()-started)
            solved=self.adapter.solve(q[None],position[None],quaternion[None])
            goal=solved['joint_goal'][0].copy()
            receipt.update(source='single_action_DLS_Cartesian_baseline_joint_response_not_modeled',
                           model_joint_goal_match=False,DLS_solves=1,iterations=int(solved['ik_iterations'][0]))
        else:
            goal=_joints(predicted,self.kinematics.dof,'selected modeled joint goal')
            receipt.update(source='selected_planner_joint_goal_exact_reuse',model_joint_goal_match=np.array_equal(goal,np.asarray(predicted)),
                           DLS_solves=0,iterations=0,match_scope='control_target_identity_not_response_accuracy')
        transform=self.kinematics(goal)
        distance=float(norm(position-transform[:3,3]))
        angle=float(norm(qlog(qmul(quaternion,qinv(matrix_quaternion(transform[:3,:3]))))))
        finite_bounds=bool(np.all(goal>=self.adapter.lower-1e-8) and np.all(goal<=self.adapter.upper+1e-8))
        valid_seed=bool(np.all(q>=self.adapter.lower-1e-8) and np.all(q<=self.adapter.upper+1e-8))
        feasible=finite_bounds and valid_seed and distance<=self.config.position_tolerance_m and angle<=self.config.angle_tolerance_rad
        if 'joint_goal_feasible' in action and not bool(action['joint_goal_feasible']):feasible=False
        receipt.update(joint_goal=goal.copy(),actual_q=q.copy(),position_error_m=distance,angle_error_rad=angle,
                       bounds_valid=finite_bounds,seed_valid=valid_seed,feasible=feasible,wall_seconds=time.perf_counter()-started)
        return (goal.copy(),None,receipt) if feasible else (None,'PoE_action_pose_or_joint_bounds_infeasible',receipt)

    def qualify_action(self,backend,state,action,*,require_predicted_joint_goal=True):
        """Pose bounds, exact selected q, endpoint + seven path collision probes."""
        p=np.asarray(action.get('tcp_position'),float)
        if p.shape!=(3,) or not np.isfinite(p).all() or np.any(p<self.config.lower) or np.any(p>self.config.upper):
            return None,'action_outside_diagnostic_workspace',dict(physical_steps=0,Native_IK_called=False,collision_probes=[])
        goal,reason,receipt=self.solve_action_target(state,action,require_predicted_joint_goal=require_predicted_joint_goal)
        probes=[];receipt['collision_probes']=probes
        if reason:return None,reason,receipt
        actual=_joints(state.get('q'),self.kinematics.dof,'actual q')
        for alpha in (1.,)+self.config.path_samples:
            target=goal.copy() if alpha==1. else (1-alpha)*actual+alpha*goal
            started=time.perf_counter();collision=bool(backend.probe_joint_collision(target))
            probes.append(dict(alpha=alpha,collision=collision,wall_seconds=time.perf_counter()-started,physical_steps=0))
            if collision:
                receipt['qualified']=False
                return None,('action_arm_static_collision' if alpha==1. else 'sampled_joint_path_arm_static_collision'),receipt
        receipt.update(qualified=True,whole_path_certified=False,collision_wall_seconds=sum(p['wall_seconds'] for p in probes),
                       original_intermediate_samples_preserved=True)
        return goal.copy(),None,receipt
