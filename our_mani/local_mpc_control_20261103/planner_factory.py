"""Diagnostic controller construction; strict CPU weights, no Native/fit.

Neither this factory nor a finite CPU plan qualifies the learned model. The
global all-height v2 model remains unqualified, and a post-hoc Native-height
subset cannot promote it to formal qualification. The caller owns collision
gates, actual dispatch and a fresh measured NodeState every 100ms decision.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from ..local_mpc_20261102.mpc import MPCConfig
from ..local_mpc_20261102.terminal_mpc import TerminalBatchMPCPlanner,TerminalCostConfig
from ..local_mpc_p2_20261103.execution_models import (
    JointResidualFKExecutionModel,SimpleBatchExecutionModel,validate_feedback_state)
from ..local_mpc_p2_20261103.reference import ReferenceLimits,reference_step
from ..local_mpc_p2_20261103.kinematics import PoEKinematics
from ..local_mpc_p2_20261103.action_adapter import matrix_quaternion
from .joint_goals import JointGoalConfig,PoEJointGoalHelper


CHECKPOINT_SHA256='25a0e0cbd49e0e0c9e8a4671649f96a1bb565013011d728e88b2c2f3eaca0073'
DEFAULT_CHECKPOINT='artifacts/framework/local_mpc_20261102/P2_servo_training_v2/best.pt'
SCHEMA='local_mpc_control_diagnostic_factory_20261103_v1'


def file_sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def validate_node_state(state):
    """Preserve the full current record while checking the actual model inputs.

    q/dq are actual measurements; actuator_state is a fresh executor snapshot.
    Two actual normalized finger openings are mandatory. Extra measured finger
    positions/velocities, servo readback and gripper history stay in the record;
    the trained model's 17 inputs remain q7/dq7/opening2/held1, without secretly
    adding new dimensions or deriving two fingers from an aperture mean.
    """
    if not isinstance(state,dict):raise ValueError('fresh NodeState dictionary required')
    if not isinstance(state.get('actuator_state'),dict):raise ValueError('actual executor actuator_state snapshot required')
    checked=validate_feedback_state(state)
    for key,size in (('q',7),('dq',7),('finger_open_amounts',2)):
        value=np.asarray(checked.get(key),float)
        if value.shape!=(size,) or not np.isfinite(value).all():raise ValueError('actual '+key+f'[{size}] required')
    if abs(float(checked['opening'])-float(np.mean(checked['finger_open_amounts'])))>1e-5:
        raise ValueError('opening mean differs from actual two finger openings')
    requested=float(state.get('requested_opening',np.nan))
    if not np.isfinite(requested) or not 0<=requested<=1:raise ValueError('actual requested_opening required')
    for key,size in (('finger_positions',2),('finger_velocities',2),('servo_target_positions',7)):
        if key in state:
            value=np.asarray(state[key],float)
            if value.shape!=(size,) or not np.isfinite(value).all():raise ValueError('finite current '+key+' required')
    result=deepcopy(state)
    result.update(checked)
    # Exercise canonical reference initialization separately when valid. It
    # rejects partial/fictional references and preserves invalid prior a.
    from ..local_mpc_p2_20261103.execution_models import initialize_execution_batch
    initialize_execution_batch(result,1,requires_joint_state=True)
    return result


class DiagnosticPlanner:
    """Validates measured feedback before each wrapped, unchanged CEM plan."""
    def __init__(self,planner,provenance):
        self.planner=planner
        self.provenance=deepcopy(provenance)
        self.config=planner.config
        self.execution_model=planner.execution_model
        self.action_adapter=planner.action_adapter
        self.terminal_config=planner.terminal_config
    def plan(self,state,candidates,*,warm_start=None):
        state=validate_node_state(state)
        result=self.planner.plan(state,candidates,warm_start=warm_start)
        result.setdefault('diagnostics',{}).update(diagnostic_only=True,formal_qualification=False,
            global_all_height_qualified=False,posthoc_subset_not_formal_qualification=True,
            controller=self.provenance['controller'],servo_checkpoint_sha256=self.provenance['checkpoint']['sha256'],
            Native_canary_required=True,precision_qualified=False)
        if result.get('action') is not None and 'joint_goal' not in result['action']:
            raise RuntimeError('selected action lost its modeled/declared joint goal; no second IK is permitted')
        return result


def _validated_fk(evidence):
    identity=evidence['identity']['fk']
    paths={key:Path(identity[key+'_path']).resolve(strict=True) for key in ('snapshot','validation')}
    for key,path in paths.items():
        if file_sha(path)!=identity[key+'_sha256']:raise ValueError('FK '+key+' evidence SHA mismatch')
    snapshot=json.loads(paths['snapshot'].read_text());validation=json.loads(paths['validation'].read_text())
    if (snapshot.get('Native_FK_validation_passed') is not True or validation.get('complete') is not True
            or validation.get('passed') is not True or validation.get('Native_controlled_physics_ticks')!=0
            or validation.get('restoration_checked') is not True
            or not np.isfinite(validation.get('maximum_TCP_position_error_m',np.inf))
            or not np.isfinite(validation.get('maximum_rotation_error_rad',np.inf))
            or validation.get('maximum_TCP_position_error_m',np.inf)>1e-5
            or validation.get('maximum_rotation_error_rad',np.inf)>1e-4):
        raise ValueError('restoring zero-step Native-validated PoE snapshot required')
    return snapshot,{key:dict(path=str(path),sha256=file_sha(path)) for key,path in paths.items()}


def build_planner(root,limits,*,controller='simple',checkpoint_path=None,
                  expected_checkpoint_sha256=CHECKPOINT_SHA256,mpc_config=None,
                  joint_goal_config=None,terminal_config=None,cpu_threads=2,reference_frequency=6.):
    """Return (diagnostic_planner, goal_helper, provenance), CPU only.

    Both modes bind the same completed checkpoint/FK evidence and use the same
    geometric mapping and terminal cost. Simple mode does not use the neural
    response, and explicitly does not predict q/dq. Learned mode uses the v2
    CPU servo response through the unchanged generic two-tick reference chain.
    Exactly the selected action joint_goal must pass helper.qualify_action and
    be given to execute_joint_target; a post-plan IK recomputation is forbidden.
    """
    started=time.perf_counter()
    if controller not in ('simple','learned'):raise ValueError('explicit simple or learned controller required')
    if type(cpu_threads) is not int or cpu_threads<=0:raise ValueError('positive CPU inference thread count required')
    if not np.isfinite(reference_frequency) or reference_frequency<=0:raise ValueError('finite positive matching executor reference frequency required')
    root=Path(root).resolve(strict=True)
    path=Path(checkpoint_path) if checkpoint_path is not None else root/DEFAULT_CHECKPOINT
    if not path.is_absolute():path=root/path
    path=path.resolve(strict=True)
    before=file_sha(path)
    if before!=expected_checkpoint_sha256:raise ValueError('factory checkpoint SHA mismatch')
    import torch
    from ..local_mpc_p2_20261103_v2.servo_adapter import TorchServoTransition
    torch.set_num_threads(cpu_threads)
    transition=TorchServoTransition.from_checkpoint(path,expected_sha256=expected_checkpoint_sha256)
    snapshot,fk_evidence=_validated_fk(transition.evidence)
    chain=PoEKinematics(snapshot)
    if chain.dof!=7:raise ValueError('P2 checkpoint and Native chain require seven actual arm joints')
    c=mpc_config if isinstance(mpc_config,MPCConfig) else MPCConfig(**(mpc_config or {}))
    jc=joint_goal_config if isinstance(joint_goal_config,JointGoalConfig) else JointGoalConfig(**(joint_goal_config or {}))
    tc=terminal_config if isinstance(terminal_config,TerminalCostConfig) else TerminalCostConfig(**(terminal_config or {}))
    helper=PoEJointGoalHelper(chain,limits,jc)
    reference_limits=ReferenceLimits.from_executor_limits(limits)
    def fk_batch(q):
        pose=chain(q)
        return dict(tcp_position=pose[...,:3,3],tcp_quaternion=matrix_quaternion(pose[...,:3,:3]))
    def matching_reference(q,dq,goal,h,bounds,*,dt):
        return reference_step(q,dq,goal,h,bounds,dt=dt,frequency=reference_frequency)
    if controller=='learned':
        execution=JointResidualFKExecutionModel(matching_reference,reference_limits,transition,fk_batch,
            config=c,model_id='P2_v2_diagnostic_'+before)
    else:
        execution=SimpleBatchExecutionModel(c)
    from ..local_mpc_20261102.constraints import DiagnosticConstraints
    constraints=DiagnosticConstraints(lower=jc.lower,upper=jc.upper)
    raw=TerminalBatchMPCPlanner(c,constraint_cost=constraints,execution_model=execution,
                               action_adapter=helper.adapter,terminal_config=tc)
    files=[Path(__file__),Path(__file__).with_name('joint_goals.py'),
           Path(__file__).parent.parent/'local_mpc_p2_20261103_v2/servo_adapter.py',
           Path(__file__).parent.parent/'local_mpc_p2_20261103/execution_models.py',
           Path(__file__).parent.parent/'local_mpc_p2_20261103/reference.py',
           Path(__file__).parent.parent/'local_mpc_20261102/terminal_mpc.py']
    provenance=dict(schema=SCHEMA,controller=controller,diagnostic_only=True,precision_qualified=False,
        formal_qualification=False,global_all_height_qualified=False,posthoc_native_height_subset_is_formal_qualification=False,
        Native_canary_required=True,Native_task_success_measured=False,Native_started=False,training_started=False,formal_TEST_used=False,
        device='cpu',cpu_threads=cpu_threads,reference_frequency=reference_frequency,
        checkpoint=dict(path=str(path),sha256=before,neural_response_used=controller=='learned',
                        selected_epoch=transition.evidence.get('selected_epoch'),training_complete=transition.evidence['training_complete'],
                        full_known_reference_chain_qualified=transition.evidence.get('full_known_reference_chain_qualified',False)),
        FK=fk_evidence,mpc_config=asdict(c),joint_goal_config=asdict(jc),terminal_config=asdict(tc),
        joint_prediction=('learned_reference_tracking_residual_plus_validated_static_PoE_not_complete_contact' if controller=='learned'
                          else 'Cartesian_simple_response_q_dq_passed_through_not_predicted'),
        state_input_order='q7_dq7_actual_finger_open_amounts2_held_anchor1',
        held_input='anchor_actual_mode_fixed_no_future_contact_label',
        dispatch_contract='exact_selected_action_joint_goal_after_original_collision_gates_no_second_IK',
        collision_qualified=False,real_time_guaranteed=False,
        source_sha256={str(p.resolve()):file_sha(p) for p in files},build_wall_seconds=time.perf_counter()-started)
    if file_sha(path)!=before:raise ValueError('checkpoint changed during factory construction')
    return DiagnosticPlanner(raw,provenance),helper,provenance
