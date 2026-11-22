"""Versioned diagnostic response/action-space combinations, without fitting.

The original pose factory and both NN source packages remain unchanged.  This
factory binds a TRAIN-fitted coupled joint response to the frozen NN's finger
response, or retains the NN joint response, with either pose or joint CEM.
Architecture choices used VAL; these are development diagnostics, not TEST.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import time

from .planner_factory import build_planner as build_original, DiagnosticPlanner, file_sha
from .coupled_response import FrozenCoupledResponse, CoupledWithFingerResponse
from .joint_goal_cem import JointGoalCEMPlanner, JointCEMConfig
from ..local_mpc_20261102.terminal_mpc import TerminalBatchMPCPlanner
from ..local_mpc_p2_20261103.execution_models import JointResidualFKExecutionModel


PARAMETER_SHA256 = 'd7c7fac5f8de7a1bf3d37c10b94a20585728a7c03c7d1a05920799e69a501f92'
DEFAULT_PARAMETERS = 'artifacts/framework/local_mpc_20261102/P2_coupled_response_v1/B_parameters_frozen_20261103.json'


def build_planner(root, limits, *, controller='coupled', action_space='joint',
                  parameter_path=None, expected_parameter_sha256=PARAMETER_SHA256,
                  joint_cem_config=None, **kwargs):
    """Return the existing dispatch contract with explicitly versioned choices.

    The selected joint goal is passed unchanged to the original collision gates
    and executor. Changing action space changes the proposals; it is not an
    assertion of trajectory equivalence or improved closed-loop success.
    """
    started = time.perf_counter()
    if controller not in ('learned', 'coupled'):
        raise ValueError('explicit learned or coupled joint response required')
    if action_space not in ('pose', 'joint'):
        raise ValueError('explicit pose or joint action space required')
    original, helper, provenance = build_original(root, limits, controller='learned', **kwargs)
    old = original.planner
    execution = old.execution_model
    provenance = deepcopy(provenance)
    if controller == 'coupled':
        path = Path(parameter_path) if parameter_path is not None else Path(root) / DEFAULT_PARAMETERS
        if not path.is_absolute():
            path = Path(root) / path
        path = path.resolve(strict=True)
        joint = FrozenCoupledResponse.from_json(path, expected_sha256=expected_parameter_sha256)
        transition = CoupledWithFingerResponse(joint, execution.transition_batch)
        execution = JointResidualFKExecutionModel(execution.reference_step, execution.limits,
            transition, execution.fk_batch, config=old.config,
            model_id='P2_coupled_B_plus_frozen_v2_fingers_' + expected_parameter_sha256)
        provenance['coupled_parameters'] = dict(path=str(path), sha256=file_sha(path),
            fit_split='train', normalization_fit_split='train', features=43,
            position_validation_only=True, task_success_qualified=False,
            architecture_selection_used_VAL=True, formal_TEST_used=False)
        provenance['joint_prediction'] = 'TRAIN_fitted_coupled_joint_delta_plus_position_difference_dq_plus_validated_static_PoE'
        provenance['checkpoint']['neural_response_scope'] = 'two_finger_response_only_NN_joint_predictions_discarded'
    else:
        provenance['checkpoint']['neural_response_scope'] = 'joint_and_two_finger_response'
    if action_space == 'joint':
        raw = JointGoalCEMPlanner(old.config, constraint_cost=old.constraint_cost,
            execution_model=execution, goal_helper=helper,
            joint_config=joint_cem_config, terminal_config=old.terminal_config)
        # DiagnosticPlanner exposes this helper for the common action contract;
        # JointGoalCEMPlanner never calls it during virtual execution rollouts.
        raw.action_adapter = helper.adapter
        provenance['joint_cem_config'] = asdict(raw.joint_config)
        provenance['virtual_step_IK_calls'] = 0
    else:
        if joint_cem_config is not None:
            raise ValueError('joint CEM settings cannot be applied to pose space')
        raw = TerminalBatchMPCPlanner(old.config, constraint_cost=old.constraint_cost,
            execution_model=execution, action_adapter=helper.adapter,
            terminal_config=old.terminal_config)
    provenance.update(schema='local_mpc_response_action_factory_20261103_v1',
        controller=controller, action_space=action_space,
        trajectory_equivalence_to_original_pose_CEM=False,
        architecture_selection_used_VAL=True, formal_TEST_used=False,
        gripper_or_contact_qualified=False, precision_qualified=False,
        real_time_guaranteed=False)
    sources = [Path(__file__), Path(__file__).with_name('coupled_response.py'),
               Path(__file__).with_name('joint_goal_cem.py')]
    provenance['source_sha256'].update({str(p.resolve()):file_sha(p) for p in sources})
    provenance['build_wall_seconds'] = time.perf_counter() - started
    return DiagnosticPlanner(raw, provenance), helper, provenance
