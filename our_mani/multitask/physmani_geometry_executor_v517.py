"""Reuse the PhysMani RRT/RML executor with geometry-supplied pose actions.

The neural policy is not constructed. The existing terminal-aware executor
owns RRT, waypoint extraction, RML, counted stepping, gripper and cleanup.
Geometry admission runs before each generated joint target advances physics.
This new component alone does not grant formal evaluation qualification.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from .physmani_executor_v7 import PhysManiAdapter as OriginalExecutor
from ..ablations.physmani_executor_v4 import (
    ProtocolInvariantError, physics_time, _check_clock)
from .evaluation import file_sha

SOURCES = ('our_mani/multitask/physmani_geometry_executor_v517.py',
    'our_mani/multitask/physmani_executor_v7.py',
    'our_mani/sensor_alignment/physmani_executor_v6.py',
    'our_mani/sensor_alignment/physmani_executor_v5.py',
    'our_mani/ablations/physmani_executor_v4.py')


def binding(root, native_repo):
    native = Path(native_repo)/'third_party/PyRep/pyrep'
    return dict(schema='r10_geometry_actions_on_original_physmani_RRT_RML_v517',
        sources_sha256={p:file_sha(Path(root)/p) for p in SOURCES},
        native_sources_sha256={str(native/p):file_sha(native/p) for p in (
            'robots/configuration_paths/arm_configuration_path.py',
            'robots/arms/arm.py','backend/sim.py')},
        inherited_executor='released_waypoints_counted_actuator_author_rgbd_terminal_v7',
        pose_action_dimension=8,decision_ticks=10,counted_dt=.05,
        neural_policy_or_baseline_checkpoint_constructed=False,
        original_RRT_RML_waypoints_gripper_and_cleanup_reused=True,
        baseline_implementation_or_parameters_modified=False,
        additional_public_geometry_admission_before_physics=True,
        actual_motion_is_measured_not_guaranteed=True,formal_VAL_TEST=False)


class GeometryTargetRejected(ValueError):
    pass


class GuardedPath:
    """Delegate the actual native path; reject a generated target before a tick."""
    def __init__(self, path, env, admission, events):
        object.__setattr__(self,'_source_path',path)
        object.__setattr__(self,'_owned_env',env)
        object.__setattr__(self,'_admission',admission)
        object.__setattr__(self,'_events',events)

    def __getattr__(self, name):
        return getattr(self._source_path,name)

    def __setattr__(self, name, value):
        setattr(self._source_path,name,value)

    def step(self):
        env = self._owned_env
        previous = np.asarray(env.arm.get_joint_target_positions(),float).copy()
        clock = physics_time()
        try:
            done = self._source_path.step()
            generated = self._source_path._joint_position_action
            if generated is not None:
                accepted, proof = self._admission(np.asarray(generated,float).copy())
                if type(accepted) is not bool:
                    raise ProtocolInvariantError('geometry target admission must return an explicit bool')
                self._events.append(dict(time=float(env.time),accepted=accepted,proof=deepcopy(proof)))
                if not accepted:
                    env.arm.set_joint_target_positions(previous.tolist())
                    raise GeometryTargetRejected('public_geometry_rejected_RML_joint_target')
            return done
        except BaseException:
            env.arm.set_joint_target_positions(previous.tolist())
            raise
        finally:
            _check_clock(clock,'geometry-admitted original RML target generation')


class PhysManiGeometryExecutor(OriginalExecutor):
    def __init__(self, config, *, observe_public_frame, admit_pose_action, admit_joint_target):
        if config.get('baseline_decision_ticks')!=10 or config.get('dt')!=.05:
            raise ValueError('reuse original10-tick/50ms PhysMani execution cadence')
        if not all(callable(f) for f in (observe_public_frame,admit_pose_action,admit_joint_target)):
            raise ValueError('explicit public observation and geometry admission callbacks required')
        # Deliberately omit the inherited neural-policy constructor. execute()
        # below follows the original class and original native path methods.
        self.config = config
        self.observe_public_frame = observe_public_frame
        self.admit_pose_action = admit_pose_action
        self.admit_joint_target = admit_joint_target
        self.geometry_events = []
        self.execution_events = []
        self._owned_env = None

    def observe(self, observation, robot):
        return self.observe_public_frame(observation,robot)

    def predict(self, *args, **kwargs):
        raise ProtocolInvariantError('geometry policy supplies pose actions; no PhysMani neural prediction')

    def execute(self, env, action, *, observer=None):
        if self._owned_env is not None and self._owned_env is not env:
            raise ProtocolInvariantError('RML geometry adapter cannot mix simulator instances')
        self._owned_env = env
        action = np.asarray(action,float)
        if action.shape!=(8,) or not np.isfinite(action).all():
            raise ValueError('finite position/quaternion/gripper action required')
        accepted, proof = self.admit_pose_action(action.copy())
        if accepted is not True:
            raise GeometryTargetRejected('public geometry pose action not admitted: '+str(proof))
        service = env.map_service
        original = service.plan_path
        def plan_path(*args, **kwargs):
            path = original(*args,**kwargs)
            return GuardedPath(path,env,self.admit_joint_target,self.geometry_events)
        service.plan_path = plan_path
        before = int(env.ledger.ticks)
        try:
            receipt = super().execute(env,action,observer=observer)
        finally:
            service.plan_path = original
        self.execution_events.append(dict(time_after=float(env.time),ticks_before=before,
            ticks_after=int(env.ledger.ticks),action=action.tolist(),pose_admission=deepcopy(proof),
            original_executor_receipt=deepcopy(receipt)))
        return receipt

    def geometry_receipt(self):
        return dict(schema='r10_geometry_actions_on_original_physmani_RRT_RML_v517',
            generated_target_admissions=deepcopy(self.geometry_events),
            actions=deepcopy(self.execution_events),native_policy_checkpoint_loaded=False,
            RRT_RML_and_native_gripper_executed_by_original_class=True,
            formal_VAL_TEST_qualification_granted=False)
