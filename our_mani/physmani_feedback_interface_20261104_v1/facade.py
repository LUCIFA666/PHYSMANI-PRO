"""Latest-plan façade over the existing counted executor and gripper latch.

Submission, IK, observation and the original eight configuration queries add no
physics. Only advance may execute the original two 50 ms ticks. An explicit
gripper request is independent of a learned command's pending opening intent.
"""
from copy import deepcopy
import math
import numpy as np

from ..local_mpc_20261102.executor import (
    ShortHorizonExecutor, JointLimits, ExecutionConfig, PoseTarget, ExecutionFailure)
from ..local_mpc_control_20261103.completed_gripper import CompletedGripperBackend
from ..generic_reference_mpc_20261104_v1.reference_mpc import qualify_first_action
from .contracts import (VERSION, RECEIPT_SCHEMA, checked_gripper_request,
                        vector, action_bundle, digest, plain)


class FeedbackFacadeV1:
    def __init__(self, backend, executor=None, *, initial_gripper_opening=None):
        self.backend = backend
        before = float(backend.clock())
        if executor is None:
            native = backend.joint_limits()
            opening = getattr(backend, 'requested_opening', None) if initial_gripper_opening is None else initial_gripper_opening
            if opening is None or isinstance(opening, bool) or not math.isfinite(opening) or opening not in (0., 1.):
                raise ValueError('Actual existing binary actuator latch or explicit initial_gripper_opening required')
            limits = JointLimits(lower=native['lower'], upper=native['upper'], velocity=native['velocity'],
                                 acceleration=12., jerk=1000., provenance=native.get('provenance', {}))
            completed = CompletedGripperBackend(backend)
            executor = ShortHorizonExecutor(completed, limits, ExecutionConfig(), initial_gripper_opening=opening)
        elif initial_gripper_opening is not None:
            raise ValueError('An existing executor retains its actual h and gripper latch; no reinitialization allowed')
        if (not isinstance(executor, ShortHorizonExecutor)
                or type(executor.backend) is not CompletedGripperBackend
                or executor.backend.backend is not backend
                or executor.config.physics_dt != .05 or executor.config.control_ticks != 2
                or len(executor.limits.lower) != 7):
            raise TypeError('Original two-tick executor and exact completion adapter for this backend required')
        if float(backend.clock()) != before:
            raise RuntimeError('Facade construction must not advance physics')
        self.executor = executor
        self.completed_gripper = executor.backend
        self._serial = 0
        self._latest = None
        self._fault = None
        self.last_receipt = None

    def _binding(self):
        before = float(self.backend.clock())
        result = dict(time=before,
            q=vector(self.backend.get_joint_positions(), 7, 'actual q'),
            dq=vector(self.backend.get_joint_velocities(), 7, 'actual dq'),
            actuator_state=deepcopy(self.executor.state()))
        if not math.isfinite(before) or float(self.backend.clock()) != before:
            self._fault = 'Mechanical read advanced physics or returned invalid clock'
            raise RuntimeError(self._fault)
        return result

    @staticmethod
    def _same(a, b):
        return (abs(a['time']-b['time']) <= 1e-9
            and np.allclose(a['q'], b['q'], atol=1e-6, rtol=0)
            and np.allclose(a['dq'], b['dq'], atol=1e-9, rtol=0)
            and digest(a['actuator_state']) == digest(b['actuator_state']))

    def _install(self, binding, action, request, phase, error=None):
        if self._fault:
            raise RuntimeError(self._fault)
        if type(phase) is not str or not phase:
            raise ValueError('Explicit nonempty execution phase required')
        self._serial += 1
        self._latest = dict(plan_id=self._serial, binding=binding,
            action=deepcopy(action), action_sha256=digest(action),
            gripper_request=request, phase=phase, error=error, consumed=False)
        return self._serial

    def submit_joint_target(self, joints_or_action, *, gripper_request='KEEP', phase='feedback'):
        request = checked_gripper_request(gripper_request)
        action = action_bundle(joints_or_action)
        binding = self._binding()
        return self._install(binding, action, request, phase)

    def submit_target(self, position, quaternion, *, gripper_request='KEEP', phase='feedback'):
        request = checked_gripper_request(gripper_request)
        pose = PoseTarget(position, quaternion)
        binding = self._binding()
        error = None
        action = dict(target_position=pose.position, target_quaternion=pose.quaternion)
        try:
            action['joint_goal'] = vector(self.backend.solve_ik(pose.position, pose.quaternion), 7, 'IK joint goal')
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        after = self._binding()
        if not self._same(binding, after):
            self._fault = 'IK changed measured clock, robot or actuator state'
            raise RuntimeError(self._fault)
        return self._install(binding, action, request, phase, error)

    def _receipt(self, plan_id, status, *, reason=None, qualification=None, execution=None):
        plan = self._latest if self._latest and plan_id == self._latest['plan_id'] else None
        execution = execution.to_dict() if hasattr(execution, 'to_dict') else execution
        ticks = 0 if execution is None else execution['executed_ticks']
        out = dict(schema=RECEIPT_SCHEMA, version=VERSION, plan_id=plan_id, status=status,
            reason=reason, completed=status == 'cycle_completed', requested_ticks=2, executed_ticks=ticks,
            execution=execution, qualification=qualification,
            phase=None if plan is None else plan['phase'],
            gripper_request=None if plan is None else plan['gripper_request'],
            submitted_action_sha256=None if plan is None else plan['action_sha256'],
            submitted_action=None if plan is None else deepcopy(plan['action']),
            actuator_state=deepcopy(self.executor.state()),
            gripper_completion=self.completed_gripper.gripper_completion_receipt(),
            current_clock=float(self.backend.clock()),
            opening_from_BC_intent_issued=False,
            completed_implies_arm_arrival=False, completed_implies_grasp_or_contact=False,
            Native_task_metrics_in_feedback=False,
            qualifier_scope='original_first_action_endpoint_plus_7_configuration_queries_not_whole_path_or_H8',
            upstream_policy_phase_alignment_or_task_qualification_replaced=False)
        self.last_receipt = plain(out)
        return deepcopy(self.last_receipt)

    def advance(self, plan_id, max_ticks=2):
        if type(max_ticks) is not int or max_ticks != 2:
            raise ValueError('V1 supports exactly max_ticks=2, with original terminal/budget truncation only')
        if type(plan_id) is not int:
            raise ValueError('Exact integer plan_id required')
        plan = self._latest
        if plan is None or plan_id != plan['plan_id'] or plan['consumed']:
            return self._receipt(plan_id, 'stale_plan', reason='Only latest unconsumed plan may advance')
        plan['consumed'] = True
        if self._fault:
            return self._receipt(plan_id, 'interface_fault', reason=self._fault)
        if plan['error'] is not None:
            return self._receipt(plan_id, 'planning_failed', reason=plan['error'])
        state = self._binding()
        if not self._same(plan['binding'], state):
            return self._receipt(plan_id, 'state_changed', reason='Plan no longer binds current q/dq/clock/actual executor h')
        # No motor command and no collision probe may be issued after a terminal
        # or exhausted episode. In particular a rejected CLOSE cannot latch.
        before = state['time']
        remaining = self.backend.remaining_ticks()
        terminal = self.completed_gripper.terminal()
        if type(remaining) is not int or remaining < 0:
            raise ValueError('Nonnegative exact integer actual tick budget required')
        if not self._same(state, self._binding()):
            self._fault = 'Terminal/budget query changed mechanical state'
            return self._receipt(plan_id, 'interface_fault', reason=self._fault)
        if terminal or remaining == 0:
            execution = dict(status='truncated', requested_ticks=2, executed_ticks=0,
                start_clock=before, end_clock=before, executed_duration=0.,
                truncated_reason='budget' if remaining == 0 else 'terminal', ticks=[])
            return self._receipt(plan_id, 'truncated', execution=execution)
        try:
            allowed, qualification = qualify_first_action(self.backend, state, plan['action'])
            if not self._same(state, self._binding()):
                raise RuntimeError('First-action qualifier changed robot/clock/actual h')
        except Exception as exc:
            self._fault = type(exc).__name__ + ': ' + str(exc)
            return self._receipt(plan_id, 'qualification_error', reason=self._fault)
        if not allowed:
            return self._receipt(plan_id, 'qualification_rejected', reason=qualification['reason'], qualification=qualification)
        request = plan['gripper_request']
        opening = None if request == 'KEEP' else float(request == 'OPEN')
        try:
            execution = self.executor.execute_joint_target(plan['action']['joint_goal'], opening=opening, phase=plan['phase'])
        except ExecutionFailure as exc:
            self._fault = 'Executor failed: ' + str(exc)
            return self._receipt(plan_id, 'execution_failed', reason=self._fault,
                                 qualification=qualification, execution=exc.receipt)
        return self._receipt(plan_id, 'cycle_completed' if execution.status == 'completed' else 'truncated',
                             qualification=qualification, execution=execution)

    def read_snapshot(self):
        from .observation import validate_feedback_snapshot
        before = self._binding()
        snapshot = validate_feedback_snapshot(self.backend.read_snapshot())
        if not self._same(before, self._binding()):
            self._fault = 'Public snapshot changed robot/clock/actual h'
            raise RuntimeError(self._fault)
        if abs(snapshot['timestamp']-before['time']) > 1e-9:
            raise ValueError('Public snapshot must be the current actual physical frame')
        if (not np.allclose(snapshot['q'], before['q'], atol=1e-6, rtol=0)
                or not np.allclose(snapshot['dq'], before['dq'], atol=1e-9, rtol=0)):
            raise ValueError('Public snapshot q/dq must match this same mechanical frame')
        return snapshot

    def legacy_policy_packet(self, snapshot):
        from .observation import legacy_policy_packet
        return legacy_policy_packet(snapshot)

    def metric_receipt(self):
        """Separate evaluation channel; never appended to a public snapshot."""
        before = self._binding()
        result = deepcopy(self.backend.metric_receipt())
        if not self._same(before, self._binding()):
            raise RuntimeError('Native metric read changed mechanical state')
        return plain(result)
