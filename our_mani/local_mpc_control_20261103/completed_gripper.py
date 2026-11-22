"""Preserve completion of the original nonblocking gripper actuation.

PyRep.actuate clears its progress cache and stops its joint velocities when it
returns True. Calling it again with the same blocked close target restarts the
motion. This adapter keeps that completed command until the request changes.
Completion is actuator status, never attachment/contact/task-success evidence.
"""
import math


class CompletedGripperBackend:
    def __init__(self, backend):
        self.backend = backend
        self._target = None
        self._done = False
        self._fault = False
        self._calls = 0
        self._held_completed = 0
        self._changes = 0
        self._completed = 0
        self._retarget_rearms = 0

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def terminal(self):
        return self._fault or self.backend.terminal()

    def actuate_gripper(self, opening):
        if not math.isfinite(opening) or not 0<=opening<=1:
            raise ValueError('finite normalized Native gripper target required')
        if self.terminal():
            raise RuntimeError('completed-command adapter cannot command a terminal backend')
        requested = float(opening >= .5)
        before = float(self.backend.clock())
        changed = requested != self._target
        previous_unfinished = self._target is not None and not self._done
        if changed:
            self._target, self._done = requested, False
            self._changes += 1
        if self._done:
            # The original completed call has already set motor velocities to
            # zero. No new actuation, grasp/release attempt or physics occurs.
            self._held_completed += 1
            return True
        try:
            def original_call():
                self._calls += 1
                value = self.backend.actuate_gripper(requested)
                if not isinstance(value,bool):
                    raise ValueError('explicit original Native completion boolean required')
                if abs(float(self.backend.clock())-before)>1e-9:
                    raise RuntimeError('Native gripper command advanced physical time')
                return value
            done = original_call()
            if changed and previous_unfinished and done:
                # PyRep interprets a reversal of its previous velocity as
                # oscillation and returns True while clearing the old cache.
                # One bounded no-step call re-arms the new request. A request
                # already at its endpoint still returns True on that call.
                self._retarget_rearms += 1
                done = original_call()
        except BaseException:
            self._fault = True
            raise
        self._done = done
        self._completed += int(done)
        return done

    def gripper_completion_receipt(self):
        return dict(protocol='original_Native_actuate_until_done_then_hold_until_target_changes',
            target=self._target,actuator_done=self._done,faulted=self._fault,
            Native_actuate_calls=self._calls,completed_same_target_ticks_not_restarted=self._held_completed,
            requested_target_changes=self._changes,completed_commands=self._completed,
            unfinished_target_changes_rearmed_without_physics=self._retarget_rearms,
            physics_steps_added=0,Native_task_or_motor_source_modified=False,
            completion_implies_held=False,completion_implies_contact=False,
            previous_finger_response_checkpoint_qualified_for_this_protocol=False)
