"""No-pose-write arm-vs-frozen-static-map query for the new MPC backend.

Integration replacement, inside the backend's existing q/qdot/servo/IK probe:

    collision = check_static_arm_collision(
        self.raw.map_service, clock=self._clock_tuple, receipt=query_receipt)

Only collidable flags are mutated.  In particular, this helper does not call
SharedKnownMap.check()/estimated_target(), target.get_pose()/set_pose(), any
joint setter, or any physics step.  The caller still owns joint restoration
and its existing native-state/clock checks.  The new diagnostic backend imports this helper; frozen original sources remain unchanged.
"""
from typing import Any, Callable, MutableMapping, Optional


class StaticCollisionRestoreError(RuntimeError):
    """One or more collidable flags could not be restored; fail closed."""

    def __init__(self, failures, *, clock_unchanged):
        self.failures = tuple(failures)
        self.clock_unchanged = clock_unchanged
        super().__init__("static collision flag restoration failed: " + repr(self.failures)
                         + "; clock_unchanged=" + repr(clock_unchanged))


class StaticCollisionClockError(RuntimeError):
    """A supposedly read-only query advanced the supplied physical clock."""


def check_static_arm_collision(service, *, clock: Callable[[], Any],
                               receipt: Optional[MutableMapping[str, Any]] = None) -> bool:
    """Query ``service.env.arm`` against immutable static ghosts only.

    ``service`` follows SharedKnownMap's existing ``static``, ``ghosts``,
    ``target``, ``env.arm`` and ``queries`` contract. ``clock`` should read the
    backend's whole (physics, task time, step id, ledger tick) tuple. The tuple
    must remain exactly equal because this function is not allowed to step.

    Save every flag before the first mutation, restore the exact prior flags
    even when a query or setup setter raises, and attempt all restoration
    setters even if an earlier one fails. No world pose writeback is required.
    """
    receipt = {} if receipt is None else receipt
    before_clock = clock()
    receipt.update(schema="static_only_no_target_pose_write_20261103_v1",
                   native_query_performed=False, collision=None,
                   target_pose_read=False, target_pose_setter_called=False,
                   joint_setter_called=False, physics_step_called=False,
                   physical_steps=None, flags_restored=False,
                   clock_before=before_clock, clock_after=None,
                   clock_unchanged=None, restoration_failures=[])
    # These lists must be disjoint in SharedKnownMap. Checking Python identity
    # is sufficient for its stored objects and avoids native calls or requiring
    # PyRep objects to be hashable.
    specifications = [("static", shape, False) for shape in service.static]
    specifications += [("ghost", shape, True) for shape in service.ghosts]
    specifications += [("target", service.target, False)]
    if len({id(shape) for _, shape, _ in specifications}) != len(specifications):
        raise ValueError("static, ghost and target flag objects must be distinct")
    saved = [(role, index, shape, bool(shape.is_collidable()), desired)
             for index, (role, shape, desired) in enumerate(specifications)]
    receipt["flag_count"] = len(saved)
    primary_error = None
    try:
        for _, _, shape, _, desired in saved:
            shape.set_collidable(desired)
        service.queries += 1
        receipt["native_query_performed"] = True
        collision = bool(service.env.arm.check_arm_collision())
        receipt["collision"] = collision
        return collision
    except BaseException as error:
        primary_error = error
        raise
    finally:
        failures = []
        for role, index, shape, previous, _ in reversed(saved):
            try:
                shape.set_collidable(previous)
            except BaseException as error:
                failures.append(dict(role=role, index=index, prior_collidable=previous,
                                     error=type(error).__name__ + ": " + str(error)))
        # Read back every flag too: a setter that silently fails is not recovery.
        for role, index, shape, previous, _ in saved:
            try:
                if bool(shape.is_collidable()) != previous:
                    failures.append(dict(role=role, index=index, prior_collidable=previous,
                                         error="restored collidable flag differs from saved value"))
            except BaseException as error:
                failures.append(dict(role=role, index=index, prior_collidable=previous,
                                     error="flag readback " + type(error).__name__ + ": " + str(error)))
        clock_error = None
        try:
            after_clock = clock()
            clock_unchanged = bool(after_clock == before_clock)
            receipt.update(clock_after=after_clock, clock_unchanged=clock_unchanged,
                           physical_steps=0 if clock_unchanged else None)
            if not clock_unchanged:
                clock_error = StaticCollisionClockError("static collision query advanced physics/task/control clock")
        except BaseException as error:
            clock_error = error
            receipt["clock_unchanged"] = False
        receipt.update(flags_restored=not failures, restoration_failures=failures)
        if failures:
            raise StaticCollisionRestoreError(failures, clock_unchanged=receipt["clock_unchanged"]) from primary_error
        if clock_error is not None:
            raise clock_error from primary_error
