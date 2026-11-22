"""Exact process-local high-cup TRAIN periodic candidate factory hook."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from . import train_teacher
from .train_teacher_requery import PlanningRequeryConfig
from .train_teacher_periodic_replan import PeriodicFeedbackReplanningFollower, PeriodicReplanningConfig

VERSION = 'r10_high_cup_TRAIN_periodic10_actual_feedback_factory_v1'
TASK = 'place_cups_on_rotating_frame_high_speed'
BEHAVIOR = 'TRAIN_teacher_committed_waypoint'
ORIGINAL_CLASS = PeriodicFeedbackReplanningFollower.__mro__[2]


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def scoped_periodic_replanning(case, *, expected_teacher_source_sha256,
                              periodic_configuration, failure_configuration, captures):
    if (case.get('split') != 'train' or case.get('task') != TASK
            or case.get('behavior') != BEHAVIOR or case.get('teacher_used') is not True
            or case.get('context_primary_view') is not False
            or case.get('interaction_eligible') is not True
            or type(case.get('replica')) is not int or case['replica'] != 2
            or type(case.get('behavior_replica')) is not int or case['behavior_replica'] != 2
            or type(case.get('maximum_collection_ticks')) is not int or case['maximum_collection_ticks'] != 200):
        raise ValueError('separate fixed high-cup TRAIN periodic replica2 scope required')
    if (not isinstance(periodic_configuration, PeriodicReplanningConfig)
            or not isinstance(failure_configuration, PlanningRequeryConfig)
            or not isinstance(captures, list) or captures):
        raise ValueError('fixed periodic/failure configurations and fresh captures required')
    source = Path(train_teacher.__file__).resolve()
    if file_sha(source) != expected_teacher_source_sha256 or train_teacher.CommittedWaypointFollower is not ORIGINAL_CLASS:
        raise ValueError('original TRAIN follower source/factory identity differs')
    receipt = dict(version=VERSION, case_id=case['case_id'], task=TASK, split='train', replica=2,
                   source_sha256=expected_teacher_source_sha256, started_utc=datetime.now(timezone.utc).isoformat(),
                   native_or_physics_steps_called_by_hook=False, infrastructure_error_retry=False,
                   original_collector_changed=False, original_factory_restored=False, complete=False)

    def factory(*args, **kwargs):
        if any(name in kwargs for name in ('periodic_configuration', 'failure_configuration', 'requery_configuration')) or captures:
            raise ValueError('one explicit periodic follower per fixed native case required')
        follower = PeriodicFeedbackReplanningFollower(*args, periodic_configuration=periodic_configuration,
            failure_configuration=failure_configuration, **kwargs)
        captures.append(follower)
        return follower

    train_teacher.CommittedWaypointFollower = factory
    failed = False
    try:
        yield receipt
    except BaseException:
        failed = True
        raise
    finally:
        owner_unchanged = train_teacher.CommittedWaypointFollower is factory
        train_teacher.CommittedWaypointFollower = ORIGINAL_CLASS
        source_unchanged = file_sha(source) == expected_teacher_source_sha256
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(),
            original_factory_restored=train_teacher.CommittedWaypointFollower is ORIGINAL_CLASS,
            hook_owner_unchanged=owner_unchanged, source_unchanged=source_unchanged,
            follower_instances=len(captures),
            actual_periodic_requeries=sum(f.periodic_requery_count for f in captures),
            actual_failed_planning_requeries=sum(f.requery_count for f in captures),
            pending_periodic_interruptions=sum(f._pending_periodic_command_id is not None for f in captures),
            raw_commands=sum(len(f.commands) for f in captures),
            complete=bool(not failed and owner_unchanged and source_unchanged and len(captures) == 1))
        if not owner_unchanged or not source_unchanged:
            raise ValueError('periodic TRAIN hook owner/source changed; original factory restored')
