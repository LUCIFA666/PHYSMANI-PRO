"""TRAIN development diagnostic: profile existing checks without changing them."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from scripts import run_trained_task_case_r10 as case_runner
from our_mani.reachability import TimedReachability
from our_mani.multitask.evaluation import atomic_json, file_sha


def main():
    args = case_runner.parser().parse_args()
    if not args.development_demo or args.split != 'train' or args.check:
        raise ValueError('diagnostic accepts only actual TRAIN development demo execution')
    prepared = case_runner.preflight(args)
    out = prepared['out']
    if 'joint_rejection_diagnostic' not in out.parts:
        raise ValueError('diagnostic requires a separate joint_rejection_diagnostic namespace')
    target_code = TimedReachability.check_joints.__code__
    rows = []

    def profile(frame, event, result):
        if event != 'return' or frame.f_code is not target_code or result is None:
            return
        values = frame.f_locals
        robot, model = values['robot'], values['self'].model
        rows.append(dict(
            observed_time=float(robot.timestamp),
            duration=float(values['duration']),
            observed_joints=robot.joints.tolist(),
            observed_joint_velocities=robot.joint_velocities.tolist(),
            requested_joints=values['q'].tolist(),
            requested_joint_velocities=values['v'].tolist(),
            velocity_limits=model.velocity_limits.tolist(),
            acceleration_limits=model.acceleration_limits.tolist(),
            feasible=bool(result.feasible), reason=str(result.reason),
            evidence=result.evidence))

    error = None
    try:
        sys.setprofile(profile)
        case_runner.run_prepared(prepared)
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        sys.setprofile(None)
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / 'joint_checks.json', dict(
            schema='r10_readonly_existing_joint_check_profile_TRAIN_development_v1',
            created_utc=datetime.now(timezone.utc).isoformat(),
            actual_diagnostic_entrypoint=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=file_sha(__file__)),
            original_case_wrapper=case_runner.pointer(case_runner.__file__),
            original_check_source=case_runner.pointer(ROOT / 'our_mani/reachability.py'),
            input_lock=case_runner.pointer(case_runner.absolute(args.frozen_inputs)),
            identity=prepared['identity'], checks=rows, error=error,
            additional_native_queries=False, original_check_results_modified=False,
            policy_or_physical_parameters_modified=False,
            fitting_source=False, formal_VAL_or_TEST_metric=False))
    print(json.dumps(dict(complete=True, diagnostic=str(out / 'joint_checks.json'), checks=len(rows))))


if __name__ == '__main__':
    main()
