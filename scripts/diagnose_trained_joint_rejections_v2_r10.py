"""TRAIN development diagnostic: record original check return values unchanged."""
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
    original_check = TimedReachability.check_joints
    rows = []

    def observed_check(self, robot, q, v, duration):
        result = original_check(self, robot, q, v, duration)
        model = self.model
        import numpy as np
        rows.append(dict(
            observed_time=float(robot.timestamp),
            duration=float(duration),
            observed_joints=robot.joints.tolist(),
            observed_joint_velocities=robot.joint_velocities.tolist(),
            requested_joints=np.asarray(q).tolist(),
            requested_joint_velocities=np.asarray(v).tolist(),
            velocity_limits=model.velocity_limits.tolist(),
            acceleration_limits=model.acceleration_limits.tolist(),
            feasible=bool(result.feasible), reason=str(result.reason),
            evidence=result.evidence))
        return result

    error = None
    try:
        TimedReachability.check_joints = observed_check
        prepared['args'].record_demo = False
        prepared['args'].demo_only = False
        case_runner.run_prepared(prepared)
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        TimedReachability.check_joints = original_check
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / 'joint_checks.json', dict(
            schema='r10_observed_original_joint_check_return_TRAIN_development_v2',
            created_utc=datetime.now(timezone.utc).isoformat(),
            actual_diagnostic_entrypoint=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=file_sha(__file__)),
            original_case_wrapper=case_runner.pointer(case_runner.__file__),
            original_check_source=case_runner.pointer(ROOT / 'our_mani/reachability.py'),
            input_lock=case_runner.pointer(case_runner.absolute(args.frozen_inputs)),
            identity=prepared['identity'], checks=rows, error=error,
            additional_native_queries=False, original_check_results_modified=False,
            global_Python_profiler_used=False, additional_video_saved=False,
            policy_or_physical_parameters_modified=False,
            fitting_source=False, formal_VAL_or_TEST_metric=False))
    print(json.dumps(dict(complete=True, diagnostic=str(out / 'joint_checks.json'), checks=len(rows))))


if __name__ == '__main__':
    main()
