"""Serial owner for exactly four preassigned calibration-only captures."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT/'scripts/launch_near_goal_fresh4_calibration_20261103_v2.py'
FAMILY = ROOT/'runtime/framework/near_goal_perception_20261103'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.plan.resolve(strict=True)
    if digest(plan_path) != args.expected_plan_sha256:
        raise ValueError('Preassigned capture plan changed')
    plan = json.loads(plan_path.read_text())
    parents = plan['parents']
    if (plan.get('scope') != 'fresh4_calibration'
            or [p['seed'] for p in parents] != list(range(261103632, 261103636))):
        raise ValueError('Only the new four-parent calibration cohort')
    output = args.output.resolve()
    if FAMILY.resolve() not in output.parents or (FAMILY/'STOP').exists():
        raise ValueError('New calibration family with no administrative STOP required')
    output.mkdir(parents=True, exist_ok=False)
    source_sha = {str(p): digest(p) for p in (Path(__file__).resolve(), HOST, plan_path)}
    write_new(output/'queue.request.json', dict(
        started_utc=datetime.now(timezone.utc).isoformat(), source_sha256=source_sha,
        parent_ids=[p['parent_id'] for p in parents], serial=True,
        calibration_only=True, fit_started=False, grasp_VAL_TEST=False))
    for parent in parents:
        if (output/'STOP').exists() or (FAMILY/'STOP').exists():
            raise RuntimeError('Administrative queue STOP preserved')
        for path, sha in source_sha.items():
            if digest(path) != sha:
                raise RuntimeError('Queue, host or plan changed')
        case = output/parent['parent_id']
        command = [sys.executable, str(HOST), '--execute', '--plan', str(plan_path),
                   '--parent-id', parent['parent_id'], '--output', str(case)]
        result = subprocess.run(command, cwd=ROOT, check=False)
        owner = case.parent/(case.name+'.owner.actual_exit.json')
        actual = json.loads(owner.read_text()) if owner.exists() else {}
        write_new(output/(case.name+'.queue_wait.json'), dict(
            host_actual_returncode=result.returncode, host_wait_completed=True,
            actual_exit_path=str(owner), actual_exit_sha256=digest(owner) if owner.exists() else None))
        if (result.returncode != 0 or actual.get('actual_process_returncode') != 0
                or actual.get('actual_wait_completed') is not True
                or actual.get('source_unchanged') is not True or actual.get('stop_reason') is not None):
            raise RuntimeError('Capture failed; retain failed case and stop without replacement')
        print(json.dumps(dict(event='capture_closed', parent_id=parent['parent_id'])), flush=True)
    write_new(output/'queue.complete.json', dict(
        complete=True, parents=4, shots=32, all_hosts_waited=True,
        all_actual_Native_exits_zero=True, serial=True, calibration_only=True,
        fit_started=False, grasp_VAL_TEST=False, source_sha256=source_sha,
        finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
