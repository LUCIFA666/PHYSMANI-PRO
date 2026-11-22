"""Serial, source-bound real16-task feedback-interface validation owner.

Default is metadata-only. Explicit --execute runs the fixed first Cube case,
then all remaining fixed cases, with no task-outcome-based case replacement.
Native startup/infrastructure failures stop this owner and preserve artifacts.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import signal

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('feedback_frozen_lifecycle', ROOT / 'scripts/run_large_network_training_continuation_20261104_v1.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
DEFAULT = ROOT / 'config/framework/physmani_feedback16_validation_20261104_v2/plan.json'
LOCK = ROOT / 'runtime/framework/physmani_feedback16_validation_20261104_v2.lock'
_old_stops = base.stop_paths
base.stop_paths = lambda config: tuple(_old_stops(config)) + (DEFAULT.parent / 'STOP',)


def load_plan(path, digest=None):
    path = Path(path).resolve(strict=True)
    if digest is not None and base.sha(path) != digest: raise ValueError('External frozen plan SHA required')
    plan = base.read(path)
    if plan.get('schema') != 'physmani_feedback16_real_interface_validation_20261104_v2':
        raise ValueError('Exact generated-TRAIN interface validation plan required')
    sys_path = str(ROOT)
    import sys
    if sys_path not in sys.path: sys.path.insert(0, sys_path)
    from our_mani.multitask_execution_pipeline_20261104_v1.registry import TASKS
    cases = plan['cases']
    if len(cases) != 16 or {c['task'] for c in cases} != set(TASKS) or cases[0]['task'] != 'pick_moving_target_on_the_table':
        raise ValueError('Fixed all16 native task cases, ordinary Cube first required')
    if len({c['id'] for c in cases}) != 16 or plan.get('formal_VAL_TEST') is not False:
        raise ValueError('Distinct generated TRAIN validation cases only')
    if plan.get('control_ticks_per_cycle') != 2 or plan.get('budget_ticks_per_case') != 16:
        raise ValueError('Fixed eight two-tick control cycles required')
    if set(plan['stage_timeout_seconds']) != {c['id'] for c in cases}:
        raise ValueError('One explicit case timeout required')
    plan['_config_path'] = str(path)
    plan['_config_sha256'] = base.sha(path)
    base.verify_sources(plan)
    for case in cases:
        calibration = Path(case['calibration_episode']).resolve(strict=True)
        if 'train_package_compressed' not in calibration.parts or calibration.name != 'data.h5':
            raise ValueError('Only predeclared actual TRAIN calibration may be read')
    return plan


def execute(path, digest):
    config = load_plan(path, digest)
    out = Path(config['output_root'])
    with base.driver_lock(LOCK):
        base.assert_running(config)
        if out.exists(): raise FileExistsError('Fresh validation owner only')
        for case in config['cases']:
            if (ROOT / case['output']).exists(): raise FileExistsError('No old case may be retried or overwritten')
        out.mkdir(parents=True, exist_ok=False)
        (out / 'owners').mkdir()
        base.enable_subreaper()
        base.write_new(out / 'request.json', dict(plan=base.pointer(path), source_map=config['method_sources'],
            fixed_case_ids=[c['id'] for c in config['cases']], child_backend_owns_global_locks=True,
            task_policy_success_rate_evaluated=False, formal_VAL_TEST=False))
        previous = {s: signal.signal(s, lambda n, f: base.SIGNAL_REQUEST.update(signal=n)) for s in (signal.SIGINT, signal.SIGTERM)}
        results = []
        status = dict(complete=False, cases_completed=0, interface_passed=0,
            task_policy_evaluated=False, formal_VAL_TEST=False, NN_forward_calls=0)
        try:
            for case in config['cases']:
                base.assert_running(config)
                result_path = ROOT / case['output'] / 'result.json'
                terminal = base.run_stage(config, case['id'], [
                    'scripts/launch_physmani_feedback_case_20261104_v2.py',
                    '--plan', str(Path(path).resolve()), '--plan-sha256', digest,
                    '--case-id', case['id'], '--output', str(ROOT / case['output'])], native=True)
                if terminal['actual_process_returncode'] != 0 or terminal['actual_wait_completed'] is not True or terminal['source_unchanged'] is not True or terminal['unreaped_descendants']:
                    raise RuntimeError('Actual clean worker exit/wait/source required')
                value = base.read(result_path)
                if value.get('native_closed') is not True or value.get('source_unchanged') is not True or value.get('complete') is not True:
                    raise RuntimeError('Complete actual native shutdown and result required')
                row = dict(case_id=case['id'], task=case['task'], actual_exit=base.pointer(out / 'owners' / case['id'] / 'actual_exit.json'),
                    result=base.pointer(result_path), interface_validation_passed=value['interface_validation_passed'],
                    actual_postreset_ticks=value['actual_postreset_ticks'],
                    full_schedule=value['full_16tick_actuator_schedule_completed'],
                    native_metrics=value['private_native_metrics'])
                results.append(row)
                base.append(out / 'case_ledger.jsonl', row)
                status.update(cases_completed=len(results), interface_passed=sum(r['interface_validation_passed'] for r in results))
                print(json.dumps(dict(event='case_completed', **row)), flush=True)
            base.verify_sources(config)
            status.update(complete=True, cases=results, all16_native_closed=True,
                all16_actual_worker_wait_completed=True, source_unchanged=True,
                full_actuation_schedules=sum(r['full_schedule'] for r in results))
            base.write_new(out / 'completion.json', status)
            return status
        except BaseException as error:
            status.update(error_type=type(error).__name__, error=str(error), cases=results)
            base.write_new(out / 'failure.json', status)
            raise
        finally:
            base.write_new(out / 'terminal_status.json', dict(status, finished_utc=base.now()))
            for signum, handler in previous.items(): signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=DEFAULT)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.execute:
        if not args.plan_sha256: parser.error('Explicit --execute requires frozen --plan-sha256')
        value = execute(args.plan, args.plan_sha256)
    else:
        plan = load_plan(args.plan, args.plan_sha256)
        value = dict(complete=True, preparation_only=True, cases=len(plan['cases']), Native_started=False,
            child_processes_started=0, NN_forward_calls=0, source_map=plan['method_sources'])
    print(json.dumps(value, ensure_ascii=False))
    return 0


if __name__ == '__main__': raise SystemExit(main())
