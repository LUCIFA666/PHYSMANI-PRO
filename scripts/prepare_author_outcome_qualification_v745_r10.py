"""Requalify all192 fixed TRAIN cases after correcting native outcome sampling.

The completed joint16 training and every operation/neural model are reused.
No old runtime case is reused across this physical protocol change.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json, subprocess, sys, traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.evaluation_roster16_v283 import TASKS, METHODS, validate_roster
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.common_rml_protocol_v530 import fingerprint
from our_mani.multitask.rml_case_preflight_v742 import SOURCES, outcome_protocol_binding
from scripts.run_native_resource_owner_v19_r10 import validate_request


def point(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=file_sha(path))


def main():
    base = ROOT/'artifacts/framework/multitask_r10'
    prior = base/'post_training_qualification_v680_setup'
    setup = base/'author_outcome_qualification_v745_setup'
    if setup.exists():
        raise FileExistsError('preserve this one-time new-protocol qualification')
    accepted = base/'joint16_after_executor_training_acceptance_v672/report.json'
    training = json.loads(accepted.read_text())
    if (training.get('training_complete') is not True
            or training['training']['TRAIN_original_parents'] != 1600
            or training['training']['completed_epochs'] != 30):
        raise ValueError('completed unchanged joint16 training required')
    verify_inputs(ROOT, training['inputs_sha256'])
    old_routes = json.loads((prior/'routes.json').read_text())
    if training['checkpoint']['sha256'] != old_routes['shared_context_checkpoint_sha256']:
        raise ValueError('do not retrain or swap the accepted checkpoint')
    handoff = ROOT/'runtime/framework/outcome_sampling_handoff_v742/scheduler_superseded.json'
    stopped = json.loads(handoff.read_text())
    if (stopped.get('complete') is not True
            or stopped.get('native_or_GPU_interrupted') is not False
            or stopped.get('CPU_interrupted') is not False
            or stopped.get('old_VAL_cohort_superseded') is not True
            or stopped.get('user_stop_received') is not False):
        raise ValueError('old-cohort natural completion and explicit supersession required')
    files = dict(json.loads((prior/'frozen_inputs.json').read_text())['files'])
    additional = [accepted, ROOT/training['checkpoint']['path'], handoff, Path(__file__),
                  ROOT/'scripts/audit_author_outcome_readiness_v746_r10.py',
                  ROOT/'scripts/queue_author_outcome_qualification_v747_r10.py',
                  ROOT/'scripts/run_native_resource_owner_v748_r10.py',
                  ROOT/'tests/test_native_outcome_sampling_v741.py',
                  ROOT/'tests/test_author_outcome_runtime_v743.py']
    for path in additional + [ROOT/name for name in SOURCES]:
        key, value = str(path.resolve()), file_sha(path)
        if key in files and files[key] != value:
            raise ValueError('old frozen evidence changed: '+key)
        files[key] = value
    manifest = ROOT/'artifacts/framework/protocol/train_manifest.json'
    roster = validate_roster(json.loads(manifest.read_text()), 'train')
    fixed = {t: [r for r in roster if r['task'] == t][:4] for t in TASKS}
    for task, rows in fixed.items():
        if len(rows) != 4:
            raise ValueError('the unchanged first four original TRAIN sources are required: '+task)
        for row in rows:
            files[str(Path(row['path']).resolve())] = file_sha(row['path'])
    protocol = outcome_protocol_binding(ROOT)
    routes = []
    worker = ROOT/'scripts/run_task_executor_smoke_author_outcome_v743_r10.py'
    consumer = ROOT/'scripts/run_task_executor_smoke_author_suite16_v744_r10.py'
    for task in TASKS:
        for method in METHODS:
            old = next(r for r in old_routes['routes'] if (r['task'], r['method']) == (task, method))
            config = ROOT/old['configuration']['path']
            if point(config) != old['configuration']:
                raise ValueError('no model or configuration change is permitted')
            c = json.loads(config.read_text())
            if method != 'physmani' and c['context_checkpoint'] != training['checkpoint']:
                raise ValueError('every geometry route must retain the accepted shared model')
            routes.append(dict(task=task, method=method, GPU_required=method == 'physmani',
                configuration=point(config), worker=point(worker), consumer=point(consumer),
                source_preflight=point(ROOT/'our_mani/multitask/rml_case_preflight_v742.py'),
                common_RML_profile_sha256=fingerprint(c['common_author_RML_profile']),
                author_outcome_protocol=protocol))
    verify_inputs(ROOT, files)
    setup.mkdir()
    lock = setup/'frozen_inputs.json'
    atomic_json(lock, dict(files=files, approved_for_execution=True,
        immutable_executables_models_and_TRAIN_qualification_sources=True,
        formal_split_data_locked_separately_after_admission=True, formal_VAL_TEST=False,
        author_outcome_protocol=protocol, prior_VAL_supersession=point(handoff)))
    for route in routes:
        route['frozen_inputs'] = point(lock)
    route_path = setup/'routes.json'
    atomic_json(route_path, dict(schema='r10_exact16_author_action_outcome_routes_v745',
        tasks=list(TASKS), methods=list(METHODS), routes=routes,
        shared_context_checkpoint_sha256=training['checkpoint']['sha256'],
        common_author_RML_profile=old_routes['common_author_RML_profile'],
        author_outcome_protocol=protocol))
    jobs = []
    for task in TASKS:
        for source in fixed[task]:
            raw = source['raw_episode_id']
            for method in METHODS:
                route = next(r for r in routes if (r['task'], r['method']) == (task, method))
                argv = ['bash', 'scripts/run_aligned_gpu.sh' if method == 'physmani' else 'scripts/run_local_sim.sh',
                    str(worker.relative_to(ROOT)), '--config', route['configuration']['path'],
                    '--manifest', str(manifest.relative_to(ROOT)), '--frozen-inputs', str(lock.relative_to(ROOT)),
                    '--task', task, '--raw-episode-id', raw, '--split', 'train', '--method', method,
                    '--development-demo', '--out', str((base/'author_outcome_qualification_v745'/
                        'development_demo_only'/task/raw/method).relative_to(ROOT))]
                name = f'author_outcome_v745_{task}_{raw}_{method.split("_")[0]}'
                validate_request(dict(schema='r10_owned_native_TRAIN_request_v19', kind='executor_smoke',
                    job_id=name, argv=argv, use_gpu=method == 'physmani', timeout_seconds=1800,
                    source_sha256=file_sha(worker)))
                jobs.append(dict(task=task, raw=raw, method=method, argv=argv, job_id=name))
    if len(jobs) != 192:
        raise ValueError('all192 new cases required; no old-protocol reuse')
    atomic_json(setup/'plan.json', dict(planned=jobs, reused=[], expected_method_cases=192, new_cases=192,
        existing_observed_baseline_cases=0, all_failures_retained=True, source_outcome_filtering=False,
        automatic_retry=False, formal_VAL_TEST=False, routes=point(route_path),
        training_acceptance=point(accepted), author_outcome_protocol=protocol))

    def check(row):
        return row, subprocess.run(row['argv']+['--check'], cwd=ROOT, text=True, capture_output=True)

    passed, failed = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_rows = {pool.submit(check, r): r for r in jobs}
        for future in as_completed(future_rows):
            row, result = future_rows[future], None
            try:
                row, result = future.result()
                if result.returncode:
                    raise RuntimeError('actual argv CPU subprocess returned '+str(result.returncode))
                proof = json.loads(result.stdout)
                if (proof.get('complete') is not True or proof.get('native_started') is not False
                        or proof.get('GPU_started') is not False or proof.get('native_runtime') != point(consumer)
                        or proof.get('author_outcome_protocol') != protocol):
                    raise ValueError('actual new-runtime CPU evidence must bind the outcome protocol')
            except Exception as error:
                atomic_json(setup/(row['job_id']+'_CPU_failed.json'), dict(stdout=result.stdout,
                    stderr=result.stderr, returncode=result.returncode, native_started=False, automatic_retry=False,
                    error=repr(error), traceback=traceback.format_exc()) if result is not None else dict(
                    native_started=False, automatic_retry=False, error=repr(error), traceback=traceback.format_exc()))
                failed.append(row)
                print(json.dumps(dict(CPU_failed=row['job_id'])), flush=True)
                continue
            atomic_json(setup/(row['job_id']+'_CPU_preflight.json'), proof)
            passed.append(row)
            print(json.dumps(dict(CPU_pass=len(passed), failed=len(failed), total=192,
                task=row['task'], method=row['method'])), flush=True)
    verify_inputs(ROOT, files)
    atomic_json(setup/'CPU_acceptance.json', dict(complete=not failed, actual_argv_preflights=passed,
        failed_preflights=failed, reused_observed_baseline_cases=[], native_started=False,
        formal_VAL_TEST=False, author_outcome_protocol=protocol))


if __name__ == '__main__':
    main()
