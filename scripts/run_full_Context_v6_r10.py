"""Execute the frozen full-data Context recipe after final TRAIN recovery.

No native collection or infrastructure retry is launched. Actual recovery
termination, all37 committed outcomes, complete original2800 lineage and the
full1400/275 training view must pass before the existing GPU fitter is called.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask import acquisition_lineage as audit
from scripts.prepare_final_TRAIN_recovery_r10 import live_plan_processes

BASE = ROOT / 'artifacts/framework/multitask_r10'
RECIPE = BASE / 'full_training_launch_preparation_v6/launch_plan_v6.json'
RECIPE_SHA = '5a6d086b8d5877618168c5c7477f95542cec379a638b01e7c1a8524542aaae09'


def now():
    return datetime.now(timezone.utc).isoformat()


def run():
    directory = RECIPE.parent
    state_path = directory / 'execution_state.json'
    audit.require(not state_path.exists(), 'full Context workflow already attempted; no automatic relaunch')
    audit.require(audit.file_sha(RECIPE) == RECIPE_SHA, 'frozen full Context recipe changed')
    recipe = audit.read(RECIPE)
    source = audit.read(audit.checked_pointer(ROOT, recipe['source_snapshot'], {}))
    registrations = audit.read(audit.checked_pointer(ROOT, recipe['actual_registrations'], {}))
    audit.require(len(registrations) == 9, 'all actual original/recovery registrations required')
    plans = [audit.local(ROOT, row['plan']['path']) for row in registrations]
    setup = BASE / 'TRAIN_recovery_v20_final_remaining_setup'
    fixed = audit.read(setup / 'preparation_report.json')
    fixed_ids = fixed['fixed_case_ids']
    audit.require(len(fixed_ids) == 37, 'unchanged final37 recovery roster required')
    script_sha = audit.file_sha(__file__)
    state = dict(schema='r10_actual_full_Context_recipe_execution_v6', started_utc=now(),
                 recipe=audit.pointer(ROOT, RECIPE), script=audit.pointer(ROOT, Path(__file__)),
                 stage='waiting_for_actual_final37_recovery', jobs=[],
                 native_collection_or_retry_started=False, optimizer_launch_attempted=False,
                 training_complete=False, TEST_used_for_fitting=False)

    def persist():
        state['updated_utc'] = now()
        temp = state_path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temp.replace(state_path)

    def source_check():
        audit.require(audit.file_sha(RECIPE) == RECIPE_SHA and audit.file_sha(__file__) == script_sha,
                      'workflow or recipe changed during execution')
        for pointer in source['sources'].values():
            audit.checked_pointer(ROOT, pointer, {})

    persist()
    try:
        source_check()
        deadline = time.monotonic() + 1800
        while True:
            live = live_plan_processes(plans)
            state['actual_native_processes_observed'] = live
            persist()
            if not live:
                break
            audit.require(time.monotonic() < deadline,
                          'actual final recovery remains live after watcher budget; do not restart it')
            time.sleep(10)
        report_path = BASE / 'TRAIN_recovery_v20_final_remaining/queue_report.json'
        report = audit.read(report_path)
        expansion = audit.read(setup / 'ROOT_full37_expansion_review.json')
        audit.require(report.get('selected_case_ids') == fixed_ids
            and report.get('selected_complete') is True and report.get('automatic_retries') is False
            and len(report.get('receipts', [])) == 37
            and all(row['status'] in ('completed', 'resumed_verified') for row in report['receipts'])
            and audit.instant(report['created_utc']) >= audit.instant(expansion['created_utc']),
            'actual complete final37 report is missing, stale, partial or contains infrastructure errors')
        state['final37_terminal_report'] = audit.pointer(ROOT, report_path)
        state['actual_native_processes_observed'] = []
        logs = directory / 'execution_logs'
        logs.mkdir(exist_ok=False)
        for job in recipe['command_jobs']:
            source_check()
            audit.require(not live_plan_processes(plans), 'TRAIN native collection resumed unexpectedly')
            entry = dict(id=job['id'], argv=job['argv'], started_utc=now(),
                         environment_overrides=job.get('environment', {}))
            state['stage'] = job['id']
            state['jobs'].append(entry)
            if job['id'] == 'context_GPU':
                state['optimizer_launch_attempted'] = True
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', **job.get('environment', {}))
            log = logs / (job['id'] + '.log')
            with log.open('x') as stream:
                process = subprocess.Popen(job['argv'], cwd=ROOT, env=env,
                                           stdout=stream, stderr=subprocess.STDOUT)
                entry['actual_process_id'] = process.pid
                persist()
                code = process.wait()
            entry.update(returncode=code, finished_utc=now(), log=audit.pointer(ROOT, log))
            persist()
            audit.require(code == 0, 'full Context recipe job failed: ' + job['id'])
        source_check()
        fitted = BASE / 'context_se3_full_v6'
        status = audit.read(fitted / 'status.json')
        metadata = status.get('metadata', {})
        audit.require(status.get('stage') == 'complete' and metadata.get('training_complete') is True
            and metadata.get('completed_epochs') == 30 and metadata.get('VAL_source_episodes') == 275
            and metadata.get('training_source_manifest', {}).get('source_episode_count') == 1400
            and metadata.get('TEST_used_for_fitting') is False
            and status.get('checkpoint_sha256') == audit.file_sha(fitted / 'dynamics.pt'),
            'actual complete full1400/275 Context checkpoint evidence missing')
        state.update(stage='complete', training_complete=True, completed_utc=now(),
                     checkpoint=audit.pointer(ROOT, fitted / 'dynamics.pt'),
                     training_report=audit.pointer(ROOT, fitted / 'report.json'),
                     policy_VAL_or_TEST_release=False)
        persist()
        print(json.dumps(state, ensure_ascii=False), flush=True)
    except BaseException as error:
        state.update(stage='failed_preserved_no_automatic_retry', error_class=type(error).__name__,
                     error=str(error), finished_utc=now())
        persist()
        raise


if __name__ == '__main__':
    run()
