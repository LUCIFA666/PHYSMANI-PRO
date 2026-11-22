"""Qualify remaining original block sources without repeating completed source0."""
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs

B = ROOT/'artifacts/framework/multitask_r10'
TASK = 'pick_moving_target_on_the_table'
SETUP = B/'block_remaining_sources_v363_setup'
OUT = B/'block_remaining_sources_v363'


def command(raw, method):
    return ['bash', 'scripts/run_local_sim.sh', 'scripts/run_task_executor_smoke_block_v317_r10.py',
        '--config', str((SETUP/'configuration.json').relative_to(ROOT)),
        '--manifest', 'artifacts/framework/protocol/train_manifest.json',
        '--frozen-inputs', str((SETUP/'frozen_inputs.json').relative_to(ROOT)),
        '--task', TASK, '--raw-episode-id', raw, '--split', 'train', '--method', method,
        '--development-demo', '--out', str((OUT/'development_demo_only'/TASK/raw/method).relative_to(ROOT))]


def main():
    if SETUP.exists():
        raise FileExistsError('preserve existing preparation')
    previous = B/'joint16_block_support_v318_setup'
    c = json.loads((previous/'configuration.json').read_text())
    files = dict(json.loads((previous/'frozen_inputs.json').read_text())['files'])
    verify_inputs(ROOT, files)
    manifest = ROOT/'artifacts/framework/protocol/train_manifest.json'
    original = [r for r in json.loads(manifest.read_text())['episodes'] if r['task'] == TASK][:4]
    assert len(original) == 4 and original[0]['raw_episode_id'] == 'episode0'
    reused = []
    for method in ('cv_geometry', 'context_geometry'):
        path = B/'joint16_block_support_v318/development_demo_only'/TASK/'episode0'/method/'case.json'
        case = json.loads(path.read_text())
        assert case['complete'] and case['infrastructure_error'] is None
        assert case['identity']['config_sha256'] == file_sha(previous/'configuration.json')
        reused.append(dict(method=method, case=dict(path=str(path.relative_to(ROOT)), sha256=file_sha(path))))
    for row in original[1:]:
        path = Path(row['path']).resolve()
        sha = file_sha(path)
        if str(path) in files:
            assert files[str(path)] == sha
        files[str(path)] = sha
    files[str(Path(__file__).resolve())] = file_sha(__file__)
    SETUP.mkdir()
    atomic_json(SETUP/'configuration.json', c)
    assert file_sha(SETUP/'configuration.json') == file_sha(previous/'configuration.json')
    files[str(SETUP/'configuration.json')] = file_sha(SETUP/'configuration.json')
    atomic_json(SETUP/'frozen_inputs.json', dict(schema='r10_same318_block_original_remaining_sources_v363',
        files=files, approved_for_execution=True, native_started=False, formal_VAL_TEST=False))
    checks = []
    try:
        for row in original[1:]:
            raw = row['raw_episode_id']
            for method in ('cv_geometry', 'context_geometry'):
                argv = command(raw, method)
                r = subprocess.run(argv+['--check'], cwd=ROOT, capture_output=True, text=True)
                label = TASK+'_'+raw+'_'+method
                (SETUP/(label+'_CPU_stdout.log')).write_text(r.stdout)
                (SETUP/(label+'_CPU_stderr.log')).write_text(r.stderr)
                r.check_returncode()
                proof = json.loads(r.stdout)
                assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
                atomic_json(SETUP/(label+'_CPU_preflight.json'), proof)
                checks.append(dict(task=TASK, raw=raw, method=method, argv=argv))
        verify_inputs(ROOT, files)
        atomic_json(SETUP/'CPU_acceptance.json', dict(complete=True,
            created_utc=datetime.now(timezone.utc).isoformat(), actual_argv_preflights=checks,
            original_four_sources=[r['raw_episode_id'] for r in original],
            first_source_reused_without_retry=reused, policy_config_bytes_identical_to318=True,
            source_outcome_filtering=False, native_started=False, GPU_started=False, formal_VAL_TEST=False))
    except BaseException as error:
        atomic_json(SETUP/'preparation_failed.json', dict(type=type(error).__name__, message=str(error),
            native_started=False, GPU_started=False))
        raise
    print(json.dumps(dict(prepared=True, frozen_inputs=len(files), CPU_checks=len(checks),
        first_source_not_repeated=True, native_started=False)), flush=True)


if __name__ == '__main__':
    main()
