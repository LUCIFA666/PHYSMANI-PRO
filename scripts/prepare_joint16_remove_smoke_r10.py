"""Fill the two missing Context runtime rows with the existing shared16 model."""
from pathlib import Path
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.interaction import InteractionLibrary, CandidateConfig, MPCConfig
from our_mani.multitask.task_actor_backend_v69 import persistent_reference_continuation_identity


def main():
    base = ROOT/'artifacts/framework/multitask_r10'
    setup, old = base/'joint16_remove_v405_setup', base/'joint16_buzz_smoke_v175_setup'
    if setup.exists():
        raise FileExistsError('preserve previous preparation')
    config = json.loads((old/'configuration.json').read_text())
    files = json.loads((old/'frozen_inputs.json').read_text())['files']
    tasks = ('remove_cups_from_rotating_frame', 'remove_cups_from_rotating_frame_high_speed')
    for task in tasks:
        library = InteractionLibrary.load(ROOT/config['learned_artifacts'][task]['interaction']['path'])
        identity = persistent_reference_continuation_identity(ROOT, config, task, 'context', library,
            CandidateConfig(**config['candidate_config']), MPCConfig(**config['mpc_config']))
        config['continuation_identities'].setdefault(task, {})['context'] = identity['sha256']
    config.update(execution_scope='TRAIN_only_existing_joint16_remove_runtime_v405', formal_comparison_ready=False)
    files[str(Path(__file__))] = file_sha(__file__)
    verify_inputs(ROOT, files)
    setup.mkdir()
    atomic_json(setup/'configuration.json', config)
    files[str(setup/'configuration.json')] = file_sha(setup/'configuration.json')
    atomic_json(setup/'frozen_inputs.json', dict(files=files, approved_for_execution=True,
        schema='r10_existing_joint16_remove_runtime405_lock', formal_VAL_TEST=False))
    plan = json.loads((base/'all16_executor_joint_train_v1/plan.json').read_text())['tasks']
    checks = []
    for task in tasks:
        raw = next(r['fixed_smoke_source'] for r in plan if r['task'] == task)
        argv = ['bash', 'scripts/run_local_sim.sh', 'scripts/run_task_executor_smoke_joint16_v174_r10.py',
            '--config', str((setup/'configuration.json').relative_to(ROOT)),
            '--manifest', 'artifacts/framework/protocol/train_manifest.json',
            '--frozen-inputs', str((setup/'frozen_inputs.json').relative_to(ROOT)),
            '--task', task, '--raw-episode-id', raw, '--split', 'train', '--method', 'context_geometry',
            '--replica', '0', '--development-demo', '--out',
            str((base/'joint16_remove_v405'/'development_demo_only'/task/raw/'context_geometry').relative_to(ROOT))]
        result = subprocess.run(argv+['--check'], cwd=ROOT, text=True, capture_output=True)
        if result.returncode:
            atomic_json(setup/'preparation_failed.json', dict(task=task, stdout=result.stdout,
                stderr=result.stderr, native_started=False))
        result.check_returncode()
        proof = json.loads(result.stdout)
        assert proof['complete'] and not proof['native_started'] and not proof['GPU_started']
        atomic_json(setup/(task+'_CPU_preflight.json'), proof)
        checks.append(dict(task=task, raw=raw, method='context_geometry', argv=argv))
        print(json.dumps(dict(CPU_pass=task)), flush=True)
    verify_inputs(ROOT, files)
    atomic_json(setup/'CPU_acceptance.json', dict(complete=True, actual_argv_preflights=checks,
        existing_shared16_and_qualified_operation_models_unchanged=True,
        native_started=False, formal_VAL_TEST=False))
    print(json.dumps(dict(prepared=True, checks=len(checks), frozen_inputs=len(files))), flush=True)


if __name__ == '__main__':
    main()
