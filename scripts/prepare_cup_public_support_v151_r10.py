"""Freeze the two original first-source cup tasks with public support geometry."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.cup_public_support_v148 import binding,TASKS
from our_mani.multitask.task_actor_backend_v69 import persistent_reference_continuation_identity
from our_mani.multitask.normal_cup_primary_library_v109 import PrimarySequenceLibrary
from our_mani.multitask.interaction import InteractionLibrary,CandidateConfig,MPCConfig

B=ROOT/'artifacts/framework/multitask_r10';SETUP=B/'cup_public_support_v151_setup';OUT=B/'cup_public_support_v151'


def command(task,raw,method):
    return ['bash','scripts/run_local_sim.sh','scripts/run_task_executor_smoke_cup_v150_r10.py',
        '--config',str((SETUP/'configuration.json').relative_to(ROOT)),
        '--manifest','artifacts/framework/protocol/train_manifest.json',
        '--frozen-inputs',str((SETUP/'frozen_inputs.json').relative_to(ROOT)),
        '--task',task,'--raw-episode-id',raw,'--split','train','--method',method,
        '--replica','0','--development-demo','--out',str((OUT/'development_demo_only'/task/raw/method).relative_to(ROOT))]


def main():
    if SETUP.exists():raise FileExistsError('preserve previous preparation')
    previous=B/'high_cup_smoke_v136_setup';old=json.loads((previous/'frozen_inputs.json').read_text());verify_inputs(ROOT,old['files'])
    c=json.loads((previous/'configuration.json').read_text())
    c['cup_public_support_v148']={task:binding(ROOT,c,task) for task in TASKS}
    c.update(execution_scope='cup_public_support_TRAIN_development_v151',formal_comparison_ready=False)
    libraries={TASKS[0]:PrimarySequenceLibrary.load(ROOT/c['learned_artifacts'][TASKS[0]]['interaction']['path']),
        TASKS[1]:InteractionLibrary.load(ROOT/c['learned_artifacts'][TASKS[1]]['interaction']['path'])}
    for task,library in libraries.items():
        for predictor in ('cv','context'):
            identity=persistent_reference_continuation_identity(ROOT,c,task,predictor,library,CandidateConfig(**c['candidate_config']),MPCConfig(**c['mpc_config']))
            c['continuation_identities'].setdefault(task,{})[predictor]=identity['sha256']
    roster_path=ROOT/'artifacts/framework/protocol/train_manifest.json';roster=json.loads(roster_path.read_text())['episodes']
    sources={task:next(r['raw_episode_id'] for r in roster if r['task']==task) for task in TASKS}
    files=dict(old['files']);paths=[Path(__file__),ROOT/'scripts/run_task_executor_smoke_cup_v150_r10.py',
        ROOT/'our_mani/multitask/cup_support_native_v149.py',ROOT/'our_mani/multitask/cup_public_support_v148.py',
        ROOT/'tests/test_cup_public_support_v148.py',ROOT/'our_mani/multitask/normal_cup_preflight_v139.py']
    for descriptor in c['cup_public_support_v148'].values():paths.extend(ROOT/name for name in descriptor['sources_sha256'])
    for p in paths:
        p=p.resolve();checksum=file_sha(p)
        if str(p) in files:assert files[str(p)]==checksum
        files[str(p)]=checksum
    SETUP.mkdir();atomic_json(SETUP/'configuration.json',c);files[str(SETUP/'configuration.json')]=file_sha(SETUP/'configuration.json')
    atomic_json(SETUP/'frozen_inputs.json',dict(schema='r10_cup_public_support_TRAIN_smoke_v151_lock',files=files,
        approved_for_execution=True,native_started=False,GPU_started=False,formal_VAL_TEST=False))
    try:
        checks=[]
        for task in TASKS:
            for method in ('cv_geometry','context_geometry'):
                raw=sources[task];argv=command(task,raw,method);result=subprocess.run(argv+['--check'],cwd=ROOT,text=True,capture_output=True)
                name=task+'_'+method
                (SETUP/(name+'_CPU_stdout.log')).write_text(result.stdout);(SETUP/(name+'_CPU_stderr.log')).write_text(result.stderr)
                result.check_returncode();d=json.loads(result.stdout);assert d['complete'] and not d['native_started'] and not d['GPU_started']
                atomic_json(SETUP/(name+'_CPU_preflight.json'),d);checks.append(dict(method=method,task=task,raw=raw,argv=argv))
        verify_inputs(ROOT,files)
        atomic_json(SETUP/'CPU_acceptance.json',dict(complete=True,created_utc=datetime.now(timezone.utc).isoformat(),
            actual_argv_preflights=checks,support_contact_fullR_quality_tests_each_environment=6,
            original_cup_operation_model_scopes_retained=True,original_motion_collision_clock_gates_not_loosened=True,
            exogenous_tracker_result_and_history_unchanged=True,native_started=False,GPU_started=False,formal_VAL_TEST=False))
    except BaseException as error:
        atomic_json(SETUP/'preparation_failed.json',dict(type=type(error).__name__,message=str(error),native_started=False,GPU_started=False));raise
    print(json.dumps(dict(prepared=True,frozen_inputs=len(files),CPU_checks=len(checks),sources=sources,native_started=False)),flush=True)


if __name__=='__main__':main()
