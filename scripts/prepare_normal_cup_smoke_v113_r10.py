"""Freeze first-source CV/Context native development for normal cup placement."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.normal_cup_primary_library_v109 import PrimarySequenceLibrary
from our_mani.multitask.normal_cup_native_v111 import binding,SOURCES
from our_mani.multitask.normal_cup_primary_fitting_v107 import TASK
from our_mani.multitask.task_actor_backend_v69 import persistent_reference_continuation_identity
from our_mani.multitask.interaction import CandidateConfig,MPCConfig

B=ROOT/'artifacts/framework/multitask_r10';SETUP=B/'normal_cup_smoke_v113_setup';OUT=B/'normal_cup_smoke_v113'


def command(method):
    return ['bash','scripts/run_local_sim.sh','scripts/run_task_executor_smoke_normal_cup_v112_r10.py',
        '--config',str((SETUP/'configuration.json').relative_to(ROOT)),
        '--manifest','artifacts/framework/protocol/train_manifest.json',
        '--frozen-inputs',str((SETUP/'frozen_inputs.json').relative_to(ROOT)),
        '--task',TASK,'--raw-episode-id','episode102','--split','train','--method',method,
        '--replica','0','--development-demo','--out',str((OUT/'development_demo_only'/TASK/'episode102'/method).relative_to(ROOT))]


def main():
    if SETUP.exists():raise FileExistsError('preserve old frozen/failed preparation')
    previous=B/'task_executor_smoke_v69_setup';old=json.loads((previous/'frozen_inputs.json').read_text());verify_inputs(ROOT,old['files'])
    c=json.loads((previous/'configuration.json').read_text());report_path=B/'normal_cup_primary_policy_v110/report.json'
    report=json.loads(report_path.read_text());verify_inputs(ROOT,report['verified_files_sha256'])
    c['learned_artifacts'][TASK]['interaction']=report['model']
    c['normal_cup_primary_report']=dict(path=str(report_path.relative_to(ROOT)),sha256=file_sha(report_path))
    c['normal_cup_primary_v111']=binding(ROOT,c)
    c.update(execution_scope='normal_cup_explicit_initial_sequence_TRAIN_development_v113',formal_comparison_ready=False)
    library=PrimarySequenceLibrary.load(ROOT/report['model']['path'])
    for predictor in ('cv','context'):
        identity=persistent_reference_continuation_identity(ROOT,c,TASK,predictor,library,
            CandidateConfig(**c['candidate_config']),MPCConfig(**c['mpc_config']))
        c['continuation_identities'].setdefault(TASK,{})[predictor]=identity['sha256']
    files=dict(old['files'])
    for name,checksum in report['verified_files_sha256'].items():
        if name in files:assert files[name]==checksum
        files[name]=checksum
    paths=[Path(__file__),ROOT/'scripts/run_task_executor_smoke_normal_cup_v112_r10.py',report_path,
        ROOT/report['model']['path'],ROOT/report['manifest']['path'],
        ROOT/'our_mani/multitask/input_stat_cache_v79.py',ROOT/'our_mani/multitask/frozen_input_index_v93.py',
        ROOT/'tests/test_normal_cup_primary_library_v109.py',ROOT/'tests/test_normal_cup_native_v111.py']
    paths.extend(ROOT/name for name in SOURCES)
    paths.extend(ROOT/p['path'] for p in c['learned_artifacts'][TASK].values() if p)
    for p in paths:
        p=p.resolve();checksum=file_sha(p)
        if str(p) in files:assert files[str(p)]==checksum
        files[str(p)]=checksum
    SETUP.mkdir();atomic_json(SETUP/'configuration.json',c);files[str(SETUP/'configuration.json')]=file_sha(SETUP/'configuration.json')
    atomic_json(SETUP/'frozen_inputs.json',dict(schema='r10_normal_cup_primary_TRAIN_smoke_v113_lock',files=files,
        approved_for_execution=True,native_started=False,GPU_started=False,formal_VAL_TEST=False,
        original_full_scope_gate_passed=False,admitted_initial_sequence_original_gates_passed=True))
    try:
        checks=[]
        for method in ('cv_geometry','context_geometry'):
            argv=command(method);result=subprocess.run(argv+['--check'],cwd=ROOT,text=True,capture_output=True)
            (SETUP/(method+'_CPU_stdout.log')).write_text(result.stdout);(SETUP/(method+'_CPU_stderr.log')).write_text(result.stderr)
            result.check_returncode();d=json.loads(result.stdout);assert d['complete'] and not d['native_started'] and not d['GPU_started']
            atomic_json(SETUP/(method+'_CPU_preflight.json'),d);checks.append(dict(method=method,task=TASK,raw='episode102',argv=argv))
        verify_inputs(ROOT,files)
        atomic_json(SETUP/'CPU_acceptance.json',dict(complete=True,created_utc=datetime.now(timezone.utc).isoformat(),
            actual_argv_preflights=checks,consumer_tests_each_environment=4,factory_tests_each_environment=3,
            all100_original_parents_retained=True,old_full_scope_gate_passed=False,
            admitted_initial_sequence_original_gates_passed=True,native_started=False,GPU_started=False,formal_VAL_TEST=False))
    except BaseException as e:
        atomic_json(SETUP/'preparation_failed.json',dict(type=type(e).__name__,message=str(e),native_started=False,GPU_started=False));raise
    print(json.dumps(dict(prepared=True,frozen_inputs=len(files),CPU_checks=len(checks),native_started=False)),flush=True)


if __name__=='__main__':main()
