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
from our_mani.multitask.normal_cup_diagonal_native_v123 import binding as diagonal_binding
from our_mani.multitask.task_actor_backend_v69 import persistent_reference_continuation_identity
from our_mani.multitask.interaction import CandidateConfig,MPCConfig

B=ROOT/'artifacts/framework/multitask_r10';SETUP=B/'normal_cup_smoke_v137_setup';OUT=B/'normal_cup_smoke_v137'


def command(method):
    return ['bash','scripts/run_local_sim.sh','scripts/run_task_executor_smoke_normal_cup_v124_r10.py',
        '--config',str((SETUP/'configuration.json').relative_to(ROOT)),
        '--manifest','artifacts/framework/protocol/train_manifest.json',
        '--frozen-inputs',str((SETUP/'frozen_inputs.json').relative_to(ROOT)),
        '--task',TASK,'--raw-episode-id','episode102','--split','train','--method',method,
        '--replica','0','--development-demo','--out',str((OUT/'development_demo_only'/TASK/'episode102'/method).relative_to(ROOT))]


def main():
    if SETUP.exists():raise FileExistsError('preserve old frozen/failed preparation')
    previous=B/'normal_cup_smoke_v126_setup';old=json.loads((previous/'frozen_inputs.json').read_text());verify_inputs(ROOT,old['files'])
    c=json.loads((previous/'configuration.json').read_text());report_path=B/'normal_cup_primary_policy_v110/report.json'
    report=json.loads(report_path.read_text());verify_inputs(ROOT,report['verified_files_sha256'])
    c['learned_artifacts'][TASK]['interaction']=report['model']
    c['normal_cup_primary_report']=dict(path=str(report_path.relative_to(ROOT)),sha256=file_sha(report_path))
    c['normal_cup_primary_v111']=binding(ROOT,c)
    c.update(execution_scope='normal_cup_3D_TRAIN_association_development_v137',formal_comparison_ready=False)
    association_path=B/'cup_public_association_radius_TRAIN_v133/report.json'
    association=json.loads(association_path.read_text());verify_inputs(ROOT,association['verified_files_sha256'])
    assert association['first_public_contact_samples']==association['new_radius_unique_intended_samples']==399
    c['attachment_association_distance_m']=association['association_radius_m']
    c['attachment_association_threshold_status']='TRAIN400 acquisitions /200 original parents; phase-free first-contact public pose calibration v133; native held IDs not read'
    c['cup_public_association_v133']=dict(path=str(association_path.relative_to(ROOT)),sha256=file_sha(association_path))
    library=PrimarySequenceLibrary.load(ROOT/report['model']['path'])
    c['normal_cup_diagonal_v123']=diagonal_binding(ROOT,c)
    original_close=dict(c['normal_cup_close_intersection_v120']['original'])
    acquire=library.for_phase('acquire')
    c['gripper_timing_config']['close_position_tolerance_m']=min(original_close['close_position_tolerance_m'],*(t.position_tolerance_m for t in acquire))
    c['gripper_timing_config']['close_orientation_tolerance_rad']=min(original_close['close_orientation_tolerance_rad'],*(t.orientation_tolerance_rad for t in acquire))
    c['normal_cup_close_intersection_v120']=dict(original=original_close,actual=c['gripper_timing_config'],source='intersection of original physical close gate and fitted acquire completion cap; same gate used by capture and close',caps_not_loosened=True)
    for predictor in ('cv','context'):
        identity=persistent_reference_continuation_identity(ROOT,c,TASK,predictor,library,
            CandidateConfig(**c['candidate_config']),MPCConfig(**c['mpc_config']))
        c['continuation_identities'].setdefault(TASK,{})[predictor]=identity['sha256']
    files=dict(old['files'])
    for name,checksum in report['verified_files_sha256'].items():
        if name in files:assert files[name]==checksum
        files[name]=checksum
    for name,checksum in association['verified_files_sha256'].items():
        if name in files:assert files[name]==checksum
        files[name]=checksum
    paths=[association_path,ROOT/'tests/test_cup_public_association_v133.py',Path(__file__),ROOT/'scripts/run_task_executor_smoke_normal_cup_v124_r10.py',report_path,
        ROOT/report['model']['path'],ROOT/report['manifest']['path'],
        ROOT/'our_mani/multitask/input_stat_cache_v79.py',ROOT/'our_mani/multitask/frozen_input_index_v93.py',
        ROOT/'tests/test_normal_cup_primary_library_v109.py',ROOT/'tests/test_normal_cup_native_v111.py']
    paths.extend(ROOT/name for name in SOURCES)
    paths.extend(ROOT/name for name in ('our_mani/multitask/normal_cup_diagonal_native_v123.py','our_mani/multitask/bounded_approach_v121.py','tests/test_bounded_approach_diagonal_lag_v121.py'))
    paths.extend(ROOT/p['path'] for p in c['learned_artifacts'][TASK].values() if p)
    for p in paths:
        p=p.resolve();checksum=file_sha(p)
        if str(p) in files:assert files[str(p)]==checksum
        files[str(p)]=checksum
    SETUP.mkdir();atomic_json(SETUP/'configuration.json',c);files[str(SETUP/'configuration.json')]=file_sha(SETUP/'configuration.json')
    atomic_json(SETUP/'frozen_inputs.json',dict(schema='r10_normal_cup_3D_TRAIN_association_v137_lock',files=files,
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
            actual_argv_preflights=checks,consumer_tests_each_environment=4,factory_tests_each_environment=3,diagonal_lag_tests_each_environment=5,association_tests_each_environment=4,
            all100_original_parents_retained=True,old_full_scope_gate_passed=False,
            admitted_initial_sequence_original_gates_passed=True,native_started=False,GPU_started=False,formal_VAL_TEST=False))
    except BaseException as e:
        atomic_json(SETUP/'preparation_failed.json',dict(type=type(e).__name__,message=str(e),native_started=False,GPU_started=False));raise
    print(json.dumps(dict(prepared=True,frozen_inputs=len(files),CPU_checks=len(checks),native_started=False)),flush=True)


if __name__=='__main__':main()
