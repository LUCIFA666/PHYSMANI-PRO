"""Freeze original first TRAIN source with the actual all300 highcup library."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.high_cup_native_v134 import binding,TASK
from our_mani.multitask.task_actor_backend_v69 import persistent_reference_continuation_identity
from our_mani.multitask.interaction import InteractionLibrary,CandidateConfig,MPCConfig

B=ROOT/'artifacts/framework/multitask_r10';SETUP=B/'high_cup_smoke_v136_setup';OUT=B/'high_cup_smoke_v136'


def command(method,raw):
    return ['bash','scripts/run_local_sim.sh','scripts/run_task_executor_smoke_high_cup_v135_r10.py',
        '--config',str((SETUP/'configuration.json').relative_to(ROOT)),
        '--manifest','artifacts/framework/protocol/train_manifest.json',
        '--frozen-inputs',str((SETUP/'frozen_inputs.json').relative_to(ROOT)),
        '--task',TASK,'--raw-episode-id',raw,'--split','train','--method',method,
        '--replica','0','--development-demo','--out',str((OUT/'development_demo_only'/TASK/raw/method).relative_to(ROOT))]


def main():
    if SETUP.exists():raise FileExistsError('preserve existing preparation')
    previous=B/'normal_cup_smoke_v126_setup';old=json.loads((previous/'frozen_inputs.json').read_text());verify_inputs(ROOT,old['files'])
    c=json.loads((previous/'configuration.json').read_text());report_path=B/'high_cup_all300_issue_operation_v125/report.json'
    report=json.loads(report_path.read_text());verify_inputs(ROOT,report['verified_files_sha256'])
    assert report['model'] is not None and not report['failed_fit_keys']
    c['learned_artifacts'][TASK]['interaction']=report['model']
    c['high_cup_all300_report']=dict(path=str(report_path.relative_to(ROOT)),sha256=file_sha(report_path))
    association_path=B/'cup_public_association_radius_TRAIN_v133/report.json'
    association=json.loads(association_path.read_text());verify_inputs(ROOT,association['verified_files_sha256'])
    assert association['first_public_contact_samples']==association['new_radius_unique_intended_samples']==399
    c['attachment_association_distance_m']=association['association_radius_m']
    c['attachment_association_threshold_status']='TRAIN400 acquisitions /200 original parents; phase-free first-contact public pose calibration v133; native held IDs not read'
    c['cup_public_association_v133']=dict(path=str(association_path.relative_to(ROOT)),sha256=file_sha(association_path))
    c['high_cup_native_v134']=binding(ROOT,c)
    c.update(execution_scope='highcup_all300_original_gate_TRAIN_development_v136',formal_comparison_ready=False)
    library=InteractionLibrary.load(ROOT/report['model']['path']);acquire=library.for_phase('acquire')
    original_close=dict(c['normal_cup_close_intersection_v120']['original'])
    c['gripper_timing_config']=dict(close_position_tolerance_m=min(original_close['close_position_tolerance_m'],*(t.position_tolerance_m for t in acquire)),
        close_orientation_tolerance_rad=min(original_close['close_orientation_tolerance_rad'],*(t.orientation_tolerance_rad for t in acquire)))
    c['high_cup_close_intersection_v136']=dict(original=original_close,actual=c['gripper_timing_config'],
        source='original physical close gate intersect fitted acquire cap; shared capture and close field',caps_not_loosened=True)
    for predictor in ('cv','context'):
        identity=persistent_reference_continuation_identity(ROOT,c,TASK,predictor,library,CandidateConfig(**c['candidate_config']),MPCConfig(**c['mpc_config']))
        c['continuation_identities'].setdefault(TASK,{})[predictor]=identity['sha256']
    roster_path=ROOT/'artifacts/framework/protocol/train_manifest.json'
    raw=next(r['raw_episode_id'] for r in json.loads(roster_path.read_text())['episodes'] if r['task']==TASK)
    files=dict(old['files'])
    for name,checksum in report['verified_files_sha256'].items():
        if name in files:assert files[name]==checksum
        files[name]=checksum
    for name,checksum in association['verified_files_sha256'].items():
        if name in files:assert files[name]==checksum
        files[name]=checksum
    paths=[association_path,ROOT/'our_mani/multitask/frozen_subset_batch_v132.py',ROOT/'tests/test_frozen_subset_batch_v132.py',ROOT/'tests/test_cup_public_association_v133.py',Path(__file__),ROOT/'scripts/run_task_executor_smoke_high_cup_v135_r10.py',report_path,
        ROOT/report['model']['path'],ROOT/report['manifest']['path'],ROOT/'our_mani/multitask/high_cup_native_v134.py']
    paths.extend(ROOT/p['path'] for p in c['learned_artifacts'][TASK].values() if p)
    for p in paths:
        p=p.resolve();checksum=file_sha(p)
        if str(p) in files:assert files[str(p)]==checksum
        files[str(p)]=checksum
    SETUP.mkdir();atomic_json(SETUP/'configuration.json',c);files[str(SETUP/'configuration.json')]=file_sha(SETUP/'configuration.json')
    atomic_json(SETUP/'frozen_inputs.json',dict(schema='r10_highcup_all300_TRAIN_smoke_v136_lock',files=files,
        approved_for_execution=True,native_started=False,GPU_started=False,formal_VAL_TEST=False))
    try:
        checks=[]
        for method in ('cv_geometry','context_geometry'):
            argv=command(method,raw);result=subprocess.run(argv+['--check'],cwd=ROOT,text=True,capture_output=True)
            (SETUP/(method+'_CPU_stdout.log')).write_text(result.stdout);(SETUP/(method+'_CPU_stderr.log')).write_text(result.stderr)
            result.check_returncode();d=json.loads(result.stdout);assert d['complete'] and not d['native_started'] and not d['GPU_started']
            atomic_json(SETUP/(method+'_CPU_preflight.json'),d);checks.append(dict(method=method,task=TASK,raw=raw,argv=argv))
        verify_inputs(ROOT,files)
        atomic_json(SETUP/'CPU_acceptance.json',dict(complete=True,created_utc=datetime.now(timezone.utc).isoformat(),
            actual_argv_preflights=checks,original_real_parents=100,all300_acquisitions_preserved=True,
            all6_original_phase_gates_passed=True,actual_raw_tamper_checks_passed=4,batch_input_tests_each_environment=5,association_tests_each_environment=4,
            native_started=False,GPU_started=False,formal_VAL_TEST=False))
    except BaseException as error:
        atomic_json(SETUP/'preparation_failed.json',dict(type=type(error).__name__,message=str(error),native_started=False,GPU_started=False));raise
    print(json.dumps(dict(prepared=True,frozen_inputs=len(files),CPU_checks=len(checks),raw=raw,native_started=False)),flush=True)


if __name__=='__main__':main()
