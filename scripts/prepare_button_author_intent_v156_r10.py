"""Freeze a separate TRAIN-only original-author-command action experiment."""
from pathlib import Path
import json,sys,subprocess
from copy import deepcopy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.button_intent_actor_v91 import binding,SOURCES
from our_mani.multitask.task_actor_backend_v69 import persistent_reference_policy_overlay
from scripts.run_task_executor_smoke_button_v153_r10 import parser,preflight
B=ROOT/'artifacts/framework/multitask_r10';SETUP=B/'button_author_intent_v156_setup';OUT=B/'button_author_intent_v156'
TASKS=('push_moving_button','push_moving_button_high_speed')


def command(task,raw,method):return ['bash','scripts/run_local_sim.sh','scripts/run_task_executor_smoke_button_v153_r10.py',
    '--config',str((SETUP/'configuration.json').relative_to(ROOT)),'--manifest','artifacts/framework/protocol/train_manifest.json',
    '--frozen-inputs',str((SETUP/'frozen_inputs.json').relative_to(ROOT)),'--task',task,'--raw-episode-id',raw,'--split','train',
    '--method',method,'--replica','0','--development-demo','--out',str((OUT/'development_demo_only'/task/raw/method).relative_to(ROOT))]


def main():
    if SETUP.exists():raise RuntimeError('preserve prior frozen or failed preparation')
    previous=B/'button_author_intent_v145_setup';lock=json.loads((previous/'frozen_inputs.json').read_text());verify_inputs(ROOT,lock['files'])
    for launcher in ([sys.executable],['bash','scripts/run_local_sim.sh']):
        for test in ('test_button_action_band_v152.py','test_button_segment_forecast_v155.py'):
            subprocess.run(launcher+['-m','unittest','discover','-s','tests','-p',test],cwd=ROOT,check=True)
    from our_mani.multitask.button_action_band_v152 import binding as guard_binding
    c=json.loads((previous/'configuration.json').read_text());c.pop('button_approach_capture_v141',None);c['button_approach_capture_v152']=guard_binding(ROOT);report_path=B/'button_author_action_intents_v70/report.json';report=json.loads(report_path.read_text())
    assert report['all_action_error_caps_passed'] and report['all200_original_parents_retained']
    verify_inputs(ROOT,report['original_inputs_sha256'])
    settings=c['policy_overlay']['configurations'];settings['causal_retarget']['maximum_reissues_per_phase']=16
    c['policy_overlay']=persistent_reference_policy_overlay(ROOT,settings['feedback'],settings['persistent_reference'],settings['causal_retarget'])
    for task in TASKS:
        c['learned_artifacts'][task]['interaction']=report['tasks'][task]['model']
        for predictor in ('cv','context'):c['continuation_identities'].setdefault(task,{})[predictor]=binding(ROOT,c,task,predictor)['sha256']
    c.update(execution_scope='TRAIN_only_button_action_band_segment_forecast_v156',formal_comparison_ready=False,
        button_supervision='original_author_commands; no completed_skill_or_public_press_completion_permission')
    # All potentially failing static dependencies are resolved before committing
    # the fresh setup directory; failed preflight never launches native work.
    paths=[ROOT/'our_mani/multitask/button_action_band_v152.py',ROOT/'tests/test_button_action_band_v152.py',ROOT/'our_mani/multitask/button_intent_prefix_v155.py',ROOT/'tests/test_button_segment_forecast_v155.py',Path(__file__),ROOT/'scripts/run_task_executor_smoke_button_v153_r10.py',ROOT/'our_mani/multitask/button_intent_native_v101.py',
        ROOT/'tests/test_button_success_metrics_v100.py',ROOT/'our_mani/multitask/button_success_metrics_v100.py',report_path]
    paths.extend((ROOT/'our_mani/multitask/input_stat_cache_v79.py',ROOT/'tests/test_input_stat_cache_v79.py',
        ROOT/'our_mani/multitask/frozen_input_index_v93.py',ROOT/'tests/test_frozen_input_index_v93.py'))
    paths.extend(ROOT/'our_mani/multitask'/name for name in SOURCES)
    for task in TASKS:
        for key in ('geometry','interaction','rigid_layout','colour'):
            pointer=c['learned_artifacts'][task].get(key)
            if pointer:
                path=ROOT/pointer['path'];assert file_sha(path)==pointer['sha256'];paths.append(path)
    for task in TASKS:paths.append(ROOT/report['tasks'][task]['model']['path'])
    paths.extend(ROOT/report[key]['path'] for key in ('raw_proofs','command_samples'))
    files=dict(lock['files'])
    for name,checksum in report['original_inputs_sha256'].items():
        p=Path(name);p=p if p.is_absolute() else ROOT/p
        assert file_sha(p)==checksum
        if str(p) in files:assert files[str(p)]==checksum
        files[str(p)]=checksum
    for p in paths:
        p=p.resolve();checksum=file_sha(p)
        if str(p) in files:assert files[str(p)]==checksum
        files[str(p)]=checksum
    SETUP.mkdir();atomic_json(SETUP/'configuration.json',c);files[str(SETUP/'configuration.json')]=file_sha(SETUP/'configuration.json')
    atomic_json(SETUP/'frozen_inputs.json',dict(schema='r10_TRAIN_button_action_band_segment_forecast_v156_lock',files=files,approved_for_execution=True,
        native_started=False,formal_VAL_TEST_released=False,actual_completed_skill_library_admitted=False))
    original=json.loads((B/'all16_executor_joint_train_v1/plan.json').read_text())['tasks']
    checks=[]
    for task in TASKS:
        raw=next(r['fixed_smoke_source'] for r in original if r['task']==task)
        for method in ('cv_geometry','context_geometry'):
            argv=command(task,raw,method)
            result=subprocess.run(argv+['--check'],cwd=ROOT,text=True,capture_output=True)
            if result.returncode:
                atomic_json(SETUP/'preparation_failed.json',dict(stage='CPU preflight',task=task,method=method,returncode=result.returncode,stderr=result.stderr,stdout=result.stdout,native_started=False,GPU_started=False))
            result.check_returncode()
            receipt=json.loads(result.stdout);atomic_json(SETUP/(task+'_'+method+'_CPU_preflight.json'),receipt)
            checks.append(dict(task=task,method=method,raw=raw,argv=argv))
    atomic_json(SETUP/'CPU_acceptance.json',dict(complete=True,actual_factory_tests_reused_from_v78_each_environment=3,strong_stat_cache_tests_reused_from_v80_each_environment=6,canonical_input_index_tests_reused_each_environment=10,native_success_metric_tests_each_environment=4,action_target_motion_band_tests_each_environment=4,segment_forecast_tests_each_environment=5,reference_reserve_factory_tests_each_environment=6,
        source_verifier="full SHA first/change read; expected SHA and dev/inode/size/mtime_ns/ctime_ns equality on reuse",
        actual_argv_preflights=checks,supervision='author_command_intent',completed_phase_support_claimed=False,
        physical_press_completion_claimed=False,formal_release=False,native_started=False))
    verify_inputs(ROOT,files);print(json.dumps(dict(prepared=True,frozen_inputs=len(files),tasks=list(TASKS),native_started=False,formal_release=False)),flush=True)

if __name__=='__main__':main()
