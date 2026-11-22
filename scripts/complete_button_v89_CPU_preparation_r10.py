"""Complete fresh CPU checks without changing the already frozen policy inputs."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from scripts.prepare_button_author_intent_v89_r10 import command,TASKS,SETUP,B


def main():
    if (SETUP/'CPU_acceptance.json').exists():raise FileExistsError('preserve completed acceptance')
    initial=SETUP/'preparation_failed.json'
    if not initial.exists():
        atomic_json(initial,dict(stage='CPU preflight push_moving_button context_geometry',
            type='CalledProcessError',returncode=1,original_stderr_not_retained=True,
            native_started=False,GPU_started=False,policy_inputs_not_modified=True))
    files=json.loads((SETUP/'frozen_inputs.json').read_text())['files'];verify_inputs(ROOT,files)
    tasks=json.loads((B/'all16_executor_joint_train_v1/plan.json').read_text())['tasks'];checks=[]
    for task in TASKS:
        raw=next(r['fixed_smoke_source'] for r in tasks if r['task']==task)
        for method in ('cv_geometry','context_geometry'):
            argv=command(task,raw,method);p=SETUP/(task+'_'+method+'_CPU_preflight.json')
            if not p.exists():
                result=subprocess.run(argv+['--check'],cwd=ROOT,text=True,capture_output=True)
                (SETUP/(task+'_'+method+'_supplemental_CPU_stdout.log')).write_text(result.stdout)
                (SETUP/(task+'_'+method+'_supplemental_CPU_stderr.log')).write_text(result.stderr)
                result.check_returncode();atomic_json(p,json.loads(result.stdout))
            d=json.loads(p.read_text());assert d['complete'] and d['check_only']
            assert not d['native_started'] and not d['GPU_started']
            assert d['identity']['frozen_inputs_sha256']==file_sha(SETUP/'frozen_inputs.json')
            checks.append(dict(task=task,method=method,raw=raw,argv=argv,
                receipt=dict(path=str(p.relative_to(ROOT)),sha256=file_sha(p))))
    verify_inputs(ROOT,files)
    atomic_json(SETUP/'CPU_acceptance.json',dict(complete=True,created_utc=datetime.now(timezone.utc).isoformat(),
        actual_argv_preflights=checks,reference_reserve_factory_tests_each_environment=6,
        original_actual_motion_limit_rad_s2=6.,reference_planning_limit_rad_s2=4.,
        frozen_inputs=len(files),supervision='author_command_intent',
        completed_phase_support_claimed=False,physical_press_completion_claimed=False,
        original_failure_preserved=dict(path=str(initial.relative_to(ROOT)),sha256=file_sha(initial)),
        original_failure_cause_not_asserted=True,all_frozen_sources_unchanged=True,
        native_started=False,GPU_started=False,formal_release=False,
        source=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=file_sha(__file__))))
    print(json.dumps(dict(complete=True,CPU_checks=len(checks),frozen_inputs=len(files),native_started=False)),flush=True)


if __name__=='__main__':main()
