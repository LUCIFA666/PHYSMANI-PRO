"""Bind unchanged v22 actor plus current-public TRAIN-only capture diagnostics."""
from pathlib import Path
import json
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,verify_inputs
from our_mani.multitask.task_native_backend_v39 import diagnostic_binding
from scripts.run_task_executor_smoke_v39_r10 import parser,preflight
BASE=ROOT/'artifacts/framework/multitask_r10';SETUP=BASE/'buzz_capture_v39_setup';OUT=BASE/'buzz_capture_v39'


def main():
    if SETUP.exists():raise FileExistsError('preserve previous capture setup')
    old=json.loads((BASE/'task_executor_smoke_v22_setup/frozen_inputs.json').read_text());verify_inputs(ROOT,old['files'])
    c=json.loads((BASE/'task_executor_smoke_v22_setup/configuration.json').read_text());c['native_metric_diagnostics']=diagnostic_binding(ROOT,c['baseline_repo'])
    for launch in ([sys.executable],['bash','scripts/run_local_sim.sh']):
        subprocess.run(launch+['-m','unittest','discover','-s','tests','-p','test_multitask_buzz_visual_capture.py'],cwd=ROOT,check=True)
    SETUP.mkdir();atomic_json(SETUP/'configuration.json',c);files=dict(old['files'])
    for p in (Path(__file__),ROOT/'scripts/run_task_executor_smoke_v39_r10.py',ROOT/'our_mani/multitask/task_native_backend_v39.py',
            ROOT/'our_mani/multitask/registration_metric_observer_v24.py',ROOT/'our_mani/multitask/buzz_capture_observer_v39.py',
            ROOT/'our_mani/multitask/buzz_visual_capture.py',SETUP/'configuration.json'):
        files[str(p.resolve())]=file_sha(p)
    atomic_json(SETUP/'frozen_inputs.json',dict(schema='r10_same_public_buffer_TRAIN_buzz_capture_v39_lock',files=files,
        approved_for_execution=True,native_started=False,policy_from_v22_unchanged=True,native_labels_metric_only=True,formal_release=False))
    requests=[]
    for task,raw in [('beat_the_rotating_buzz','episode10'),('beat_the_rotating_buzz_high_speed','episode102')]:
        argv=['bash','scripts/run_local_sim.sh','scripts/run_task_executor_smoke_v39_r10.py','--config',str((SETUP/'configuration.json').relative_to(ROOT)),
            '--manifest','artifacts/framework/protocol/train_manifest.json','--frozen-inputs',str((SETUP/'frozen_inputs.json').relative_to(ROOT)),
            '--task',task,'--raw-episode-id',raw,'--split','train','--method','cv_geometry','--replica','0','--development-demo',
            '--out',str((OUT/'development_demo_only'/task/raw).relative_to(ROOT))]
        preflight(parser().parse_args(argv[3:]));requests.append(dict(task=task,raw_episode_id=raw,argv=argv))
    atomic_json(SETUP/'plan.json',dict(requests=requests,fixed_TRAIN_sources=True,actor_inputs_unchanged=True,formal_metric=False,TEST_read=False))
    print(json.dumps(dict(prepared=True,frozen_inputs=len(files),tasks=2)),flush=True)


if __name__=='__main__':main()
