"""Freeze reviewed PhysMani runtime for fixed first TRAIN source of all14 R10 tasks."""
from pathlib import Path
import json
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha,verify_inputs
from our_mani.multitask.specs import TASK_SPECS
BASE=ROOT/'artifacts/framework/multitask_r10'
SETUP=BASE/'physmani_task_smoke_v40_setup'
OUT=BASE/'physmani_task_smoke_v40'


def main():
    if SETUP.exists():raise FileExistsError('preserve previous baseline preparation')
    old=json.loads((BASE/'PhysMani_early_VAL_approved_v12/frozen_inputs.json').read_text());verify_inputs(ROOT,old['files'])
    geometry=json.loads((BASE/'task_executor_smoke_v22_setup/frozen_inputs.json').read_text());verify_inputs(ROOT,geometry['files'])
    files=dict(old['files'])
    for name,sha in geometry['files'].items():
        if name in files and files[name]!=sha:raise ValueError('previous frozen inputs disagree')
        files[name]=sha
    worker=ROOT/'scripts/run_task_executor_smoke_physmani_v40_r10.py'
    for p in (Path(__file__),worker,ROOT/'scripts/run_aligned_gpu.sh'):files[str(p.resolve())]=file_sha(p)
    config=BASE/'task_executor_smoke_v22_setup/configuration.json'
    manifest=ROOT/'artifacts/framework/protocol/train_manifest.json';source=json.loads(manifest.read_text())['episodes'];requests=[]
    for task in TASK_SPECS:
        row=next(r for r in source if r['task']==task)
        argv=['bash','scripts/run_aligned_gpu.sh',str(worker.relative_to(ROOT)),'--config',str(config.relative_to(ROOT)),
            '--manifest',str(manifest.relative_to(ROOT)),'--frozen-inputs',str((SETUP/'frozen_inputs.json').relative_to(ROOT)),
            '--task',task,'--raw-episode-id',row['raw_episode_id'],'--split','train','--method','physmani','--replica','0',
            '--development-demo','--out',str((OUT/'development_demo_only'/task/row['raw_episode_id']).relative_to(ROOT))]
        requests.append(dict(task=task,raw_episode_id=row['raw_episode_id'],argv=argv))
    SETUP.mkdir();atomic_json(SETUP/'frozen_inputs.json',dict(schema='r10_PhysMani_reviewed_runtime_fixed_TRAIN_task_smoke_v40',
        files=files,approved_for_execution=True,formal_VAL_TEST_ready=False,native_started=False,GPU_started=False))
    atomic_json(SETUP/'plan.json',dict(tasks=14,requests=requests,selection='first original TRAIN source, outcome independent',
        GPU_slots=1,native_slots_per_case=1,automatic_retry=False,formal_metric=False,TEST_read=False,
        baseline_training_started=False,source_bound_to_reviewed_v12_runtime=True))
    check=subprocess.run(requests[0]['argv']+['--check'],cwd=ROOT,text=True,capture_output=True,check=True)
    atomic_json(SETUP/'native_CPU_preflight.json',json.loads(check.stdout));verify_inputs(ROOT,files)
    print(json.dumps(dict(prepared=True,tasks=14,frozen_inputs=len(files),native_started=False,GPU_started=False)),flush=True)


if __name__=='__main__':main()
