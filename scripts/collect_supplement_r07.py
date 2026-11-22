"""Preserve r06 TRAIN; collect every TRAIN source once and all VAL in author sensors."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.benchmark_protocol import file_sha256,verify_lock
BASE=ROOT/'artifacts/framework/aligned_r07'
TASKS=('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed')

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def status(stage,description,**extra):
    write(BASE/'status.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),stage=stage,
        description=description,formal_test_started=False,FASTER_used=False,NGFF_used=False,**extra))

def main():
    old=json.loads((ROOT/'artifacts/framework/aligned_r06/collection_inputs.json').read_text());verify_lock(ROOT,old)
    paths=[ROOT/'config/framework/aligned_protocol_r07.json',Path(__file__).resolve(),
        ROOT/'scripts/collect_candidate_success_r07.py',ROOT/'scripts/evaluate_aligned_r05_v2.py',
        ROOT/'our_mani/risk_v2/selector.py',ROOT/'our_mani/sensor_alignment/author_rgbd.py',
        ROOT/'artifacts/framework/aligned_r06/collection_inputs.json',
        ROOT/'artifacts/framework/aligned_r07/author_rgbd_probe/report.json']
    lock_path=BASE/'collection_inputs.json'
    if lock_path.exists():
        lock=json.loads(lock_path.read_text());verify_lock(ROOT,lock)
    else:
        lock=dict(created_utc=datetime.now(timezone.utc).isoformat(),
            files={str(p.relative_to(ROOT)):file_sha256(p) for p in paths},formal_TEST_used_for_fitting=False,
            selection_policy='geometry/softmax/uniform frozen r06 collector',
            preserved_training_domain='r06 OPENGL3 TRAIN only; no r06 VAL used for model fitting or r07 calibration',
            sensor_domain='author_OPENGL',expected_rollouts=640)
        write(lock_path,lock)
    def collect(predictor,split,task,shard,shards):
        verify_lock(ROOT,lock);replicas=1 if split=='train' else 3
        name=f'collection_{predictor}_{split}_{task}_s{shard}'
        args=['bash','scripts/run_local_sim.sh','scripts/collect_candidate_success_r07.py',
              '--predictor',predictor,'--split',split,'--task',task,'--shard',str(shard),
              '--shards',str(shards),'--replicas',str(replicas),'--out',f'artifacts/framework/aligned_r07/{name}']
        with (ROOT/'logs/framework'/f'aligned_r07_{name}.log').open('a') as log:
            subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        verify_lock(ROOT,lock);print(name,flush=True)
    try:
        status('collecting_author_sensor_supplement','保留旧TRAIN数据；补充每个TRAIN来源一次，并在作者OPENGL传感器下重新采集全部VAL三种探索。',expected_rollouts=640)
        jobs=[(p,s,t,i,n) for p in ('cv','context') for s,n in (('train',3),('validation',1)) for t in TASKS for i in range(n)]
        with ThreadPoolExecutor(max_workers=6) as pool:
            for future in as_completed([pool.submit(collect,*job) for job in jobs]):future.result();verify_lock(ROOT,old)
        status('author_sensor_data_complete','640次补充采集完成，待联合TRAIN拟合与新域VAL校准。',expected_rollouts=640)
    except Exception as exc:
        status('needs_attention',str(exc),error=repr(exc));raise

if __name__=='__main__':main()
