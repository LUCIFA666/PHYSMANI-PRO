"""Full r05: collect all TRAIN/VAL, refit Values, freeze, evaluate paired TEST.

Workers are independent native simulator processes, one task per process.
The external PhysMani checkout is read-only. Original r04 inputs stay frozen.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.benchmark_protocol import file_sha256,verify_lock
BASE=ROOT/'artifacts/framework/aligned_r05'
CONFIG='config/framework/aligned_protocol_r05.json'
TASKS=('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed')

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def status(stage,description,**kwargs):
    write(BASE/'status.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),stage=stage,
        description=description,FASTER_used=False,NGFF_used=False,formal_test_started=False,blockers=[],**kwargs))

def run(arguments,logname,*,sim=False,gpu=False):
    prefix=['bash','scripts/run_aligned_gpu.sh' if gpu else 'scripts/run_local_sim.sh'] if sim or gpu else [sys.executable]
    with (ROOT/'logs/framework'/logname).open('a') as log:
        subprocess.run(prefix+arguments,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def collection(predictor,split,task,shard,shards):
    config=json.loads((ROOT/CONFIG).read_text())
    name=f'collection_{predictor}_{split}_{task}_shard{shard}'
    args=['scripts/collect_aligned_r05.py','--predictor',predictor,'--split',split,'--task',task,
          '--shard-index',str(shard),'--shards',str(shards),'--out',f'artifacts/framework/aligned_r05/{name}']
    if predictor=='context':args+=['--latent-checkpoint',config['dynamics_checkpoint']]
    run(args,'aligned_r05_'+name+'.log',sim=True)
    return name

def merge(predictor,split):
    rows=json.loads((ROOT/f'artifacts/framework/protocol/{split}_manifest.json').read_text())['episodes']
    expected={(r['task'],r['raw_episode_id']) for r in rows if r['task'] in TASKS}
    shards=3 if split=='train' else 1
    records=[];keys=set();sources=[]
    out=BASE/f'{predictor}_{split}';out.mkdir(parents=True,exist_ok=True)
    for task in TASKS:
        for shard in range(shards):
            source=BASE/f'collection_{predictor}_{split}_{task}_shard{shard}'
            report=json.loads((source/'report.json').read_text())
            if report['completed']!=report['expected']:raise ValueError(f'incomplete shard {source}')
            sources.append(str(source.relative_to(ROOT)))
            for path in sorted(source.glob('pick*__*__*.json')):
                record=json.loads(path.read_text())
                if record['key'] in keys:raise ValueError('duplicate training attempt')
                if record['split']!=split or record['formal_test_used']:raise ValueError('forbidden training split')
                keys.add(record['key']);records.append(record)
                for p in (path,path.with_suffix('.npz')):
                    target=out/p.name
                    if target.exists() and file_sha256(target)!=file_sha256(p):raise ValueError('changed merged data')
                    if not target.exists():shutil.copy2(p,target)
    valid=[r for r in records if r.get('initial_features') is not None and r.get('failure')!='deadline_stop']
    groups={(r['task'],r['source_episode']) for r in valid}
    if len(records)!=4*len(expected) or groups!=expected:
        write(out/'coverage_failure.json',dict(attempts=len(records),expected_attempts=4*len(expected),missing_valid_sources=sorted(expected-groups)))
        raise ValueError('full TRAIN/VAL coverage required; inspect coverage failure')
    controllers={r['controller_sha256'] for r in records}
    if len(controllers)!=1:raise ValueError('mixed actuator/controller versions')
    write(out/'report.json',dict(completed=len(records),expected=len(expected)*4,valid_initializations=len(valid),
        source_episodes=len(expected),all_source_episodes_have_valid_outcomes=True,formal_test_used=False,
        full_task_executed=True,controller_sha256=next(iter(controllers)),split=split,shards=sources,
        per_task={t:dict(source_episodes=sum(task==t for task,_ in groups),valid=sum(r['task']==t for r in valid),
            successes=sum(r['task']==t and r['official_task_success_immediate'] for r in valid)) for t in TASKS}))
    return str(out.relative_to(ROOT))

def fit(predictor):
    train=merge(predictor,'train');val=merge(predictor,'validation')
    out=f'artifacts/framework/aligned_r05/{predictor}_success'
    if not (ROOT/out/'value.pt').exists():
        run(['scripts/train_success_predictor_full.py','--data',train,'--validation',val,'--out',out,
             '--outcome','official_task_success_immediate','--context-dim','0','--updates','2000'],
            f'aligned_r05_{predictor}_success_training.log')
    return predictor

def inputs(path,include_values=False):
    if path.exists():
        lock=json.loads(path.read_text());verify_lock(ROOT,lock);return lock
    config=json.loads((ROOT/CONFIG).read_text())
    files=[ROOT/CONFIG,*sorted((ROOT/'our_mani').glob('*.py')),*sorted((ROOT/'our_mani/aligned').glob('*.py')),
           ROOT/'scripts/collect_aligned_r05.py',ROOT/'scripts/collect_local_rollouts.py',
           ROOT/'scripts/evaluate_aligned_r05.py',ROOT/'scripts/evaluate_physmani_aligned_r05.py',
           ROOT/'scripts/run_local_sim.sh',ROOT/'scripts/run_aligned_gpu.sh',
           ROOT/'scripts/train_success_predictor_full.py',Path(__file__).resolve(),ROOT/config['dynamics_checkpoint'],ROOT/config['goal_calibration']]
    for split in ('train','validation'):files.append(ROOT/f'artifacts/framework/protocol/{split}_manifest.json')
    files.append(ROOT/'artifacts/framework/pipeline_v1/official_test_manifest.json')
    if include_values:
        for key in ('cv_local_value','context_local_value'):
            files += [ROOT/config[key],(ROOT/config[key]).with_name('value_support.json'),(ROOT/config[key]).with_name('report.json')]
    lock=dict(created_utc=datetime.now(timezone.utc).isoformat(),files={str(p.relative_to(ROOT)):file_sha256(p) for p in files},
        config=config,baseline_checkpoint_sha256=file_sha256(Path(config['baseline_checkpoint'])),
        original_r04_unchanged=True,information_setting=config['information_setting'],
        baseline='released epoch99999 fixed before TEST; shared-map/counting adapter; not paper Table1 reproduction',
        test_previous_exposure='public TEST already evaluated by older versions; no fresh-unseen-test claim')
    write(path,lock);return lock

def ours(split,predictor,ranking,lock):
    verify_lock(ROOT,lock)
    name=f'{split}_{predictor}_{ranking}'
    args=['scripts/evaluate_aligned_r05.py','--split',split,'--predictor',predictor,'--ranking',ranking,
          '--episodes-per-task','0','--out',f'artifacts/framework/aligned_r05/{name}']
    if split=='test' and predictor=='context' and ranking=='value':args+=['--videos','2']
    run(args,'aligned_r05_'+name+'.log',sim=True)
    verify_lock(ROOT,lock)
    return name

def baseline(split,lock):
    verify_lock(ROOT,lock)
    run(['scripts/evaluate_physmani_aligned_r05.py','--split',split,'--episodes-per-task','0',
         '--out',f'artifacts/framework/aligned_r05/{split}_physmani'],
        'aligned_r05_'+split+'_physmani.log',gpu=True)
    verify_lock(ROOT,lock)
    return split+'_physmani'

def main():
    try:
        r04=json.loads((ROOT/'artifacts/framework/pipeline_v2/comparison_r04_frozen_inputs.json').read_text());verify_lock(ROOT,r04)
        for name in ('smoke_protocol_v2','dynamics_validation'):
            report=json.loads((BASE/name/'report.json').read_text())
            if not (report.get('passed') or report.get('complete')):raise ValueError(f'acceptance gate incomplete: {name}')
        collection_lock=inputs(BASE/'collection_inputs.json')
        status('collecting_full_TRAIN_VAL','在共同环境下重采两种控制器的局部成功率数据；每套200 TRAIN来源、40 VAL来源，每来源4次尝试。',train_attempts_per_predictor=800,validation_attempts_per_predictor=160)
        jobs=[(predictor,split,task,shard,shards) for predictor in ('cv','context') for split,shards in (('train',3),('validation',1)) for task in TASKS for shard in range(shards)]
        with ThreadPoolExecutor(max_workers=6) as pool:
            for future in as_completed([pool.submit(collection,*job) for job in jobs]):
                print(future.result(),flush=True)
                verify_lock(ROOT,collection_lock)
        status('training_success_predictors','完整TRAIN覆盖已验收；分别训练CV/Context成功率头，VAL分组独立用于选模与校准。')
        for predictor in ('cv','context'):fit(predictor)
        lock=inputs(BASE/'frozen_inputs.json',include_values=True)
        status('frozen_validation','权重与输入已冻结；开始四组完整VAL和本地PhysMani VAL。')
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures=[pool.submit(ours,'validation',p,r,lock) for p in ('cv','context') for r in ('geometry','value')]
            futures.append(pool.submit(baseline,'validation',lock))
            for future in as_completed(futures):print(future.result(),flush=True)
        status('frozen_official_TEST','完整VAL运行结束；保持冻结配置，在同一80个官方TEST案例上运行四种组合及本地PhysMani。')
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures=[pool.submit(ours,'test',p,r,lock) for p in ('cv','context') for r in ('geometry','value')]
            futures.append(pool.submit(baseline,'test',lock))
            for future in as_completed(futures):print(future.result(),flush=True)
        run(['scripts/summarize_aligned_r05.py'],'aligned_r05_final_report.log')
        verify_lock(ROOT,r04)
        status('complete','r05完整训练、验证、400次正式TEST及条件验收全部完成。',report='artifacts/framework/aligned_r05/final_report.json')
        run(['scripts/report_aligned_r05.py'],'aligned_r05_final_report.log')
        subprocess.run(['systemctl','--user','stop','our-mani-r05-report.timer'],check=True)
    except Exception as exc:
        status('needs_attention',str(exc),error=repr(exc),blockers_detail=[str(exc)])
        raise

if __name__=='__main__':main()
