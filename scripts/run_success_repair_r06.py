"""Full TRAIN/VAL repair, deployment validation, then gated restart of comparison."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.benchmark_protocol import file_sha256,verify_lock
BASE=ROOT/'artifacts/framework/aligned_r06'
TASKS=('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed')

def write(p,j):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def status(stage,description,**extra):
    write(BASE/'status.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),stage=stage,description=description,
        formal_test_started=stage in ('official_TEST','complete'),FASTER_used=False,NGFF_used=False,blockers=[],**extra))

def run(args,name,gpu=False,sim=True):
    prefix=['bash','scripts/run_aligned_gpu.sh' if gpu else 'scripts/run_local_sim.sh'] if sim else [sys.executable]
    with (ROOT/'logs/framework'/('aligned_r06_'+name+'.log')).open('a') as f:
        subprocess.run(prefix+args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)

def freeze(name,values=False):
    p=BASE/name
    if p.exists():
        j=json.loads(p.read_text());verify_lock(ROOT,j);return j
    files=[ROOT/'config/framework/aligned_protocol_r06.json',ROOT/'scripts/run_success_repair_r06.py',
        ROOT/'scripts/collect_candidate_success_r06.py',ROOT/'scripts/train_candidate_success_r06.py',ROOT/'scripts/evaluate_aligned_r06.py',
        ROOT/'scripts/evaluate_physmani_aligned_r05_v3.py',ROOT/'scripts/summarize_aligned_r06.py',
        *sorted((ROOT/'our_mani/risk_v2').glob('*.py')),ROOT/'artifacts/framework/aligned_r05/frozen_inputs.json']
    if values:
        files += [BASE/f'{predictor}_success/{file}' for predictor in ('cv','context') for file in ('value.pt','value_support.json','report.json')]
    j=dict(created_utc=datetime.now(timezone.utc).isoformat(),files={str(f.relative_to(ROOT)):file_sha256(f) for f in files},
        formal_TEST_used_for_fitting=False,previous_public_TEST_exposure='r04 completed and r05 partial; no fresh-unseen-TEST claim')
    write(p,j);return j

def collect(predictor,split,task,shard,shards,lock):
    verify_lock(ROOT,lock)
    name=f'collection_{predictor}_{split}_{task}_s{shard}'
    run(['scripts/collect_candidate_success_r06.py','--predictor',predictor,'--split',split,'--task',task,
        '--shard',str(shard),'--shards',str(shards),'--replicas','3','--out',f'artifacts/framework/aligned_r06/{name}'],name)
    verify_lock(ROOT,lock);print(name,flush=True)

def ours(split,predictor,ranking,lock):
    verify_lock(ROOT,lock);name=f'{split}_{predictor}_{ranking}'
    args=['scripts/evaluate_aligned_r06.py','--split',split,'--predictor',predictor,'--ranking',ranking,
        '--episodes-per-task','0','--out',f'artifacts/framework/aligned_r06/{name}']
    if split=='test' and predictor=='context' and ranking=='value':args+=['--videos','2']
    run(args,name);verify_lock(ROOT,lock);print(name,flush=True)

def acceptance():
    results={};failures=[]
    for predictor in ('cv','context'):
        def rows(ranking):return [json.loads(p.read_text()) for p in (BASE/f'validation_{predictor}_{ranking}').glob('pick*__episode*.json') if not p.name.endswith('infrastructure_error.json')]
        geom,value=rows('geometry'),rows('value')
        if len(geom)!=40 or len(value)!=40:failures.append(predictor+' incomplete deployed VAL')
        report=json.loads((BASE/f'{predictor}_success/report.json').read_text())
        results[predictor]={}
        for task in TASKS:
            g=sum(r['official_task_success'] for r in geom if r['task']==task)
            v=sum(r['official_task_success'] for r in value if r['task']==task)
            cross=report['calibration_source_crossfit'][task]
            raw=report['calibration_source_raw'][task]
            results[predictor][task]=dict(geometry_successes=g,value_successes=v,episodes=20,
                crossfit_calibration_brier=cross['brier'],raw_brier=raw['brier'],crossfit_ece=cross['ece_5_bins'])
            if v<g-1:failures.append(f'{predictor} {task}: learned ranking regresses by more than one VAL episode')
            if cross['brier']>raw['brier']+.01:failures.append(f'{predictor} {task}: correction worsens crossfit Brier')
            if cross['ece_5_bins']>.15:failures.append(f'{predictor} {task}: crossfit calibration error > .15')
    j=dict(passed=not failures,formal_test_used=False,results=results,failures=failures,
        gate='all TRAIN executed; source-disjoint selection/calibration; crossfit calibration ECE<=.15; no >1/20 deployed VAL loss vs geometry',
        interpretation='development acceptance, not a statistical proof of improvement')
    write(BASE/'success_acceptance.json',j);return j

def baseline(split,task,lock):
    verify_lock(ROOT,lock)
    run(['scripts/evaluate_physmani_aligned_r05_v3.py','--config','config/framework/aligned_protocol_r06.json',
        '--split',split,'--task',task,'--episodes-per-task','0','--out',f'artifacts/framework/aligned_r06/{split}_physmani_v3'],split+'_physmani_'+task,gpu=True)
    verify_lock(ROOT,lock)

def main():
    try:
        original=json.loads((ROOT/'artifacts/framework/aligned_r05/frozen_inputs.json').read_text());verify_lock(ROOT,original)
        lock=freeze('collection_inputs.json')
        status('collecting_candidate_TRAIN_VAL','采集真实候选执行结果：每种动力学覆盖全部200 TRAIN、40 VAL来源，各3种选点探索；不使用TEST。',expected_rollouts=1440)
        jobs=[(p,s,t,i,n) for p in ('cv','context') for s,n in (('train',3),('validation',1)) for t in TASKS for i in range(n)]
        with ThreadPoolExecutor(max_workers=6) as pool:
            for future in as_completed([pool.submit(collect,*job,lock) for job in jobs]):future.result();verify_lock(ROOT,original)
        status('fitting_candidate_success','真实接近数据已采完；训练两套5成员小网络，并做任务条件仿射校准及来源交叉验证。')
        for predictor in ('cv','context'):
            if not (BASE/f'{predictor}_success/value.pt').exists():run(['scripts/train_candidate_success_r06.py','--predictor',predictor],predictor+'_fit',sim=False)
        lock=freeze('frozen_inputs.json',values=True)
        status('validating_repaired_success','新权重冻结；运行四组完整VAL，检查概率校准和实际选点收益，验收后才重启正式TEST。')
        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(ours,'validation',p,r,lock) for p in ('cv','context') for r in ('geometry','value')]):future.result()
        gate=acceptance()
        if not gate['passed']:
            status('success_repair_needs_attention','成功率模块尚未通过开发验收，正式TEST保持停止。',acceptance_failures=gate['failures'])
            run(['scripts/report_aligned_r06.py'],'report',sim=False);return
        status('baseline_validation','成功率修复通过开发验收；修正后的PhysMani完成完整VAL，然后重新开始共同TEST。')
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(baseline,'validation',task,lock) for task in TASKS]):future.result()
        status('official_TEST','成功率模块和基线执行验收完成；共同外部条件下重新运行400次TEST。')
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures=[pool.submit(ours,'test',p,r,lock) for p in ('cv','context') for r in ('geometry','value')]
            futures += [pool.submit(baseline,'test',task,lock) for task in TASKS]
            for future in as_completed(futures):future.result()
        run(['scripts/summarize_aligned_r06.py'],'final_report',sim=False)
        verify_lock(ROOT,original)
        status('complete','成功率模块修复、开发验收及400次对齐TEST完成。',report='artifacts/framework/aligned_r06/final_report.json')
        run(['scripts/report_aligned_r06.py'],'report',sim=False)
    except Exception as exc:
        status('needs_attention',str(exc),error=repr(exc));raise

if __name__=='__main__':main()
