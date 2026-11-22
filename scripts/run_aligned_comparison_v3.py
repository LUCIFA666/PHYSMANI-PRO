"""Resume frozen r05 policies with a source-faithful released baseline executor."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.benchmark_protocol import file_sha256,verify_lock
BASE=ROOT/'artifacts/framework/aligned_r05'
TASKS=('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed')

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def status(stage,description,**kw):
    write(BASE/'status.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),stage=stage,
        description=description,FASTER_used=False,NGFF_used=False,formal_test_started=kw.pop('formal_test_started',True),blockers=[],**kw))

def run(args,name,gpu=False,sim=True):
    prefix=['bash','scripts/run_aligned_gpu.sh' if gpu else 'scripts/run_local_sim.sh'] if sim else [sys.executable]
    with (ROOT/'logs/framework'/('aligned_r05_v3_'+name+'.log')).open('a') as log:
        subprocess.run(prefix+args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

def freeze():
    path=BASE/'comparison_v3_inputs.json'
    if path.exists():
        lock=json.loads(path.read_text());verify_lock(ROOT,lock);return lock
    files=['artifacts/framework/aligned_r05/frozen_inputs.json',
        'config/framework/aligned_protocol_r05_baseline_v3.json',
        'our_mani/ablations/physmani_executor_v3.py','scripts/evaluate_physmani_aligned_r05_v3.py',
        'scripts/probe_physmani_executor_v3.py','scripts/summarize_aligned_r05.py',
        'scripts/run_aligned_comparison_v3.py','artifacts/framework/aligned_r05/acceptance_tolerances.json',
        'artifacts/framework/aligned_r05/baseline_executor_v3_probe/report.json']
    lock=dict(created_utc=datetime.now(timezone.utc).isoformat(),files={p:file_sha256(ROOT/p) for p in files},
        original_r05_core_inputs_unchanged=True,
        baseline_revision='preserve author Mover ignore-collision mode and Cartesian waypoint modification; remove uncounted physics only',
        selection='source parity, not TEST success-rate tuning',frozen_before_formal_TEST=True)
    write(path,lock);return lock

def ours(predictor,ranking,lock,original):
    verify_lock(ROOT,lock);verify_lock(ROOT,original)
    name=f'test_{predictor}_{ranking}'
    args=['scripts/evaluate_aligned_r05_v2.py','--split','test','--predictor',predictor,'--ranking',ranking,
        '--episodes-per-task','0','--out',f'artifacts/framework/aligned_r05/{name}']
    if predictor=='context' and ranking=='value':args+=['--videos','2']
    run(args,name)
    verify_lock(ROOT,lock);verify_lock(ROOT,original)
    print(name,flush=True)

def baseline_split(split,lock):
    directory=f'artifacts/framework/aligned_r05/{split}_physmani_v3'
    def task(t):
        verify_lock(ROOT,lock)
        run(['scripts/evaluate_physmani_aligned_r05_v3.py','--split',split,'--episodes-per-task','0',
            '--task',t,'--out',directory],split+'_physmani_'+t,gpu=True)
        verify_lock(ROOT,lock)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(task,t) for t in TASKS]):future.result()
    records=[json.loads(p.read_text()) for p in (ROOT/directory).glob('pick*__episode*.json') if not p.name.endswith('infrastructure_error.json')]
    expected=80 if split=='test' else 40
    if len(records)!=expected:raise RuntimeError('baseline coverage incomplete')
    write(ROOT/directory/'report.json',dict(completed=len(records),expected=expected,complete=True,formal_test_used=split=='test',
        per_task={t:dict(completed=sum(r['task']==t for r in records),successes=sum(r['task']==t and r['official_task_success'] for r in records)) for t in TASKS}))
    print(split+'_physmani_v3',flush=True)

def baseline(lock):
    baseline_split('validation',lock)
    write(BASE/'baseline_v3_stage.json',dict(stage='official_TEST',validation_complete=True,formal_test_started=True))
    baseline_split('test',lock)

def main():
    try:
        original=json.loads((BASE/'frozen_inputs.json').read_text());verify_lock(ROOT,original)
        while True:
            probe=BASE/'baseline_executor_v3_probe/report.json'
            smoke=BASE/'smoke_physmani_v3/report.json'
            ready=probe.exists() and json.loads(probe.read_text()).get('passed') and smoke.exists() and json.loads(smoke.read_text()).get('complete')
            if ready:break
            if probe.exists() and not json.loads(probe.read_text()).get('passed'):raise RuntimeError('baseline path parity failed')
            status('baseline_execution_acceptance','基线路径处理修正完成；等待TRAIN动作执行与作者运动学一致性验收，四组VAL已完成。',formal_test_started=False)
            time.sleep(5)
        lock=freeze()
        status('frozen_official_TEST','四组冻结policy正式TEST运行；修正后的PhysMani先完成VAL，再运行同一80个TEST案例。')
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures=[pool.submit(ours,p,r,lock,original) for p in ('cv','context') for r in ('geometry','value')]
            futures.append(pool.submit(baseline,lock))
            for future in as_completed(futures):future.result()
        run(['scripts/summarize_aligned_r05.py'],'final_report',sim=False)
        verify_lock(ROOT,original);verify_lock(ROOT,lock)
        status('main_comparison_complete','400次完整policy正式TEST及共同条件验收完成；反馈频率、地图来源消融继续单列。',report='artifacts/framework/aligned_r05/final_report.json')
        run(['scripts/report_aligned_r05.py'],'final_report',sim=False)
    except Exception as exc:
        status('needs_attention',str(exc),error=repr(exc),blockers_detail=[str(exc)])
        raise

if __name__=='__main__':main()
