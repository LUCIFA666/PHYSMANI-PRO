"""TRAIN/VAL then frozen four-way no-FASTER benchmark, two isolated simulators."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.benchmark_protocol import file_sha256,verify_lock,wilson

OUT=ROOT/'artifacts/framework/pipeline_v2'
VARIANTS=[(p,r) for p in ('cv','context') for r in ('geometry','value')]
RUN='comparison_r03'


def read(path):return json.loads(path.read_text()) if path.exists() else {}


def write(path,data):
    q=path.with_suffix('.tmp');q.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');q.replace(path)


def status(stage,**extra):
    write(OUT/'comparison_status.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'stage':stage,'FASTER_used':False,'variants':[f'{p}_{r}' for p,r in VARIANTS],**extra})


def run_variant(split,count,predictor,ranking):
    name=f'{RUN}_{split}_{predictor}_{ranking}'
    output=f'artifacts/framework/pipeline_v2/{name}'
    # A separately launched TRAIN check may already own this output. Wait for
    # it, then resume its completed episodes instead of racing two writers.
    while True:
        existing=False
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:argv=path.read_bytes().decode().split('\0')
            except (OSError,UnicodeDecodeError):continue
            if len(argv)>1 and argv[1].endswith('evaluate_interception_v2.py') and '--out' in argv:
                i=argv.index('--out')
                if i+1<len(argv) and argv[i+1]==output:existing=True;break
        if not existing:break
        time.sleep(5)
    command=['bash','scripts/run_local_sim.sh','scripts/evaluate_interception_v2.py',
        '--predictor',predictor,'--ranking',ranking,'--split',split,'--episodes-per-task',str(count),
        '--out',output]
    if split=='test' and predictor=='context' and ranking=='value':command+=['--videos','2']
    with (ROOT/f'logs/framework/{name}.log').open('a') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    return name


def freeze():
    path=OUT/(RUN+'_frozen_inputs.json')
    if path.exists():
        lock=read(path);verify_lock(ROOT,lock);return lock
    config=read(ROOT/'config/framework/pipeline_v2_intercept.json')
    files=[ROOT/'config/framework/pipeline_v2_intercept.json',ROOT/'scripts/evaluate_interception_v2.py',
        ROOT/'scripts/collect_local_rollouts.py',*sorted((ROOT/'our_mani').glob('*.py'))]
    for key in ('dynamics_checkpoint','context_local_value','cv_local_value','goal_calibration'):files.append(ROOT/config[key])
    for key in ('context_local_value','cv_local_value'):files.append((ROOT/config[key]).with_name('value_support.json'))
    for split in ('train','validation'):files.append(ROOT/f'artifacts/framework/protocol/{split}_manifest.json')
    files.append(ROOT/'artifacts/framework/pipeline_v1/official_test_manifest.json')
    lock={'created_utc':datetime.now(timezone.utc).isoformat(),'FASTER_used':False,'config':config,
        'files':{str(p.relative_to(ROOT)):file_sha256(p) for p in files},
        'training_selection':'reuses TRAIN-selected dynamics and controller-specific TRAIN-fit/calibrated Value; v2 parameters changed only with TRAIN checks',
        'test_status':'same public benchmark previously evaluated by v1; repeated benchmark, no held-out-new-test claim'}
    write(path,lock);return lock


def phase(split,count,lock):
    verify_lock(ROOT,lock)
    status(f'{split}_four_way_running',formal_test_used=split=='test',completed_variants=[])
    completed=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(run_variant,split,count,p,r):(p,r) for p,r in VARIANTS}
        for future in as_completed(futures):
            completed.append(future.result())
            status(f'{split}_four_way_running',formal_test_used=split=='test',completed_variants=completed)


def summarize():
    rows={}
    paper=read(ROOT/'config/framework/paper_reference_comparison.json')
    for predictor,ranking in VARIANTS:
        report=read(OUT/f'{RUN}_test_{predictor}_{ranking}/report.json')
        for task,result in report['per_task'].items():
            result['success_rate']=result['successes']/result['completed']
            result['wilson_95']=wilson(result['successes'],result['completed'])
        rows[f'{predictor}_{ranking}']=report['per_task']
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'FASTER_used':False,'complete':True,
        'variants':rows,'paper_reported_PhysMani':paper['task_success_reference']['methods']['PhysMani'],
        'paper_url':paper['paper'],'paper_aggregation':paper['task_success_reference']['aggregation'],
        'ours_aggregation':'one frozen model per variant, same 80 official resets, synchronous nominal 10s budget',
        'paired_comparisons_possible_only_between_ours_variants':True,'no_paired_test_against_paper_aggregate':True,
        'test_scope':'public benchmark previously seen by v1; v2 tuning restricted to TRAIN/VAL'}
    write(OUT/'comparison_report.json',report)
    lines=['无FASTER截获框架：四组对比','任务成功率；各组80条官方TEST（普通40、高速40）。']
    for variant,tasks in rows.items():
        for task,r in tasks.items():lines.append(f'{variant} / {task}: 成功{r["successes"]}/{r["completed"]}，接管{r["handovers"]}/{r["completed"]}。')
    lines+=['PhysMani基线来自论文Table1，未本地复现；论文标准差与单模型二项置信区间含义不同。',
        '局部Value是接管后的条件概率，接近阶段采用名义可达性硬筛；当前不声明整链路概率校准。']
    (ROOT/'无FASTER四组对比.txt').write_text('\n'.join(lines)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        while True:
            if (ROOT/'runtime/framework/pipeline_v2_STOP').exists():raise InterruptedError('v2 STOP')
            value=read(OUT/'value_training_status.json')
            if value.get('stage')=='failed':raise RuntimeError(f'CV Value training failed: {value}')
            if value.get('stage')=='ready':break
            status('waiting_for_controller_matched_CV_Value',formal_test_used=False)
            time.sleep(10)
        lock=freeze()
        phase('train',4,lock)
        phase('validation',20,lock)
        phase('test',0,lock)
        summarize();status('complete',formal_test_used=True,report='artifacts/framework/pipeline_v2/comparison_report.json')
        subprocess.run([sys.executable,'scripts/overnight_report.py'],cwd=ROOT,check=True)
    except Exception as error:
        status('failed',error=repr(error));raise


if __name__=='__main__':main()
