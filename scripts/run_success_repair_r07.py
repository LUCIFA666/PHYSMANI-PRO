"""Complete the author-sensor supplement, fit critics, gate VAL, restart aligned TEST."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.benchmark_protocol import file_sha256,verify_lock
BASE=ROOT/'artifacts/framework/aligned_r07'
ORCHESTRATOR_SHA=file_sha256(Path(__file__))
TASKS=('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed')

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def status(stage,description,**extra):
    prior=json.loads((BASE/'status.json').read_text()) if (BASE/'status.json').exists() else {}
    exposed=prior.get('formal_test_started',False) or stage in ('official_TEST','complete') or any(BASE.glob('test_*/pick*__episode*.json'))
    write(BASE/'status.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),stage=stage,
        description=description,formal_test_started=bool(exposed),FASTER_used=False,NGFF_used=False,**extra))

def run(args,name,gpu=False,sim=True):
    if (ROOT/'runtime/framework/success_r07_STOP').exists():raise InterruptedError('r07 STOP')
    prefix=['bash','scripts/run_aligned_gpu.sh' if gpu else 'scripts/run_local_sim.sh'] if sim else [sys.executable]
    environment=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (ROOT/'logs/framework'/('aligned_r07_'+name+'.log')).open('a') as log:
        subprocess.run(prefix+args,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)

def freeze(name,values=False,acceptance=False):
    if file_sha256(Path(__file__))!=ORCHESTRATOR_SHA:raise RuntimeError('orchestrator source changed since process start; restart before freeze')
    path=BASE/name
    if path.exists():
        lock=json.loads(path.read_text());verify_lock(ROOT,lock);return lock
    paths=[Path(__file__),ROOT/'config/framework/aligned_protocol_r07.json',
        ROOT/'scripts/collect_candidate_success_r07.py',ROOT/'scripts/train_candidate_success_r07.py',
        ROOT/'scripts/evaluate_aligned_r07.py',ROOT/'scripts/evaluate_physmani_aligned_r07_v6.py',
        ROOT/'scripts/summarize_aligned_r07_v6.py',ROOT/'config/framework/aligned_protocol_r07_baseline_v6.json',ROOT/'our_mani/ablations/physmani_executor_v4.py',
        *sorted((ROOT/'our_mani/risk_v3').glob('*.py')),*sorted((ROOT/'our_mani/sensor_alignment').glob('*.py')),
        ROOT/'artifacts/framework/aligned_r06/frozen_inputs.json',BASE/'collection_inputs.json',
        BASE/'acceptance_tolerances.json',BASE/'author_rgbd_probe/report.json',
        ROOT/'artifacts/framework/pipeline_v1/official_test_manifest.json',
        ROOT/'artifacts/framework/protocol/train_manifest.json',ROOT/'artifacts/framework/protocol/validation_manifest.json']
    config=json.loads((ROOT/'config/framework/aligned_protocol_r07.json').read_text())
    paths += [ROOT/config['dynamics_checkpoint'],ROOT/config['goal_calibration'],Path(config['baseline_checkpoint'])]
    if values:paths += [BASE/f'{p}_success/{f}' for p in ('cv','context') for f in ('value.pt','value_support.json','report.json')]
    if acceptance:
        snapshot=BASE/'pretest_data_quality_audit.json'
        if not snapshot.exists():snapshot.write_bytes((BASE/'data_quality_audit.json').read_bytes())
        paths += [BASE/'success_acceptance.json',snapshot,BASE/'execution_protocol_review.json',BASE/'execution_sensitivity_probe/report.json',BASE/'baseline_executor_v6_smoke/report.json']
    lock=dict(created_utc=datetime.now(timezone.utc).isoformat(),files={os.path.relpath(p,ROOT):file_sha256(p) for p in paths},
        formal_TEST_used_for_fitting=False,previous_public_TEST_exposure='r04 complete and r05 partial; no fresh-unseen TEST claim')
    write(path,lock);return lock

def ours(split,predictor,ranking,lock):
    verify_lock(ROOT,lock);name=f'{split}_{predictor}_{ranking}'
    args=['scripts/evaluate_aligned_r07.py','--split',split,'--predictor',predictor,'--ranking',ranking,
        '--episodes-per-task','0','--out',f'artifacts/framework/aligned_r07/{name}']
    if split=='test' and predictor=='context' and ranking=='value':args+=['--videos','2']
    run(args,name);verify_lock(ROOT,lock);print(name,flush=True)

def baseline(split,task,lock):
    verify_lock(ROOT,lock)
    run(['scripts/evaluate_physmani_aligned_r07_v6.py','--config','config/framework/aligned_protocol_r07_baseline_v6.json',
        '--split',split,'--task',task,'--episodes-per-task','0','--out',f'artifacts/framework/aligned_r07/{split}_physmani_v6'],split+'_physmani_'+task,gpu=True)
    verify_lock(ROOT,lock)

def acceptance():
    results={};failures=[]
    for predictor in ('cv','context'):
        def rows(ranking):return [json.loads(p.read_text()) for p in (BASE/f'validation_{predictor}_{ranking}').glob('pick*__episode*.json') if not p.name.endswith('infrastructure_error.json')]
        geom,value=rows('geometry'),rows('value')
        if len(geom)!=40 or len(value)!=40:failures.append(predictor+' incomplete deployed VAL')
        report=json.loads((BASE/f'{predictor}_success/report.json').read_text());results[predictor]={}
        for task in TASKS:
            g=sum(r['official_task_success'] for r in geom if r['task']==task)
            v=sum(r['official_task_success'] for r in value if r['task']==task)
            detail=report['calibration_crossfit'][task];cross=detail['metrics'];raw=detail['raw_metrics']
            coverage=report['source_coverage'][task]
            results[predictor][task]=dict(geometry_successes=g,value_successes=v,episodes=20,
                calibration_crossfit=detail,TRAIN_coverage=coverage,
                learned_ranking_enabled=report['metadata']['per_task'][task]['ranking_enabled'])
            if coverage['TRAIN_executed_sources']!=100 or not coverage['all_observed_TRAIN_sources_have_positive_ensemble_weights']:
                failures.append(f'{predictor} {task}: incomplete TRAIN execution/ensemble coverage')
            if v<g-1:failures.append(f'{predictor} {task}: learned selection loses more than one VAL episode')
            if not detail['valid']:failures.append(f'{predictor} {task}: insufficient valid calibration crossfit')
            else:
                if cross['brier']>raw['brier']+.01:failures.append(f'{predictor} {task}: correction worsens source-crossfit Brier')
                if cross['ece_5_bins']>.15:failures.append(f'{predictor} {task}: source-crossfit ECE > .15')
    value=dict(passed=not failures,formal_test_used=False,results=results,failures=failures,
        gate='full TRAIN execution; TRAIN-only grouped model selection; valid source-crossfit calibration ECE<=.15 and Brier<=raw+.01; deployed VAL loss<=1/20',
        interpretation='development acceptance, not proof of ranking gain; per-task TRAIN ranking guard can deliberately fall back to geometry')
    write(BASE/'success_acceptance.json',value);return value

def main():
    try:
        original=json.loads((ROOT/'artifacts/framework/aligned_r06/frozen_inputs.json').read_text());verify_lock(ROOT,original)
        # A separate immutable supplement collector is already running; never launch duplicates.
        while True:
            state=json.loads((BASE/'status.json').read_text())
            if state['stage']=='author_sensor_data_complete':break
            if state['stage']=='needs_attention':raise RuntimeError(state['description'])
            if (ROOT/'runtime/framework/success_r07_STOP').exists():raise InterruptedError('r07 STOP')
            time.sleep(5)
        run(['scripts/audit_candidate_data_r07.py'],'data_audit',sim=False)
        source_lock=freeze('training_inputs.json')
        status('fitting_task_success','全部补充数据完成；联合旧TRAIN与新TRAIN进行分任务5折来源交叉验证，VAL只做概率校准。')
        for predictor in ('cv','context'):
            verify_lock(ROOT,source_lock)
            if not (BASE/f'{predictor}_success/report.json').exists():run(['scripts/train_candidate_success_r07.py','--predictor',predictor],predictor+'_fit')
        lock=freeze('frozen_inputs.json',values=True)
        status('validating_repaired_success','权重与选择器已冻结；四组完整VAL验证校准和保守选点，未通过前不运行TEST。')
        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(ours,'validation',p,r,lock) for p in ('cv','context') for r in ('geometry','value')]):future.result()
        gate=acceptance()
        if not gate['passed']:
            status('success_repair_needs_attention','开发验收仍未通过，正式TEST保持停止。',acceptance_failures=gate['failures'])
            run(['scripts/report_aligned_r07.py'],'report',sim=False);return
        review_path=BASE/'execution_protocol_review.json'
        if not review_path.exists() or not json.loads(review_path.read_text()).get('approved_for_comparison',False):
            status('execution_protocol_review_pending','成功率开发验收已通过；等待TRAIN执行敏感性诊断审阅，正式TEST保持停止。')
            run(['scripts/report_aligned_r07.py'],'report',sim=False);return
        comparison_lock=freeze('comparison_v6_inputs.json',values=True,acceptance=True)
        status('baseline_validation','成功率模块通过开发验收；作者传感器和v6作者执行的PhysMani运行完整VAL。')
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(baseline,'validation',task,comparison_lock) for task in TASKS]):future.result()
        verify_lock(ROOT,comparison_lock)
        status('official_TEST','共同外部条件下运行四组策略与PhysMani共400次TEST。')
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures=[pool.submit(ours,'test',p,r,comparison_lock) for p in ('cv','context') for r in ('geometry','value')]
            futures += [pool.submit(baseline,'test',task,comparison_lock) for task in TASKS]
            for future in as_completed(futures):future.result()
        run(['scripts/summarize_aligned_r07_v6.py'],'final_report',sim=False);verify_lock(ROOT,original)
        final=json.loads((BASE/'final_report.json').read_text())
        # Summary itself enforces all paired sensor/clock and baseline identity checks.
        status('complete','修复、开发验收及400次对齐TEST完成。',report='artifacts/framework/aligned_r07/final_report.json')
        run(['scripts/report_aligned_r07.py'],'report',sim=False)
    except Exception as exc:
        status('needs_attention',str(exc),error=repr(exc));raise

if __name__=='__main__':main()
