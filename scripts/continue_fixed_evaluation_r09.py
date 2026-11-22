"""Evaluate a frozen candidate despite documented VAL performance-screen failure.

This changes only the stopping rule after VAL under the user's explicit request
for complete TRAIN/VAL/TEST. It never changes a model, threshold or outcome.
All calibration, coverage, process and paired initial-condition checks remain.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import run_fresh_case_comparison_r09 as run
from our_mani.benchmark_protocol import file_sha256,verify_lock
BASE=run.BASE
SOURCE_SHA=file_sha256(Path(__file__))

def check_clock(row):
    receipt=row['alignment_receipt'];clock=receipt['clock']
    values=[clock['ticks'],clock['simulation_seconds'],clock['dt'],clock['budget_seconds'],
        receipt['actual_physics_seconds'],receipt['actual_task_seconds']]
    if not np.isfinite(np.asarray(values,dtype=float)).all():raise ValueError('nonfinite VAL clocks')
    if (type(clock['ticks']) is not int or not 0<=clock['ticks']<=200
        or abs(clock['dt']-.05)>1e-9 or abs(clock['budget_seconds']-10.)>1e-9
        or abs(clock['simulation_seconds']-clock['ticks']*.05)>1e-8
        or abs(receipt['actual_physics_seconds']-clock['simulation_seconds'])>.001
        or abs(receipt['actual_task_seconds']-clock['simulation_seconds'])>.001
        or row['official_task_success']!=receipt['native_success_latched']):raise ValueError('VAL clock/native success invariant failed')
    if row['split']!='validation' or row['formal_test_used'] is not False:raise ValueError('wrong VAL split/exposure flag')


def hard_checks():
    lock=json.loads((BASE/'frozen_inputs.json').read_text());verify_lock(ROOT,lock)
    conditions=json.loads((BASE/'validation_conditions.json').read_text())
    if not conditions['passed']:raise ValueError('strict VAL initial conditions failed')
    usage=json.loads((BASE/'success_selection_usage_validation.json').read_text())
    if not usage['diagnostics_valid'] or not usage['coverage_complete']:raise ValueError('incomplete/invalid selection diagnostics')
    gate=json.loads((BASE/'success_acceptance.json').read_text())
    expected_failure='context pick_moving_target_on_the_table selection VAL regression'
    if gate['passed'] is not False or gate['failures']!=[expected_failure]:raise ValueError('only the documented exact performance failure can be exempted')
    expected={(row['task'],row['raw_episode_id']) for row in run.manifest('validation')}
    for variant in run.VARIANTS[:-1]:
        rows=run.records('validation',variant)
        if len(rows)!=40 or {(r['task'],r['raw_episode_id']) for r in rows}!=expected:raise ValueError('incomplete VAL cases')
        for row in rows:
            check=run.case_process_check(row,variant,lock['files'])
            if not check['accepted']:raise ValueError('VAL process evidence failed')
            check_clock(row)
    for predictor in ('cv','context'):
        report=json.loads((BASE/f'{predictor}_success/report.json').read_text())
        if report['metadata'].get('formal_test_used') or not report['metadata']['all_200_TRAIN_source_episodes_executed']:
            raise ValueError('TRAIN coverage or TEST leakage')
        for task in run.TASKS:
            cross=report['calibration_crossfit'][task]
            if not cross['valid'] or not cross['metrics']:raise ValueError('invalid source calibration crossfit')
            ece,brier,raw=cross['metrics']['ece_5_bins'],cross['metrics']['brier'],cross['raw_metrics']['brier']
            if not np.isfinite([ece,brier,raw]).all() or ece>.15 or brier>raw+.01:raise ValueError('calibration hard gate failed')
            coverage=report['source_coverage'][task]
            if coverage['TRAIN_executed_sources']!=100 or not coverage['all_observed_TRAIN_sources_have_positive_ensemble_weights']:
                raise ValueError('critic TRAIN source coverage failed')
    return lock,gate

def decision(gate):
    path=BASE/'fixed_candidate_evaluation_release.json'
    if path.exists():
        value=json.loads(path.read_text())
        if value['original_performance_screen_sha256']!=file_sha256(BASE/'success_acceptance.json'):raise ValueError('VAL screening evidence changed')
        return value
    value=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        evaluation_allowed=True,deployment_or_benefit_validated=False,development_performance_screen_passed=gate['passed'],
        original_performance_screen_sha256=file_sha256(BASE/'success_acceptance.json'),
        original_performance_failures=gate['failures'],original_VAL_results=gate['results'],
        protocol_deviation='After observing VAL, the performance-based stopping rule is changed under the explicit user instruction to complete TRAIN/VAL/formal TEST. The failed screen is retained. No model or threshold is changed.',
        user_instruction='我早上来验收 你先一直跑完TRAIN VAL 和正式的TEST',
        purpose='complete comparison of a fixed candidate, including negative findings; evaluation rather than deployment',
        unchanged_hard_requirements=['full TRAIN source execution and no TEST fitting','valid nested source calibration, ECE<=.15 and Brier<=raw+.01',
            'all VAL cases and valid selection/process evidence','unchanged strict initial-condition tolerances','all five methods, every TEST case, no outcome-based selection'],
        model_or_threshold_retuned_after_VAL=False,TEST_success_labels_used_for_release=False,
        previous_public_TEST_exposure='r04 full and r05 partial; no fresh-unseen TEST claim')
    run.write(path,value)
    body=('R09 固定候选正式评估说明\n'
        'TRAIN和概率校准/严格初始条件检查已完成。开发性能筛选未通过：Context普通速度geometry20/20，success18/20。\n'
        '原failed筛选JSON、模型、阈值和所有VAL结果保持不变。依据用户继续完成TRAIN、VAL和正式TEST的明确指令，本轮继续评价固定候选，并如实保留负面结果。\n'
        '这是在观察VAL后改变性能停止规则的协议偏离，不能称作预注册性能门槛通过、部署放行或成功率模块收益已验证。\n'
        '概率/数据/计时/进程/严格初始条件门槛继续保留；五组完整TEST均运行，不按结果删case或调参数。\n')
    (BASE/'固定候选评估说明.txt').write_text(body)
    return value

def extra_lock(comparison):
    if file_sha256(Path(__file__))!=SOURCE_SHA:raise ValueError('continuation source changed since launch')
    path=BASE/'fixed_evaluation_extra_inputs.json'
    if path.exists():
        value=json.loads(path.read_text());verify_lock(ROOT,value);return value
    files=dict(comparison['files'])
    for source in (Path(__file__),BASE/'fixed_candidate_evaluation_release.json',BASE/'固定候选评估说明.txt',BASE/'comparison_v6_inputs.json'):
        files[str(source.relative_to(ROOT))]=file_sha256(source)
    value=dict(created_utc=datetime.now(timezone.utc).isoformat(),files=files,
        development_performance_screen_passed=False,stopping_rule_changed_after_VAL=True,
        fixed_model_comparison_not_benefit_validation=True)
    run.write(path,value);return value

def main():
    try:
        lock,gate=hard_checks();release=decision(gate)
        run.status('baseline_validation','TRAIN与完整VAL已完成；性能筛选失败原样保留，按用户要求对固定候选继续完整基线VAL和TEST。',development_performance_screen_passed=gate['passed'])
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(run.run_group,'validation','physmani',lock,task) for task in run.TASKS]):future.result()
        if file_sha256(BASE/'success_acceptance.json')!=release['original_performance_screen_sha256']:raise ValueError('original failed VAL screen changed')
        for variant in run.VARIANTS:
            for row in run.records('validation',variant):check_clock(row)
        conditions=run.parity('validation',run.VARIANTS)
        if not conditions['passed']:raise ValueError('five-method VAL initial conditions failed; no tolerance relaxation')
        comparison=run.freeze('comparison_v6_inputs.json',acceptance=True)
        extra=extra_lock(comparison);verify_lock(ROOT,extra)
        run.status('official_TEST','完成所有VAL与严格配对；固定模型执行五组共400次正式TEST，开发性能筛选失败的披露保留。',development_performance_screen_passed=gate['passed'])
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures=[pool.submit(run.run_group,'test',variant,extra) for variant in run.VARIANTS[:-1]]
            futures += [pool.submit(run.run_group,'test','physmani',extra,task) for task in run.TASKS]
            for future in as_completed(futures):future.result()
        conditions=run.parity('test',run.VARIANTS);run.selection_usage('test')
        subprocess.run([sys.executable,'scripts/summarize_aligned_r09_v6.py'],cwd=ROOT,check=True)
        if not conditions['passed']:raise ValueError('full TEST strict pairwise initial conditions failed')
        verify_lock(ROOT,extra)
        report=json.loads((BASE/'final_report.json').read_text())
        report.update(development_performance_screen_passed=gate['passed'],
            fixed_candidate_evaluation_release=release,selection_usage_report='success_selection_usage_test.json',
            evaluation_interpretation='complete frozen-candidate comparison after a disclosed failed VAL performance screen; not validated deployment or guaranteed improvement')
        run.write(BASE/'final_report.json',report)
        run.status('complete','TRAIN、完整VAL、五组400次正式TEST及两段视频完成；VAL性能筛选失败与正式结果一并披露。',
            development_performance_screen_passed=gate['passed'],report='artifacts/framework/aligned_r09/final_report.json')
        run.report()
    except Exception as exc:
        run.status('needs_attention',str(exc),error=repr(exc));raise

if __name__=='__main__':main()
