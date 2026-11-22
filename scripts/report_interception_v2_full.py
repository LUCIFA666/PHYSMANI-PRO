"""Audit every frozen official outcome and summarize the four-way full test."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import fcntl
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.benchmark_protocol import file_sha256,verify_lock,wilson

OUT=ROOT/'artifacts/framework/pipeline_v2'
RUN='comparison_r03'
VARIANTS=('cv_geometry','cv_value','context_geometry','context_value')
TASKS=('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed')
LABELS={'cv_geometry':'恒速＋几何','cv_value':'恒速＋成功率','context_geometry':'context＋几何','context_value':'context＋成功率'}


def write(path,data):
    q=path.with_suffix('.tmp');q.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(path)


def percentile(values,p):
    if not values:return None
    values=sorted(values);k=(len(values)-1)*p;a=int(k);b=min(a+1,len(values)-1)
    return values[a]+(values[b]-values[a])*(k-a)


def failure_stage(row):
    if row['official_task_success']:return 'success'
    if row.get('failure')=='development_reset_ik_failed':return 'reset_failure'
    if not row['handovers']:
        reasons=Counter(s['decision'].get('reason') for s in row['steps'])
        if reasons['no_feasible_interception_candidate']>=len(row['steps'])*.2:
            return 'no_handover_frequent_infeasible_candidates'
        return 'no_handover_time_or_capture_gate'
    if not row.get('captured'):return 'handover_without_grasp'
    if any('grasp lost' in e.get('reason','') for e in row.get('events',[])):return 'grasp_lost_after_capture'
    return 'captured_without_goal_completion'


def metrics(rows):
    n=len(rows);k=sum(r['official_task_success'] for r in rows)
    handed=sum(r['handovers']>0 for r in rows);captured=sum(bool(r.get('captured')) for r in rows)
    success_times=[r['success_time'] for r in rows if r['official_task_success'] and 'success_time' in r]
    walls=[r['wall_seconds'] for r in rows]
    errors=[];retargets=[]
    for row in rows:
        arrival=[s['candidate_goal']['arrival'] for s in row['steps'] if s.get('candidate_goal') and s['stage']=='intercept']
        retargets.append(sum(abs(a-b)>.025 for a,b in zip(arrival[1:],arrival[:-1])))
        errors.extend(s['perception_error_metric_only_m'] for s in row['steps'] if s.get('perception_error_metric_only_m') is not None)
    global_acc=[r.get('maximum_measured_global_joint_acceleration_rad_s2',0.) for r in rows]
    ci=wilson(k,n)
    return {'episodes':n,'successes':k,'success_rate':k/n if n else None,
        'success_wilson_95':[max(0.,ci[0]),min(1.,ci[1])] if ci else None,
        'episodes_with_handover':handed,'handover_rate':handed/n if n else None,
        'episodes_with_capture':captured,'capture_rate':captured/n if n else None,
        'success_given_handover':k/handed if handed else None,
        'success_given_capture':k/captured if captured else None,
        'median_success_simulation_seconds':statistics.median(success_times) if success_times else None,
        'mean_wall_seconds':statistics.mean(walls) if walls else None,
        'median_wall_seconds':statistics.median(walls) if walls else None,'wall_seconds_p90':percentile(walls,.9),
        'mean_perception_error_metric_only_m':statistics.mean(errors) if errors else None,
        'median_arrival_retarget_count':statistics.median(retargets) if retargets else None,
        'maximum_observed_global_joint_acceleration_rad_s2':max(global_acc) if global_acc else None,
        'episodes_with_observed_global_joint_acceleration_above_6_1':sum(a>6.1 for a in global_acc),
        'local_joint_acceleration_hard_constraint_verified':False,
        'failure_stages':dict(Counter(failure_stage(r) for r in rows))}


def distance(a,b):return math.sqrt(sum((float(x)-float(y))**2 for x,y in zip(a,b)))


def matching_reset(a,b):
    ra,rb=a['reset_receipt'],b['reset_receipt']
    checks={'same_target_driver':ra['target_state_sha256']==rb['target_state_sha256'],
        'same_variation':ra['variation']==rb['variation'],
        'target_position_within_1mm':distance(ra['target_position_metric_only'],rb['target_position_metric_only'])<=.001,
        'goal_position_within_1mm':distance(ra['goal_pose_metric_only'][:3],rb['goal_pose_metric_only'][:3])<=.001,
        'robot_position_within_1mm':distance(ra['robot_pose'][:3],rb['robot_pose'][:3])<=.001,
        'robot_joint_difference_within_1mrad':max(abs(x-y) for x,y in zip(ra['robot_joints'],rb['robot_joints']))<=.001,
        'distractors_within_1mm':all(distance(x[:3],y[:3])<=.001 for x,y in zip(ra['distractor_poses_metric_only'],rb['distractor_poses_metric_only']))}
    return all(checks.values()),checks


def paired_comparison(a,b):
    by_a={(r['task'],r['raw_episode_id']):r for r in a}
    by_b={(r['task'],r['raw_episode_id']):r for r in b}
    results={}
    for task in TASKS:
        eligible=[];excluded=[]
        for key in sorted(set(by_a)&set(by_b)):
            if key[0]!=task:continue
            ok,checks=matching_reset(by_a[key],by_b[key])
            if ok:eligible.append((by_a[key]['official_task_success'],by_b[key]['official_task_success']))
            else:excluded.append({'raw_episode_id':key[1],'checks':checks})
        a_only=sum(x and not y for x,y in eligible);b_only=sum(y and not x for x,y in eligible)
        n=a_only+b_only
        p=min(1.,2*sum(math.comb(n,i) for i in range(min(a_only,b_only)+1))/(2**n)) if n else 1.
        results[task]={'matched_reset_episodes':len(eligible),'excluded_from_pairing_only':excluded,
            'a_success_b_failure':a_only,'a_failure_b_success':b_only,
            'b_minus_a_success_rate':(b_only-a_only)/len(eligible) if eligible else None,
            'exact_mcnemar_p_unadjusted':p,
            'interpretation':'single-run exploratory within-ours pairing; not adjusted for multiple comparisons; primary success includes all40 episodes'}
    return results


def collect():
    manifest=json.loads((ROOT/'artifacts/framework/pipeline_v1/official_test_manifest.json').read_text())['episodes']
    expected={(r['task'],r['raw_episode_id']) for r in manifest}
    all_rows={};progress={};issues=[]
    for variant in VARIANTS:
        directory=OUT/f'{RUN}_test_{variant}'
        reports=[]
        for p in sorted(directory.glob('pick*.json')):
            if p.name.endswith('.infrastructure_error.json'):
                actual=directory/p.name.replace('.infrastructure_error.json','.json')
                if not actual.exists():issues.append({'variant':variant,'episode':p.name,'error':json.loads(p.read_text())})
                continue
            row=json.loads(p.read_text())
            if type(row.get('official_task_success')) is not bool:raise ValueError(f'missing explicit outcome: {p}')
            if row.get('formal_test_used') is not True or row.get('split')!='test' or row.get('FASTER_used') is not False:
                raise ValueError(f'incompatible benchmark record: {p}')
            if f"{row.get('predictor')}_{row.get('ranking')}" != variant:
                raise ValueError(f'variant identity mismatch: {p}')
            reports.append(row)
        keys=[(r['task'],r['raw_episode_id']) for r in reports]
        if len(set(keys))!=len(keys) or not set(keys)<=expected:raise ValueError(f'official episode coverage mismatch: {variant}')
        if len({r['inputs_sha256'] for r in reports})>1:raise ValueError(f'mixed frozen inputs: {variant}')
        all_rows[variant]=reports
        progress[variant]={'completed':len(reports),'expected':80,
            'per_task':{task:metrics([r for r in reports if r['task']==task]) for task in TASKS}}
    return all_rows,progress,issues


def plot(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    labels=['CV\nGeometry','CV\nValue','Context\nGeometry','Context\nValue']
    colors=['#7a8da4','#a8b9c9','#2c8c87','#61b8ae']
    for ax,task,title in zip(axes,TASKS,['Normal moving pick','High-speed moving pick']):
        values=[report['variants'][v]['per_task'][task] for v in VARIANTS]
        rates=np.array([r['success_rate']*100 for r in values])
        ci=np.array([r['success_wilson_95'] for r in values])*100
        ax.bar(range(4),rates,color=colors,width=.65)
        ax.errorbar(range(4),rates,yerr=np.maximum(0.,np.vstack([rates-ci[:,0],ci[:,1]-rates])),fmt='none',ecolor='#243447',capsize=4)
        for i,r in enumerate(values):ax.text(i,max(rates[i],ci[i,1])+2,f'{r["successes"]}/40',ha='center',fontsize=10)
        paper=report['paper_reported_PhysMani'][task]
        ax.axhline(paper['mean_percent'],color='#c76a37',linestyle='--',label=f'PhysMani paper {paper["mean_percent"]:.1f}%')
        ax.set_xticks(range(4),labels);ax.set_ylim(0,112);ax.set_yticks(range(0,101,20))
        ax.set_title(title);ax.set_ylabel('Official task success (%)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='upper right',fontsize=8)
    fig.suptitle(f'No-FASTER candidate interception — frozen {RUN}, 40 episodes per task')
    fig.text(.5,-.015,'Bars: our single-run result; intervals: Wilson 95%. Dashed lines: authors’ reported PhysMani means, not a local rerun.',ha='center',fontsize=8)
    fig.savefig(OUT/'full_test_success_rates.png',dpi=180,bbox_inches='tight');plt.close(fig)


def once():
    lock=json.loads((OUT/f'{RUN}_frozen_inputs.json').read_text());verify_lock(ROOT,lock)
    rows,progress,issues=collect()
    val_progress={}
    for variant in VARIANTS:
        p=OUT/f'{RUN}_validation_{variant}/report.json'
        r=json.loads(p.read_text()) if p.exists() else {}
        val_progress[variant]={'completed':r.get('completed',0),'expected':40}
    complete=all(p['completed']==80 for p in progress.values()) and not issues
    all_complete=complete and all(p['completed']==40 for p in val_progress.values())
    paper=json.loads((ROOT/'config/framework/paper_reference_comparison.json').read_text())
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'run':RUN,'FASTER_used':False,
        'official_test_complete':complete,'test_and_validation_complete':all_complete,
        'completed_test_runs':sum(p['completed'] for p in progress.values()),'expected_test_runs':320,
        'variants':progress,'validation_progress':val_progress,'pending_infrastructure_issues':issues,
        'frozen_inputs_sha256':file_sha256(OUT/f'{RUN}_frozen_inputs.json'),
        'paper_reported_PhysMani':paper['task_success_reference']['methods']['PhysMani'],
        'paper_url':paper['paper'],'paper_aggregation':paper['task_success_reference']['aggregation'],
        'protocol_limits':['two moving-pick tasks, not the complete16-task benchmark',
            'task-specific RGBD red-box perception and TRAIN-calibrated goal marker',
            'single frozen model per variant; no multi-seed training experiment',
            'our20Hz control and native gripper execution differ from the released policy executor',
            'nominal10s episode budget; synchronous simulation pauses computation time',
            'local success Value does not estimate global approach probability',
            'public TEST benchmark was previously evaluated by v1; no fresh-unseen-test claim',
            'actual servo errors and observed acceleration violations retained; local joint acceleration hard constraint not verified']}
    if RUN=='comparison_r04':
        protocol_path=OUT/'full_train_r04/protocol.json'
        report['training_protocol']=json.loads(protocol_path.read_text())
        report['training_coverage']={}
        for name,folder in [('dynamics','dynamics'),('cv_value','cv_success'),('context_value','context_success')]:
            training=json.loads((OUT/f'full_train_r04/{folder}/report.json').read_text())
            report['training_coverage'][name]={'metadata':training['metadata'],
                'partitions':training.get('partitions'), 'checkpoint_sha256':training['checkpoint_sha256']}
        report['protocol_limits'].append('all40 VAL episodes contribute to model selection/calibration; VAL metrics are development diagnostics')
    if complete:
        report['paired_comparisons']={f'{a}_vs_{b}':paired_comparison(rows[a],rows[b]) for a,b in
            [('cv_geometry','context_geometry'),('cv_value','context_value'),('cv_geometry','cv_value'),('context_geometry','context_value')]}
    write(OUT/'full_test_report.json',report)
    lines=['无FASTER四组完整测试',f'更新时间UTC：{report["created_utc"]}',
        f'官方TEST：{report["completed_test_runs"]}/320；VAL：{sum(p["completed"] for p in val_progress.values())}/160。',
        '普通/高速两个任务；各组每任务40条官方TEST，完整保留失败。']
    if RUN=='comparison_r04':
        lines.append('r04：动力学及两个局部成功率头均使用全部200条对应任务TRAIN源episode训练；VAL仅选模型与校准。')
    for variant,p in progress.items():
        for task,label in zip(TASKS,['普通','高速']):
            r=p['per_task'][task]
            rate=f'{100*r["success_rate"]:.1f}%' if r['success_rate'] is not None else '待测'
            lines.append(f'{LABELS[variant]} / {label}: 成功{r["successes"]}/{r["episodes"]}（{rate}），接管{r["episodes_with_handover"]}，抓住{r["episodes_with_capture"]}。')
            if r['episodes']:lines.append(f'  失败分类：{r["failure_stages"]}；平均单episode墙钟{r["mean_wall_seconds"]:.1f}秒。')
    lines+=['基线：PhysMani论文Table1作者报告普通84.2±3.1%、高速18.3±2.4%；未本地复现。',
        '论文±为最佳3/最后5检查点的标准差；ours置信区间为单次运行Wilson区间，含义不同。',
        '当前Value为局部接管后的条件概率，未校准全链路成功率。',
        '正式主成功率包含全部40条episode；复位匹配筛选只用于附加配对分析。',
        '完整证据：artifacts/framework/pipeline_v2/full_test_report.json']
    (ROOT/'完整测试进度_无FASTER.txt').write_text('\n'.join(lines)+'\n')
    if all_complete:
        (ROOT/'完整测试结果_无FASTER.txt').write_text('\n'.join(lines)+'\n')
        plot(report)
    print(json.dumps({'test':report['completed_test_runs'],'validation':sum(p['completed'] for p in val_progress.values()),'complete':all_complete,'infrastructure_issues':len(issues)}),flush=True)
    return all_complete


def main():
    global RUN
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--run',default='comparison_r03')
    args=parser.parse_args()
    if args.run not in ('comparison_r03','comparison_r04'):
        raise ValueError('unknown frozen comparison run')
    RUN=args.run
    while True:
        lock_path=ROOT/'runtime/framework/interception_full_report.lock'
        lock_path.parent.mkdir(parents=True,exist_ok=True)
        with lock_path.open('a') as reporting_lock:
            fcntl.flock(reporting_lock,fcntl.LOCK_EX)
            complete=once()
        if complete or not args.watch:return
        time.sleep(30)


if __name__=='__main__':main()
