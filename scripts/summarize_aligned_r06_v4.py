"""Accept full coverage/conditions and summarize paired native TEST outcomes."""
from datetime import datetime,timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import binomtest
from scipy.spatial.transform import Rotation
from our_mani.benchmark_protocol import file_sha256,verify_lock,wilson
BASE=ROOT/'artifacts/framework/aligned_r06'
TASKS=('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed')
VARIANTS=('cv_geometry','cv_value','context_geometry','context_value','physmani')
TOLERANCES=json.loads((BASE/'acceptance_tolerances.json').read_text())
LABELS={'cv_geometry':'CV + geometry','cv_value':'CV + success','context_geometry':'Context + geometry','context_value':'Context + success','physmani':'PhysMani (aligned adapter)'}

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def cases(name):
    directory=BASE/('test_physmani_v4' if name=='physmani' else 'test_'+name)
    records=[json.loads(path.read_text()) for path in sorted(directory.glob('pick*__episode*.json')) if not path.name.endswith('infrastructure_error.json')]
    return {(r['task'],r['raw_episode_id']):r for r in records}

def compare_conditions(a,b):
    problems=[]
    x,y=a['reset_receipt'],b['reset_receipt']
    for key in ('task','raw_episode_id','variation','official_episode_index','target_state_sha256'):
        if x[key]!=y[key]:problems.append('reset_'+key)
    for key in ('target_position_metric_only','goal_pose_metric_only','distractor_poses_metric_only','robot_joints'):
        if not np.allclose(x[key],y[key],rtol=0.,atol=1e-4):problems.append('reset_'+key)
    px,py=np.asarray(x['robot_pose']),np.asarray(y['robot_pose'])
    robot_position_delta=float(np.linalg.norm(px[:3]-py[:3]))
    robot_orientation_delta=float((Rotation.from_quat(px[3:]).inv()*Rotation.from_quat(py[3:])).magnitude())
    if robot_position_delta>TOLERANCES['robot_position_m']:problems.append('reset_robot_position')
    if robot_orientation_delta>TOLERANCES['robot_orientation_rad']:problems.append('reset_robot_orientation')
    x,y=a['public_initial_conditions'],b['public_initial_conditions']
    if x['information_setting']!=y['information_setting']:problems.append('information_setting')
    for name in x['cameras']:
        for key in ('intrinsics','camera_to_world'):
            if not np.allclose(x['cameras'][name][key],y['cameras'][name][key],rtol=0.,atol=1e-5):problems.append('camera_'+name+'_'+key)
    xm={r['name']:r for r in x['static_map']};ym={r['name']:r for r in y['static_map']}
    if xm.keys()!=ym.keys():problems.append('static_map_objects')
    else:
        for name in xm:
            if xm[name]['mesh_sha256']!=ym[name]['mesh_sha256']:problems.append('static_mesh_'+name)
            if not np.allclose(xm[name]['pose'],ym[name]['pose'],rtol=0.,atol=1e-4):problems.append('static_pose_'+name)
    return dict(accepted=not problems,problems=problems,initial_rgbd_bitwise_identical=x['rgbd_sha256']==y['rgbd_sha256'],
                robot_position_delta_m=robot_position_delta,robot_orientation_delta_rad=robot_orientation_delta)

def main():
    lock=json.loads((BASE/'frozen_inputs.json').read_text());verify_lock(ROOT,lock)
    verify_lock(ROOT,json.loads((BASE/'comparison_v4_inputs.json').read_text()))
    verify_lock(ROOT,json.loads((ROOT/'artifacts/framework/aligned_r05/frozen_inputs.json').read_text()))
    verify_lock(ROOT,json.loads((ROOT/'artifacts/framework/pipeline_v2/comparison_r04_frozen_inputs.json').read_text()))
    source=json.loads((ROOT/'artifacts/framework/aligned_r05/baseline_source_audit.json').read_text())
    changed=[path for path,digest in source['files'].items() if file_sha256(Path(path))!=digest]
    expected={(row['task'],row['raw_episode_id']) for row in json.loads((ROOT/'artifacts/framework/pipeline_v1/official_test_manifest.json').read_text())['episodes']}
    data={name:cases(name) for name in VARIANTS}
    errors=[];rates={};checks=[]
    for name,records in data.items():
        if set(records)!=expected:errors.append(dict(variant=name,error='TEST coverage',missing=sorted(expected-set(records))))
        rates[name]={}
        for task in TASKS:
            rows=[r for (t,_),r in records.items() if t==task]
            successes=sum(r['official_task_success'] for r in rows)
            rates[name][task]=dict(episodes=len(rows),successes=successes,success_rate=successes/len(rows) if rows else None,
                                  wilson_95=wilson(successes,len(rows)),mean_wall_seconds=float(np.mean([r['wall_seconds'] for r in rows])) if rows else None)
        for key,row in records.items():
            receipt=row['alignment_receipt'];clock=receipt['clock']
            accepted=(clock['ticks']<=200 and abs(clock['simulation_seconds']-receipt['actual_physics_seconds'])<.001
                      and abs(clock['simulation_seconds']-receipt['actual_task_seconds'])<.001
                      and row['official_task_success']==receipt['native_success_latched'])
            if name!='physmani':accepted=accepted and np.allclose(row['planning_acceleration_limits'],6.)
            if not accepted:errors.append(dict(variant=name,key=key,error='clock/success/limits acceptance'))
            checks.append(dict(variant=name,key=key,accepted=bool(accepted)))
    conditions=[]
    for name in VARIANTS[:-1]:
        for key in sorted(expected & data[name].keys() & data['physmani'].keys()):
            check=compare_conditions(data[name][key],data['physmani'][key]);conditions.append(dict(variant=name,key=key,**check))
            if not check['accepted']:errors.append(dict(variant=name,key=key,error='paired conditions',problems=check['problems']))
    paired={}
    for a,b in itertools.combinations(VARIANTS,2):
        paired[a+'_vs_'+b]={}
        for task in TASKS:
            keys=[key for key in sorted(expected) if key[0]==task and key in data[a] and key in data[b]]
            a_only=sum(data[a][k]['official_task_success'] and not data[b][k]['official_task_success'] for k in keys)
            b_only=sum(not data[a][k]['official_task_success'] and data[b][k]['official_task_success'] for k in keys)
            paired[a+'_vs_'+b][task]=dict(n=len(keys),a_only=a_only,b_only=b_only,
                b_minus_a_percentage_points=100*(b_only-a_only)/len(keys) if keys else None,
                exact_mcnemar_p_unadjusted=float(binomtest(b_only,a_only+b_only,p=.5).pvalue) if a_only+b_only else 1.)
    if changed:errors.append(dict(error='baseline sources changed during run',paths=changed))
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),complete=all(len(x)==80 for x in data.values()),
        protocol_acceptance_passed=not errors,total_test_runs=sum(len(x) for x in data.values()),expected_test_runs=400,
        variants=rates,paired_comparisons=paired,acceptance_errors=errors,conditions=conditions,clock_checks=checks,
        information_setting='simulation_assisted_current_instance_masks_and_shared_static_map',baseline_checkpoint_sha256=file_sha256(Path(json.loads((ROOT/'config/framework/aligned_protocol_r06.json').read_text())['baseline_checkpoint'])),
        baseline_scope='one fixed released epoch99999; released path truncation/spacing and ignore-collision mode retained; common counted actuator adaptation, not author Table1 reproduction',
        baseline_executor='released_waypoints_no_hidden_physics_v4',acceptance_tolerances=TOLERANCES,
        paper_reference={'normal_speed_percent':84.2,'high_speed_percent':18.3,'url':'https://arxiv.org/pdf/2607.01938','comparison':'different original protocol; reference only'},
        limitations=['two selected tasks; ours specialized perception/grasp priors versus released multitask policy',
                    'current simulation instance masks and known static map available to both; not RGB-D-only',
                    'one checkpoint per method; no multi-seed claim; paired p-values exploratory and unadjusted',
                    'public TEST previously evaluated by older versions; no fresh-unseen-test claim',
                    'Value predicts success on this candidate attempt before replacement, including actual approach and remaining time; not eventual success across retries',
                    'synchronous simulation; concurrent-job wall times are diagnostics, not real-time guarantees',
                    'baseline v4 preserves released waypoint target reset and recoverable arm errors; common joint clipping is recorded',
                    'each learned success predictor is one frozen five-member bootstrap ensemble'])
    write(BASE/'final_report.json',report)
    lines=['r06 成功率修复后完整policy、共同外部条件测试',f"正式TEST：{report['total_test_runs']}/400；条件验收：{'通过' if not errors else '未通过'}",
           '输入：共同RGB-D、当前实例mask、公开静态地图；不使用真实未来轨迹。',
           '基线：作者epoch99999权重＋统一协议适配；论文表1数字仅作原协议参考。','',
           '方法 | 普通速度 | 高速']
    for name in VARIANTS:
        values=[rates[name][t] for t in TASKS]
        lines.append(LABELS[name]+' | '+' | '.join(f"{v['successes']}/{v['episodes']} ({100*v['success_rate']:.1f}%)" for v in values))
    lines += ['','候选成功率头分别执行200 TRAIN来源各3次完整候选探索；VAL用于选模/校准，不是独立TEST。',
              '算法内部规划、碰撞筛选和MPC保留；控制频率与地图来源的贡献另做消融。',
              '单次40回合/任务，不能仅凭点估计宣称统计显著。',f'验收问题：{json.dumps(errors,ensure_ascii=False)}']
    (ROOT/'r06完整测试结果.txt').write_text('\n'.join(lines)+'\n')
    if errors:raise RuntimeError(f'alignment acceptance failed: {errors[:3]}')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    colors=['#6386a6','#406e96','#61a18a','#267f67','#ba8553']
    for axis,task,title,paper in zip(axes,TASKS,('Normal speed','High speed'),(.842,.183)):
        values=[rates[name][task]['success_rate'] for name in VARIANTS]
        intervals=[rates[name][task]['wilson_95'] for name in VARIANTS]
        axis.bar(np.arange(5),values,color=colors,yerr=np.array([[v-ci[0] for v,ci in zip(values,intervals)],[ci[1]-v for v,ci in zip(values,intervals)]]),capsize=4)
        axis.set_xticks(np.arange(5),['CV\ngeometry','CV\nsuccess','Context\ngeometry','Context\nsuccess','PhysMani\naligned'])
        axis.set_title(title);axis.set_ylim(0,1.13)
        axis.axhline(paper,color='#777777',linestyle='--',linewidth=1,label='Paper reference (original protocol)')
        for index,value in enumerate(values):axis.text(index,value+.035,f'{100*value:.1f}%',ha='center',fontsize=10)
        axis.grid(axis='y',alpha=.15);axis.set_axisbelow(True)
    axes[0].set_ylabel('Native task success rate (Wilson 95% interval)')
    axes[1].legend(loc='upper left',fontsize=8)
    figure.suptitle('Aligned r06: complete policies, common environment and information access')
    figure.tight_layout();figure.savefig(BASE/'success_rates.png',dpi=160);plt.close(figure)

if __name__=='__main__':main()
