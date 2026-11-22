"""Read-only v6 geometry and backward GT-twist collection regression audit."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import analyze_multitask_training_v5 as v5
from scripts import analyze_multitask_smoke_r10 as common
from scripts import collect_multitask_training_r10 as c
from our_mani.multitask.motion_labels import SCHEMA,RAW_LINEAR,RAW_ANGULAR
from our_mani.multitask.track_dataset import validate_motion_label_schema
from our_mani.multitask.training_geometry import _check_asset
from our_mani.multitask.component_flags import partition_components
BASE=c.BASE


def audit_case(path,plan,case):
    result=v5.audit_case(path,plan,case)
    record=common.read(path);entry=record['entry'];files=record['files_sha256']
    with np.load(ROOT/entry['cache'],allow_pickle=False) as stored:a={key:stored[key].copy() for key in stored.files}
    common.require(plan['protocol']['dynamics_target_label_schema']==SCHEMA,'v6 motion schema missing')
    validate_motion_label_schema(a,SCHEMA,entry.get('dynamics_target_label_schema'))
    common.require(not a['target_twist_valid'][0].any(),'first-frame derivative falsely valid')
    common.require(not np.any(a['target_valid']&~a['target_twist_valid']),'unified future mask lost twist boundary')
    label_path=ROOT/entry['motion_label_diagnostics'];label=common.read(label_path)
    common.require(files.get(entry['motion_label_diagnostics'])==common.sha(label_path) and label['schema']==SCHEMA,
        'motion diagnostics not SHA-bound')
    common.require(label['native_pose_marked_valid_but_nonfinite_or_non_SO3']==0,'invalid GT pose hidden by derivative mask')
    geometry_paths=[ROOT/name for name in files if Path(name).name=='TRAIN_geometry.json']
    common.require(len(geometry_paths)==1,'TRAIN geometry asset missing/duplicate')
    asset=common.read(geometry_paths[0]);_check_asset(asset)
    common.require(set(asset['bodies'])==set(map(str,a['body_ids'])),'asset/body identity coverage differs')
    geometry={}
    for j,body in enumerate(map(str,a['body_ids'])):
        shape=asset['bodies'][body];roles=partition_components(shape['owned_component_annotations'])
        for name in ('visual_component_names','collision_component_names','empty_visual_role','empty_collision_role'):
            common.require(roles[name]==shape[name],'component classification differs from native TRAIN flags: '+body)
        components={component['name'] for component in shape['components']}
        common.require(components==set(shape['visual_component_names']) and
            set(record['current_instance_bindings'][body])==components,'collision-only or undeclared component entered visual binding')
        if shape['empty_visual_role']:
            common.require(not a['valid'][:,j].any(),'nonrenderable role marked directly visible history')
        if shape['empty_collision_role']:
            common.require(not shape['collision_vertices_body'] and not shape['collision_triangles'],'visual-only role has collision geometry')
        geometry[body]=dict(visual_components=shape['visual_component_names'],collision_components=shape['collision_component_names'],
            empty_visual=shape['empty_visual_role'],empty_collision=shape['empty_collision_role'],
            collision_vertices=len(shape['collision_vertices_body']),collision_triangles=len(shape['collision_triangles']),
            actual_valid_history_rows=int(a['valid'][:,j].sum()),
            full_horizon_windows=entry['context_window_eligibility']['by_body'][body]['full_horizon_windows'])
        raw=dict(a,target_linear_velocities=a[RAW_LINEAR],target_angular_velocities=a[RAW_ANGULAR])
        if not case['teacher_used']:
            raw.update(teacher_action_valid=np.ones(len(a['times']),bool),teacher_stage=np.full(len(a['times']),'hold'),teacher_phase=np.full(len(a['times']),'hold'))
        result['bodies'][body]['native_twist']=common.native_twist_audit(raw,j,v5.THRESHOLDS)
        result['bodies'][body]['native_twist']['raw_comparator_arrays']=[RAW_LINEAR,RAW_ANGULAR]
        result['bodies'][body]['derived_twist_valid_rows']=int(a['target_twist_valid'][:,j].sum())
    result.update(geometry_asset=common.pointer(geometry_paths[0]),geometry_component_roles=geometry,
        dynamics_target_label_schema=SCHEMA,motion_label_math_verified=True,motion_label_diagnostics=label,
        raw_reported_velocity_fields_present=True,raw_native_velocity_rewritten=False,
        derived_target_velocity_fields_are_raw_native=False,
        label_file=common.pointer(label_path))
    return result


def analyze(plan_path,queue_path):
    plan=common.read(plan_path)
    common.require(plan['plan_sha256']==common.digest({k:v for k,v in plan.items() if k!='plan_sha256'}),'v6 plan digest invalid')
    common.require(len(plan['cases'])==28 and len({case['parent_source_group'] for case in plan['cases']})==14,'fixed28/14 inventory differs')
    verified=[];invalid=[];missing=[];infra=[]
    for case in plan['cases']:
        path=Path(queue_path)/'cases'/case['split']/case['task']/case['case_id']/'case.json'
        item=dict(case_id=case['case_id'],task=case['task'],behavior=case['behavior'])
        if path.exists():
            try:verified.append(audit_case(path,plan,case))
            except Exception as error:invalid.append(dict(item,error=str(error)))
        else:
            receipts=list((Path(queue_path)/'infrastructure_attempts'/case['case_id']).glob('*/parent_receipt.json'))
            failures=[common.read(p) for p in receipts if common.read(p)['status']=='infrastructure_error']
            if failures:infra.append(dict(item,attempts=failures))
            else:missing.append(item)
    teacher=[row for row in verified if row['teacher'] is not None];hold=[row for row in verified if row['context_primary_view']]
    return dict(version='r10_offline_fixed_teacher_hold_smoke_v6',created_utc=c.utc(),
        plan=common.pointer(plan_path),plan_sha256=plan['plan_sha256'],analysis_source=common.pointer(__file__),
        reused_analysis_sources=[common.pointer(v5.__file__),common.pointer(common.__file__)],
        mathematical_schema_validator=common.pointer(ROOT/'our_mani/multitask/track_dataset.py'),
        expected=28,verified=len(verified),invalid=len(invalid),infrastructure=len(infra),missing=len(missing),
        complete=len(verified)==28 and not invalid,counts_by_behavior=dict(Counter(row['behavior'] for row in verified)),
        teacher_cases=len(teacher),teacher_native_successes=sum(row['native_success'] for row in teacher),
        hold_parents=len(hold),hold_full_horizon_parents=sum(row['parent_has_1p1_second_horizon'] for row in hold),
        teacher_completed_commands=sum(row['teacher']['completed_commands'] for row in teacher),
        explicit_phase_completion_events=sum(len(row['teacher']['phase_completion_events']) for row in teacher),
        public_joint_recovery_ticks=sum(row['public_joint_recovery_ticks'] for row in verified),
        actual_release_hold_ticks=sum(row['actual_release_hold_ticks'] for row in verified),
        native_nonrenderable_roles=[dict(task=row['task'],behavior=row['behavior'],body=body) for row in verified
            for body,info in row['geometry_component_roles'].items() if info['empty_visual']],
        noncolliding_visual_roles=[dict(task=row['task'],behavior=row['behavior'],body=body) for row in verified
            for body,info in row['geometry_component_roles'].items() if info['empty_collision']],
        zero_full_horizon_hold_bodies=[dict(task=row['task'],body=body) for row in hold for body,info in row['bodies'].items()
            if info['windows']['full_horizon_windows']==0],
        cases=verified,invalid_cases=invalid,missing_cases=missing,infrastructure_cases=infra,
        interpretation='fixed TRAIN geometry/label integration regression only; native teacher success is not pipeline SR; one parent/task cannot establish formal skill support; no outcome-selected retries',
        allnone_orientation_is_not_full_SE3_observability=True,full_source_training_started=False,
        model_fitting_performed=False,labels_modified=False,VAL_read=False,TEST_read=False,native_or_GPU_launched=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=BASE/'TRAIN_pilot_collection_plan_v6.json')
    parser.add_argument('--queue',type=Path,default=BASE/'TRAIN_smoke_collection_v6')
    parser.add_argument('--report',type=Path,default=BASE/'TRAIN_smoke_analysis_v6.json')
    args=parser.parse_args();report=analyze(args.plan,args.queue);c.write_json(args.report,report,overwrite=True)
    text=[f"R10 v6固定TRAIN smoke：{report['verified']}/28 SHA/数学验收；invalid={report['invalid']} infra={report['infrastructure']} pending={report['missing']}",
        f"hold完整1.1s父源 {report['hold_full_horizon_parents']}/{report['hold_parents']}；teacher原生成功 {report['teacher_native_successes']}/{report['teacher_cases']}（采集诊断，不是pipeline SR）",
        f"phase事件={report['explicit_phase_completion_events']}，world-waypoint完成commands={report['teacher_completed_commands']}；geometry视觉/碰撞分开、非renderable角色不伪造可见性。",
        '新GT后向差分与边界mask逐项复算，raw原生速度独立保留。没有全量训练/VAL/TEST。']
    text += [row['task']+' | '+row['behavior']+f" | ticks={row['ticks']} native={row['native_success']} fullwin={row['context_window_eligibility']['full_horizon_windows']}" for row in report['cases']]
    text += ['INVALID '+str(row) for row in report['invalid_cases']]
    Path(args.report).with_suffix('.txt').write_text('\n'.join(text)+'\n')
    print({key:report[key] for key in ('verified','invalid','infrastructure','missing','hold_parents','hold_full_horizon_parents','teacher_cases','teacher_native_successes')})
if __name__=='__main__':main()
