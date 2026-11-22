"""Accept isolated r09 TEST conditions and summarize frozen-policy outcomes."""
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
from our_mani.sensor_alignment.author_rgbd import PROTOCOL as SENSOR_PROTOCOL,RGB_SPEC
from our_mani.sensor_alignment.physmani_executor_v6 import EXECUTOR,REFERENCE_FILTER
from our_mani.risk_v2.selector import attempt_controller_hash
BASE=ROOT/'artifacts/framework/aligned_r09'
FROZEN_CRITIC_BASE=ROOT/'artifacts/framework/aligned_r08'
CASE_ISOLATION='fresh_simulator_process_per_episode'
TASKS=('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed')
VARIANTS=('cv_geometry','cv_value','context_geometry','context_value','physmani')
TOLERANCES=None
LABELS={'cv_geometry':'CV + geometry','cv_value':'CV + success','context_geometry':'Context + geometry','context_value':'Context + success','physmani':'PhysMani (aligned adapter)'}

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def cases(name):
    directory=BASE/('test_physmani_v6' if name=='physmani' else 'test_'+name)
    records=[json.loads(path.read_text()) for path in sorted(directory.glob('pick*__episode*.json')) if not path.name.endswith('infrastructure_error.json')]
    indexed={(r['task'],r['raw_episode_id']):r for r in records}
    if len(indexed)!=len(records):raise ValueError('duplicate episode identity in '+str(directory))
    return indexed

def sensor_problems(conditions):
    problems=[]
    expected_cameras={'front','8','16','36'}
    sensor=conditions.get('sensor_protocol',{})
    if sensor.get('version')!=SENSOR_PROTOCOL:problems.append('sensor_protocol_version')
    if sensor.get('camera_render_modes')!={name:'OPENGL' for name in expected_cameras}:problems.append('sensor_protocol_render_modes')
    if sensor.get('rgb_processing')!=RGB_SPEC:problems.append('sensor_protocol_rgb_processing')
    cameras=conditions.get('cameras',{})
    if set(cameras)!=expected_cameras:problems.append('sensor_camera_set')
    for name in expected_cameras & set(cameras):
        if cameras[name].get('render_mode')!='OPENGL':problems.append('camera_'+name+'_render_mode')
        if cameras[name].get('raw_rgb_255_processing')!=RGB_SPEC['observation/raw_rgb_255'][name]:
            problems.append('camera_'+name+'_raw_rgb_255_processing')
    return problems

def case_process_check(row,name,locked_files):
    """Verify preserved child evidence for one independently launched case.

    PIDs are reported as evidence only: the operating system may reuse a PID
    after a prior process exits, so uniqueness across the full run is not an
    acceptance condition.
    """
    problems=[]
    receipt=row.get('case_process_receipt',{})
    if row.get('case_isolation')!=CASE_ISOLATION:problems.append('case_isolation')
    if receipt.get('version')!='fresh_process_r09_v1':problems.append('case_process_receipt_version')
    child_evaluator='scripts/evaluate_physmani_aligned_r09_v6.py' if name=='physmani' else 'scripts/evaluate_aligned_r09.py'
    launcher='scripts/run_aligned_gpu.sh' if name=='physmani' else 'scripts/run_local_sim.sh'
    if receipt.get('child_evaluator')!=child_evaluator:problems.append('child_evaluator')
    if receipt.get('child_evaluator_sha256')!=locked_files.get(child_evaluator):problems.append('frozen_child_evaluator_sha256')
    if receipt.get('launcher')!=launcher:problems.append('launcher')
    pid=receipt.get('parent_observed_process_id')
    if type(pid) is not int or pid<=0:problems.append('parent_observed_process_id')
    for field in ('started_utc','finished_utc'):
        try:
            value=datetime.fromisoformat(receipt[field])
            if value.utcoffset() is None:problems.append(field+'_timezone')
        except (KeyError,TypeError,ValueError):problems.append(field+'_format')
    if not any(problem.startswith(('started_utc','finished_utc')) for problem in problems):
        if datetime.fromisoformat(receipt['finished_utc'])<datetime.fromisoformat(receipt['started_utc']):
            problems.append('process_timestamp_order')
    command=receipt.get('launch_command')
    if not isinstance(command,list) or not all(isinstance(token,str) for token in command):
        problems.append('launch_command');command=[]
    def argument(flag):
        if command.count(flag)!=1:return None
        index=command.index(flag)+1
        return command[index] if index<len(command) else None
    if child_evaluator not in command or launcher not in command:problems.append('launch_command_entrypoints')
    for flag,value in (('--raw-episode-id',row['raw_episode_id']),('--task',row['task']),('--split',row['split'])):
        if argument(flag)!=value:problems.append('launch_command_'+flag[2:])
    if '--one-decision' in command or '--init-only' in command:problems.append('diagnostic_command_in_full_evaluation')
    child=None;child_path=None;inputs=None;inputs_path=None
    for path_field,sha_field in (('child_record_path','child_record_sha256'),('child_inputs_path','child_inputs_sha256')):
        try:
            raw_path=receipt[path_field]
            if not isinstance(raw_path,str) or Path(raw_path).is_absolute():raise ValueError('must be a relative path')
            path=(ROOT/raw_path).resolve()
            if not path.is_relative_to(BASE.resolve()):raise ValueError('must stay within r09 artifacts')
            data=path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=receipt.get(sha_field):problems.append(sha_field)
            parsed=json.loads(data)
            if path_field=='child_record_path':child,child_path=parsed,path
            else:inputs,inputs_path=parsed,path
        except (OSError,KeyError,TypeError,ValueError) as error:problems.append(path_field+':'+str(error))
    if child is not None:
        for field in ('task','raw_episode_id','split','official_task_success'):
            if child.get(field)!=row.get(field):problems.append('preserved_child_'+field)
        if any(field not in row or row[field]!=value for field,value in child.items()):
            problems.append('parent_changed_original_child_fields')
        child_pid=child.get('evaluation_process_pid')
        if child_pid is not None and (type(child_pid) is not int or child_pid<=0):problems.append('child_evaluation_process_pid')
    if inputs is not None:
        input_content={field:value for field,value in inputs.items() if field!='sha256'}
        identity=hashlib.sha256(json.dumps(input_content,sort_keys=True).encode()).hexdigest()
        if inputs.get('sha256')!=identity:problems.append('child_inputs_declared_identity')
        if inputs.get('files',{}).get(child_evaluator)!=locked_files.get(child_evaluator):
            problems.append('child_inputs_evaluator_source_identity')
        if child is not None and 'inputs_sha256' in child and child['inputs_sha256']!=identity:
            problems.append('child_record_input_identity')
    if child_path is not None:
        output_argument=argument('--out')
        if output_argument is None or (ROOT/output_argument).resolve()!=child_path.parent:
            problems.append('launch_command_output_directory')
        if inputs_path is not None and inputs_path.parent!=child_path.parent:problems.append('child_inputs_directory')
    return dict(accepted=not problems,problems=problems,receipt=receipt,
                child_evaluation_process_pid=child.get('evaluation_process_pid') if child else None,
                pid_interpretation='process evidence only; PID reuse after exit is permitted')

def compare_conditions(a,b,tolerances=None):
    tolerances=tolerances or TOLERANCES
    if tolerances is None:tolerances=json.loads((BASE/'acceptance_tolerances.json').read_text())
    problems=[]
    x,y=a['reset_receipt'],b['reset_receipt']
    for key in ('task','raw_episode_id','variation','official_episode_index','target_state_sha256'):
        if x[key]!=y[key]:problems.append('reset_'+key)
    for key,tolerance in (('target_position_metric_only',tolerances['target_position_m']),
                          ('goal_pose_metric_only',tolerances['fixture_pose_component']),
                          ('distractor_poses_metric_only',tolerances['fixture_pose_component']),
                          ('robot_joints',tolerances['robot_joints_rad'])):
        if not np.allclose(x[key],y[key],rtol=0.,atol=tolerance):problems.append('reset_'+key)
    px,py=np.asarray(x['robot_pose']),np.asarray(y['robot_pose'])
    robot_position_delta=float(np.linalg.norm(px[:3]-py[:3]))
    robot_orientation_delta=float((Rotation.from_quat(px[3:]).inv()*Rotation.from_quat(py[3:])).magnitude())
    if robot_position_delta>tolerances['robot_position_m']:problems.append('reset_robot_position')
    if robot_orientation_delta>tolerances['robot_orientation_rad']:problems.append('reset_robot_orientation')
    x,y=a['public_initial_conditions'],b['public_initial_conditions']
    if x['information_setting']!=y['information_setting']:problems.append('information_setting')
    if x.get('sensor_protocol')!=y.get('sensor_protocol'):problems.append('sensor_protocol_mismatch')
    problems += ['a_'+problem for problem in sensor_problems(x)]
    problems += ['b_'+problem for problem in sensor_problems(y)]
    if x['cameras'].keys()!=y['cameras'].keys():problems.append('camera_sets_differ')
    for name in x['cameras'].keys() & y['cameras'].keys():
        for key in ('intrinsics','camera_to_world'):
            if not np.allclose(x['cameras'][name][key],y['cameras'][name][key],rtol=0.,atol=tolerances['camera_calibration_component']):problems.append('camera_'+name+'_'+key)
        for key in ('render_mode','raw_rgb_255_processing'):
            if x['cameras'][name].get(key)!=y['cameras'][name].get(key):problems.append('camera_'+name+'_'+key+'_mismatch')
    xm={r['name']:r for r in x['static_map']};ym={r['name']:r for r in y['static_map']}
    if xm.keys()!=ym.keys():problems.append('static_map_objects')
    else:
        for name in xm:
            if xm[name]['mesh_sha256']!=ym[name]['mesh_sha256']:problems.append('static_mesh_'+name)
            if not np.allclose(xm[name]['pose'],ym[name]['pose'],rtol=0.,atol=tolerances['fixture_pose_component']):problems.append('static_pose_'+name)
    return dict(accepted=not problems,problems=problems,initial_rgbd_bitwise_identical=x['rgbd_sha256']==y['rgbd_sha256'],
                robot_position_delta_m=robot_position_delta,robot_orientation_delta_rad=robot_orientation_delta)

def main():
    global TOLERANCES
    TOLERANCES=json.loads((BASE/'acceptance_tolerances.json').read_text())
    config=json.loads((ROOT/'config/framework/aligned_protocol_r09.json').read_text())
    baseline_config=json.loads((ROOT/'config/framework/aligned_protocol_r09_baseline_v6.json').read_text())
    allowed_filter_fields={'baseline_executor','baseline_reference_filter','baseline_reference_filter_scope'}
    config_differences={key:dict(ours=config.get(key),baseline=baseline_config.get(key))
                        for key in set(config)|set(baseline_config) if config.get(key)!=baseline_config.get(key)}
    if set(config_differences)-allowed_filter_fields:raise ValueError('unmatched common configuration: '+str(config_differences))
    if baseline_config['baseline_executor']!=EXECUTOR or baseline_config.get('baseline_reference_filter')!=REFERENCE_FILTER:
        raise ValueError('unexpected v6 baseline reference-filter identity')
    lock=json.loads((BASE/'frozen_inputs.json').read_text());verify_lock(ROOT,lock)
    comparison_lock=json.loads((BASE/'comparison_v6_inputs.json').read_text());verify_lock(ROOT,comparison_lock)
    frozen_critic_lock=json.loads((FROZEN_CRITIC_BASE/'frozen_inputs.json').read_text());verify_lock(ROOT,frozen_critic_lock)
    locked_paths=set(lock['files'])|set(comparison_lock['files'])
    locked_files={**lock['files'],**comparison_lock['files']}
    required_paths={'scripts/summarize_aligned_r09_v6.py','scripts/evaluate_physmani_aligned_r09_v6.py',
                    'our_mani/ablations/physmani_executor_v4.py','our_mani/sensor_alignment/author_rgbd.py',
                    'our_mani/sensor_alignment/physmani_executor_v5.py','config/framework/aligned_protocol_r09.json',
                    'our_mani/sensor_alignment/physmani_executor_v6.py','config/framework/aligned_protocol_r09_baseline_v6.json',
                    'our_mani/risk_v4/model.py','scripts/evaluate_aligned_r09.py',
                    'artifacts/framework/aligned_r09/acceptance_tolerances.json',
                    'artifacts/framework/pipeline_v1/official_test_manifest.json',
                    config['dynamics_checkpoint'],config['cv_local_value'],config['context_local_value']}
    if required_paths-locked_paths:raise ValueError('required r09 inputs missing from locks: '+str(sorted(required_paths-locked_paths)))
    verify_lock(ROOT,json.loads((ROOT/'artifacts/framework/aligned_r05/frozen_inputs.json').read_text()))
    verify_lock(ROOT,json.loads((ROOT/'artifacts/framework/pipeline_v2/comparison_r04_frozen_inputs.json').read_text()))
    source=json.loads((ROOT/'artifacts/framework/aligned_r05/baseline_source_audit.json').read_text())
    changed=[path for path,digest in source['files'].items() if file_sha256(Path(path))!=digest]
    expected={(row['task'],row['raw_episode_id']) for row in json.loads((ROOT/'artifacts/framework/pipeline_v1/official_test_manifest.json').read_text())['episodes']}
    data={name:cases(name) for name in VARIANTS}
    baseline_checkpoint_sha256=file_sha256(Path(config['baseline_checkpoint']))
    errors=[];rates={};checks=[]
    success_predictors={}
    for predictor in ('cv','context'):
        checkpoint=ROOT/config['cv_local_value' if predictor=='cv' else 'context_local_value']
        if checkpoint!=FROZEN_CRITIC_BASE/(predictor+'_success/value.pt'):
            raise ValueError('r09 must use the frozen r08 '+predictor+' success checkpoint')
        predictor_report=json.loads(checkpoint.with_name('report.json').read_text())
        metadata=predictor_report['metadata']
        if (metadata.get('classifier_retrained') is not False or metadata.get('formal_test_used') is not False
                or metadata.get('classifier_selection_used_CAL') is not False
                or metadata.get('calibration_selector_used_CAL') is not True):
            raise ValueError('unexpected frozen r08 classifier/CAL lineage for '+predictor)
        controller_sha=attempt_controller_hash(ROOT,config,predictor)
        if metadata['controller_sha256']!=controller_sha:
            raise ValueError('frozen critic continuation controller differs for '+predictor)
        success_predictors[predictor]=dict(checkpoint=str(checkpoint.relative_to(ROOT)),
            checkpoint_sha256=file_sha256(checkpoint),report_sha256=file_sha256(checkpoint.with_name('report.json')),
            support_sha256=file_sha256(checkpoint.with_name('value_support.json')),
            controller_sha256=controller_sha,metadata=metadata,
            source_coverage=predictor_report.get('source_coverage'),
            calibration_source_crossfit=predictor_report.get('calibration_source_crossfit'),
            calibration_source_raw=predictor_report.get('calibration_source_raw'),
            calibration_source_crossfit_prior=predictor_report.get('calibration_source_crossfit_prior'),
            calibration_crossfit_valid=predictor_report.get('calibration_crossfit_valid'),
            calibration_per_task=metadata.get('calibration_per_task'),
            diagnostics_origin='frozen r08 source-nested CAL diagnostics, not newly fitted r09 VAL metrics',
            classifier_retrained_for_r09=False,calibration_refitted_for_r09=False)
    case_isolation_checks=[]
    clipping=dict(decisions=0,command_ticks=0,velocity_clip_ticks=0,position_clip_ticks=0,
                  any_clip_ticks=0,maximum_velocity_clip_rad=0.,maximum_position_clip_rad=0.,
                  maximum_requested_minus_applied_rad=0.)
    counterfactual=dict(decisions=0,command_ticks=0,velocity_clip_ticks=0,position_clip_ticks=0,
                        any_clip_ticks=0,maximum_velocity_clip_rad=0.,maximum_position_clip_rad=0.,
                        maximum_requested_minus_applied_rad=0.)
    actual_speed=dict(command_ticks=0,native_joint_upper_velocity_limits_rad_per_s=None,
                      maximum_actual_average_speed_fraction=0.,maximum_actual_native_reported_speed_fraction=0.,
                      ticks_average_speed_over_nominal=0,ticks_native_speed_over_nominal=0,
                      ticks_average_speed_over_nominal_plus_1percent=0,ticks_native_speed_over_nominal_plus_1percent=0,
                      measurement='native actual qdot and 50ms actual q differences, not reference speed; no substep peak guarantee')
    for name,records in data.items():
        if set(records)!=expected:errors.append(dict(variant=name,error='TEST coverage',missing=sorted(expected-set(records)),unexpected=sorted(set(records)-expected)))
        rates[name]={}
        for task in TASKS:
            rows=[r for (t,_),r in records.items() if t==task]
            successes=sum(r['official_task_success'] for r in rows)
            rates[name][task]=dict(episodes=len(rows),successes=successes,success_rate=successes/len(rows) if rows else None,
                                  wilson_95=wilson(successes,len(rows)),mean_wall_seconds=float(np.mean([r['wall_seconds'] for r in rows])) if rows else None)
        for key,row in records.items():
            process_check=case_process_check(row,name,locked_files)
            case_isolation_checks.append(dict(variant=name,key=key,**process_check))
            if not process_check['accepted']:errors.append(dict(variant=name,key=key,
                error='fresh per-episode process isolation receipt',problems=process_check['problems']))
            receipt=row['alignment_receipt'];clock=receipt['clock']
            accepted=(row.get('split')=='test' and row.get('formal_test_used') is True
                      and 0<=clock['ticks']<=round(config['simulation_seconds']/config['dt'])
                      and abs(clock['dt']-config['dt'])<1e-9
                      and abs(clock['budget_seconds']-config['simulation_seconds'])<1e-9
                      and abs(clock['simulation_seconds']-clock['ticks']*config['dt'])<1e-8
                      and abs(clock['simulation_seconds']-receipt['actual_physics_seconds'])<.001
                      and abs(clock['simulation_seconds']-receipt['actual_task_seconds'])<.001
                      and row['official_task_success']==receipt['native_success_latched'])
            if name!='physmani':
                accepted=accepted and np.allclose(row['planning_acceleration_limits'],config['global_joint_acceleration'],rtol=0,atol=1e-7)
                predictor,ranking=name.split('_',1)
                predictor_info=success_predictors[predictor]
                if (row.get('predictor')!=predictor or row.get('ranking')!=ranking
                        or row.get('controller_sha256')!=predictor_info['controller_sha256']):
                    errors.append(dict(variant=name,key=key,error='ours controller/predictor/ranking identity'))
                expected_value=predictor_info['checkpoint_sha256'] if ranking=='value' else None
                if (row.get('value_checkpoint_sha256')!=expected_value
                        or row.get('value_controller_match_verified')!=(ranking=='value')):
                    errors.append(dict(variant=name,key=key,error='frozen r08 success checkpoint identity'))
                if row.get('selection_policy_sha256')!=file_sha256(ROOT/'our_mani/risk_v3/selector.py'):
                    errors.append(dict(variant=name,key=key,error='selection policy source identity'))
                if row.get('sensor_protocol_sha256')!=file_sha256(ROOT/'our_mani/sensor_alignment/author_rgbd.py'):
                    errors.append(dict(variant=name,key=key,error='sensor protocol source identity'))
            if not accepted:errors.append(dict(variant=name,key=key,error='clock/success/limits acceptance'))
            checks.append(dict(variant=name,key=key,accepted=bool(accepted)))
            individual_sensor_errors=sensor_problems(row['public_initial_conditions'])
            if individual_sensor_errors:errors.append(dict(variant=name,key=key,error='sensor domain acceptance',problems=individual_sensor_errors))
            if name=='physmani':
                if row.get('executor_revision')!=baseline_config['baseline_executor'] or row.get('method')!='PhysMani_released_weights_released_waypoints_counted_actuator_author_rgbd_v6':
                    errors.append(dict(variant=name,key=key,error='baseline v6 identity'))
                if row.get('checkpoint_sha256')!=baseline_checkpoint_sha256:errors.append(dict(variant=name,key=key,error='baseline checkpoint identity'))
                for step in row.get('steps',[]):
                    execution=step['execution'];values=execution.get('common_joint_command_clipping')
                    if execution.get('executor')!=EXECUTOR or execution.get('sensor_protocol')!=SENSOR_PROTOCOL or execution.get('reference_filter')!=REFERENCE_FILTER:
                        errors.append(dict(variant=name,key=key,error='baseline v6 step identity'))
                    if values is None:
                        errors.append(dict(variant=name,key=key,error='missing common joint clipping receipt'));continue
                    clipping['decisions']+=1
                    for field in ('command_ticks','velocity_clip_ticks','position_clip_ticks','any_clip_ticks'):clipping[field]+=values[field]
                    for field in ('maximum_velocity_clip_rad','maximum_position_clip_rad','maximum_requested_minus_applied_rad'):
                        clipping[field]=max(clipping[field],values[field])
                    if values['velocity_clip_ticks']!=0:errors.append(dict(variant=name,key=key,error='unexpected baseline velocity reference filter'))
                    hypothetical=execution.get('counterfactual_ours_reference_clip')
                    measured=execution.get('actual_speed_receipt')
                    if hypothetical is None or measured is None:
                        errors.append(dict(variant=name,key=key,error='missing reference sensitivity or actual speed receipt'));continue
                    counterfactual['decisions']+=1
                    for field in ('command_ticks','velocity_clip_ticks','position_clip_ticks','any_clip_ticks'):counterfactual[field]+=hypothetical[field]
                    for field in ('maximum_velocity_clip_rad','maximum_position_clip_rad','maximum_requested_minus_applied_rad'):
                        counterfactual[field]=max(counterfactual[field],hypothetical[field])
                    native_limits=measured['native_joint_upper_velocity_limits_rad_per_s']
                    if actual_speed['native_joint_upper_velocity_limits_rad_per_s'] is None:actual_speed['native_joint_upper_velocity_limits_rad_per_s']=native_limits
                    elif not np.allclose(actual_speed['native_joint_upper_velocity_limits_rad_per_s'],native_limits,rtol=0,atol=1e-9):
                        errors.append(dict(variant=name,key=key,error='native velocity parameters changed'))
                    if measured['command_ticks']!=execution['ticks']:errors.append(dict(variant=name,key=key,error='actual-speed sampling coverage'))
                    for field in ('command_ticks','ticks_average_speed_over_nominal','ticks_native_speed_over_nominal',
                                  'ticks_average_speed_over_nominal_plus_1percent','ticks_native_speed_over_nominal_plus_1percent'):
                        actual_speed[field]+=measured[field]
                    for field in ('maximum_actual_average_speed_fraction','maximum_actual_native_reported_speed_fraction'):
                        if not np.isfinite(measured[field]):errors.append(dict(variant=name,key=key,error='nonfinite actual speed'))
                        actual_speed[field]=max(actual_speed[field],measured[field])
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
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),complete=all(set(x)==expected for x in data.values()),
        protocol_acceptance_passed=not errors,total_test_runs=sum(len(x) for x in data.values()),expected_test_runs=400,
        variants=rates,paired_comparisons=paired,acceptance_errors=errors,conditions=conditions,clock_checks=checks,
        case_isolation=CASE_ISOLATION,case_isolation_checks=case_isolation_checks,
        frozen_success_predictors=success_predictors,
        information_setting=config['information_setting'],baseline_checkpoint_sha256=baseline_checkpoint_sha256,
        shared_sensor_protocol=dict(version=SENSOR_PROTOCOL,camera_render_modes={name:'OPENGL' for name in ('front','8','16','36')},rgb_processing=RGB_SPEC),
        baseline_actual_reference_position_bounds=clipping,
        baseline_counterfactual_ours_reference_clip=counterfactual,baseline_actual_joint_speed=actual_speed,
        controller_reference_policy=dict(ours='current-q reference clipping is part of our controller algorithm',
                                         baseline='author direct RML setpoints with native joint position bounds',
                                         shared='same native robot parameters, sensors, counted 10s budget and incremental gripper',
                                         baseline_config_only_differences=config_differences),
        baseline_scope='one fixed released epoch99999; released waypoint rules and direct RML targets retained; shared counted time/gripper/sensors; not original Table1 reproduction',
        baseline_executor=EXECUTOR,acceptance_tolerances=TOLERANCES,
        paper_reference={'normal_speed_percent':84.2,'high_speed_percent':18.3,'normal_speed_checkpoint_std_percent':3.1,'high_speed_checkpoint_std_percent':2.4,
                         'url':'https://arxiv.org/pdf/2607.01938','comparison':'Table 1 original protocol; reference only; not paired with these rollouts',
                         'aggregation':'paper selected-checkpoint mean/std; our fixed-checkpoint Wilson intervals are different quantities'},
        limitations=['two selected tasks; ours specialized perception/grasp priors versus released multitask policy',
                    'current simulation instance masks and known static map available to both; not RGB-D-only',
                    'one checkpoint per method; no multi-seed claim; paired p-values exploratory and unadjusted',
                    'public TEST previously evaluated by older versions; no fresh-unseen-test claim',
                    'Value predicts success on this candidate attempt before replacement, including actual approach and remaining time; not eventual success across retries',
                    'synchronous simulation; concurrent-job wall times are diagnostics, not real-time guarantees',
                    'baseline v6 preserves author direct RML targets with shared native physics/counting/gripper; ours reference filter is a controller algorithm, not an external hardware cap',
                    'native nominal velocity parameters do not imply zero numerical/transient overshoot; actual measured speed and exceedance counts are reported',
                    'each learned success predictor is one frozen five-member bootstrap ensemble',
                    'r06 GL3-domain training data remain preserved; r07 author-OPENGL adaptation/recollection must follow frozen TRAIN/VAL lineage',
                    'front OPENGL and native RGB buffers are shared; task-specific tracker still consumes the declared uint8 image channel',
                    'r08 reuses frozen r07 raw heads/TRAIN/support/selector/controller/sensors and collected data; only CAL probability calibration changes via nested source-separated selection',
                    'r09 freezes those r08 heads and nested CAL corrections; it does not retrain classifiers or refit calibration on the new r09 VAL',
                    'every r09 VAL/TEST case starts a fresh simulator process; original official TEST reset, physics and acceptance tolerances are retained',
                    'earlier development evaluations reused simulator fixtures and could drift across episodes; success differences with zero actual learned overrides are not evidence of critic benefit',
                    'critic probability quality, actual changes to selected candidates, and policy success gains are different measurements; task success alone does not identify critic contribution'])
    write(BASE/'final_report.json',report)
    lines=['r09 每回合独立进程、冻结R08成功率模块共同条件测试',f"正式TEST：{report['total_test_runs']}/400；条件验收：{'通过' if not errors else '未通过'}",
           '输入：共同作者OPENGL传感器、公开原生RGB通道、当前实例mask与静态地图；不使用真实未来轨迹。',
           '基线：作者epoch99999权重＋统一协议适配；论文表1数字仅作原协议参考。','',
           '方法 | 普通速度 | 高速']
    for name in VARIANTS:
        values=[rates[name][t] for t in TASKS]
        lines.append(LABELS[name]+' | '+' | '.join(f"{v['successes']}/{v['episodes']} ({100*v['success_rate']:.1f}%)" if v['success_rate'] is not None else '0/0（未完成）' for v in values))
    lines += ['','TRAIN/VAL来源与混合传感器训练域以R08冻结清单为准；VAL用于选模/校准，不是独立TEST。',
              '算法内部规划、碰撞筛选和MPC保留；控制频率与地图来源的贡献另做消融。',
              'R09直接使用冻结R08预测头与嵌套CAL校准；本轮VAL不重训或重新校准模型。',
              '每个VAL/TEST回合独立启动仿真进程；官方TEST重置、物理参数与验收容差保持既定协议。',
              '此前开发评估复用场景存在fixtures漂移；真实选点override为0时，SR差异不能归因为成功率模块。',
              '概率评分质量、实际改变选点和最终任务收益分别报告；成功率点估计本身不能识别critic贡献。',
              '机械臂原生参数、10秒计时与夹爪协议一致；我们的方法裁剪参考位置，基线保留作者直接RML目标，均视为各自控制算法。',
              '公开TEST此前已评估；这是同一公开任务上的版本比较，不能称为从未见过的新TEST。',
              '同步仿真冻结计算期间物理时间；墙钟耗时不能证明实时运行能力。',
              '两任务使用已知红色方块、姿态与目标区域标定先验；不能外推为多物体通用策略。',
              '单次40回合/任务，不能仅凭点估计宣称统计显著。',
              f'基线实际位置边界处理：{json.dumps(clipping,ensure_ascii=False)}',
              f'若使用我们参考滤波的反事实诊断：{json.dumps(counterfactual,ensure_ascii=False)}',
              f'基线实际关节速度测量：{json.dumps(actual_speed,ensure_ascii=False)}',
              '原生标称速度可能有数值/瞬态超出；不能把参考速度当作实际速度或宣称硬速度完全不超。',
              f'验收问题：{json.dumps(errors,ensure_ascii=False)}']
    (ROOT/'r09完整测试结果.txt').write_text('\n'.join(lines)+'\n')
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
    figure.suptitle('Aligned r09: isolated episodes, frozen critics and common environment')
    figure.tight_layout();figure.savefig(BASE/'success_rates.png',dpi=160);plt.close(figure)

if __name__=='__main__':main()
