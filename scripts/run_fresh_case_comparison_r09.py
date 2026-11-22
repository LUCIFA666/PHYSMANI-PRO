"""Frozen critics; fresh native simulator process for each VAL and TEST case."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.benchmark_protocol import file_sha256,verify_lock
from summarize_aligned_r09_v6 import compare_conditions,sensor_problems,case_process_check
BASE=ROOT/'artifacts/framework/aligned_r09'
SOURCE_SHA=file_sha256(Path(__file__))
TASKS=('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed')
VARIANTS=('cv_geometry','cv_value','context_geometry','context_value','physmani')
REPORT_LOCK=threading.Lock()

def now():return datetime.now(timezone.utc).isoformat()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def status(stage,description,**extra):
    previous=json.loads((BASE/'status.json').read_text()) if (BASE/'status.json').exists() else {}
    exposed=previous.get('formal_test_started',False) or stage in ('official_TEST','complete') or any(BASE.glob('test_*/pick*__episode*.json'))
    write(BASE/'status.json',dict(created_utc=now(),stage=stage,description=description,
        formal_test_started=bool(exposed),FASTER_used=False,NGFF_used=False,case_isolation='fresh_simulator_process_per_episode',**extra))

def source_paths():
    config=json.loads((ROOT/'config/framework/aligned_protocol_r09.json').read_text())
    paths=[Path(__file__),ROOT/'config/framework/aligned_protocol_r09.json',ROOT/'config/framework/aligned_protocol_r09_baseline_v6.json',
        ROOT/'scripts/evaluate_aligned_r09.py',ROOT/'scripts/evaluate_physmani_aligned_r09_v6.py',
        ROOT/'scripts/summarize_aligned_r09_v6.py',ROOT/'scripts/analyze_success_usage_r09.py',
        ROOT/'scripts/run_local_sim.sh',ROOT/'scripts/run_aligned_gpu.sh',
        ROOT/'scripts/recalibrate_candidate_success_r08.py',
        *sorted((ROOT/'our_mani/risk_v2').glob('*.py')),*sorted((ROOT/'our_mani/risk_v3').glob('*.py')),
        *sorted((ROOT/'our_mani/risk_v4').glob('*.py')),*sorted((ROOT/'our_mani/sensor_alignment').glob('*.py')),
        ROOT/'our_mani/ablations/physmani_executor_v4.py',
        ROOT/'artifacts/framework/aligned_r08/frozen_inputs.json',
        ROOT/'artifacts/framework/aligned_r08/success_acceptance.json',
        ROOT/'scripts/probe_official_reset_parity_r08.py',ROOT/'scripts/summarize_aligned_r08_v6.py',
        *sorted((ROOT/'artifacts/framework/aligned_r08/official_reset_parity_probe').glob('*__*.json')),
        ROOT/'artifacts/framework/aligned_r08/official_reset_parity_probe/report.json',
        ROOT/'artifacts/framework/aligned_r07/baseline_executor_v6_smoke/report.json',
        ROOT/'artifacts/framework/aligned_r07/execution_sensitivity_probe/report.json',
        BASE/'case_filter_smoke/report.json',BASE/'geometry_fallback_equivalence_review.json',BASE/'acceptance_tolerances.json',
        ROOT/'artifacts/framework/protocol/train_manifest.json',ROOT/'artifacts/framework/protocol/validation_manifest.json',
        ROOT/'artifacts/framework/pipeline_v1/official_test_manifest.json',
        ROOT/config['dynamics_checkpoint'],ROOT/config['goal_calibration'],Path(config['baseline_checkpoint'])]
    certificate=json.loads((BASE/'geometry_fallback_equivalence_review.json').read_text())
    verify_lock(ROOT,certificate)
    for name,digest in certificate.get('native_prediction_source_files',{}).items():
        if file_sha256(Path(name))!=digest:raise ValueError('native critic implementation differs from source certificate')
    paths += [ROOT/name for name in certificate['files']]
    paths += [Path(name) for name in certificate.get('native_prediction_source_files',{})]
    baseline_audit=ROOT/'artifacts/framework/aligned_r05/baseline_source_audit.json'
    baseline_sources=json.loads(baseline_audit.read_text())
    for name,digest in baseline_sources['files'].items():
        if file_sha256(Path(name))!=digest:raise ValueError('released baseline source changed')
    paths += [baseline_audit,*[Path(name) for name in baseline_sources['files']]]
    paths += [*sorted((ROOT/'our_mani').glob('*.py')),*sorted((ROOT/'our_mani/aligned').glob('*.py'))]
    paths += [ROOT/config[key] for key in ('cv_local_value','context_local_value')]
    paths += [ROOT/config[key].replace('value.pt',file) for key in ('cv_local_value','context_local_value') for file in ('value_support.json','report.json')]
    return paths

def freeze(name,acceptance=False):
    if file_sha256(Path(__file__))!=SOURCE_SHA:raise RuntimeError('parent source changed since launch')
    path=BASE/name
    if path.exists():
        lock=json.loads(path.read_text());verify_lock(ROOT,lock);return lock
    paths=source_paths()
    if acceptance:
        paths += [BASE/'success_acceptance.json',BASE/'validation_conditions.json',BASE/'success_selection_usage_validation.json']
        paths += [p for directory in BASE.glob('validation_*') if directory.is_dir() for p in directory.rglob('*.json')]
    value=dict(created_utc=now(),files={os.path.relpath(p,ROOT):file_sha256(p) for p in paths},
        fitting_or_calibration_on_TEST=False,critic_retrained=False,critic_lineage='frozen R08 heads and source-nested calibration',
        case_isolation='fresh_simulator_process_per_episode',
        previous_TEST_exposure='r04 complete/r05 partial and two already-seen official-reset-only probes; no fresh-unseen TEST claim',
        selection_gate='unchanged calibration thresholds; tasks with actual learned overrides require <=1/20 VAL loss; zero-override tasks require reviewed functional equivalence and no gain attribution')
    write(path,value);return value

def manifest(split):
    path=ROOT/('artifacts/framework/pipeline_v1/official_test_manifest.json' if split=='test' else f'artifacts/framework/protocol/{split}_manifest.json')
    return [row for row in json.loads(path.read_text())['episodes'] if row['task'] in TASKS]

def group_path(split,variant):return BASE/(split+'_physmani_v6' if variant=='physmani' else split+'_'+variant)

def records(split,variant):
    return [json.loads(path.read_text()) for path in sorted(group_path(split,variant).glob('pick*__episode*.json')) if not path.name.endswith('infrastructure_error.json')]

def update_group(split,variant):
    with REPORT_LOCK:
        rows=records(split,variant);expected=len(manifest(split))
        write(group_path(split,variant)/'report.json',dict(created_utc=now(),completed=len(rows),expected=expected,complete=len(rows)==expected,
            split=split,case_isolation='fresh_simulator_process_per_episode',formal_test_used=split=='test',
            per_task={task:dict(completed=sum(r['task']==task for r in rows),successes=sum(r['task']==task and r['official_task_success'] for r in rows)) for task in TASKS}))

def run_case(split,variant,row,lock):
    if (ROOT/'runtime/framework/success_r09_STOP').exists():raise InterruptedError('r09 STOP')
    verify_lock(ROOT,lock)
    name=row['task']+'__'+row['raw_episode_id'];directory=group_path(split,variant);directory.mkdir(parents=True,exist_ok=True)
    final_path=directory/(name+'.json')
    if final_path.exists():
        check=case_process_check(json.loads(final_path.read_text()),variant,lock['files'])
        if not check['accepted']:raise ValueError('invalid case resume: '+str(check['problems']))
        return
    child=directory/'cases'/name
    if child.exists() and any(child.iterdir()):
        # A partial previous child has no durable parent receipt. Preserve it and launch a genuinely new process/output.
        child=child/('attempt_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    child.mkdir(parents=True,exist_ok=True)
    baseline=variant=='physmani'
    evaluator='scripts/evaluate_physmani_aligned_r09_v6.py' if baseline else 'scripts/evaluate_aligned_r09.py'
    launcher='scripts/run_aligned_gpu.sh' if baseline else 'scripts/run_local_sim.sh'
    config='config/framework/aligned_protocol_r09_baseline_v6.json' if baseline else 'config/framework/aligned_protocol_r09.json'
    command=['bash',launcher,evaluator,'--config',config,'--split',split,'--task',row['task'],
        '--raw-episode-id',row['raw_episode_id'],'--episodes-per-task','0','--out',str(child.relative_to(ROOT))]
    if not baseline:
        predictor,ranking=variant.split('_',1);command += ['--predictor',predictor,'--ranking',ranking]
        if split=='test' and variant=='context_value' and row==next(r for r in manifest(split) if r['task']==row['task']):command += ['--videos','1']
    environment=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONHASHSEED='0')
    started=now()
    log_path=ROOT/'logs/framework'/f'aligned_r09_{split}_{variant}_{name}.log'
    with log_path.open('a') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        code=process.wait()
    if code:
        write(directory/(name+'.infrastructure_error.json'),dict(error=f'child exit {code}',child_directory=str(child.relative_to(ROOT)),log=str(log_path.relative_to(ROOT)),counted_as_policy_failure=False))
        raise RuntimeError(f'{split}/{variant}/{name} child exit {code}; inspect {log_path}')
    child_record=child/(name+'.json');raw=json.loads(child_record.read_text())
    if raw['task']!=row['task'] or raw['raw_episode_id']!=row['raw_episode_id'] or raw['split']!=split:raise ValueError('child returned wrong episode')
    inputs=child/(row['task']+'_inputs.json' if baseline else 'inputs.json')
    receipt=dict(version='fresh_process_r09_v1',child_evaluator=evaluator,child_evaluator_sha256=file_sha256(ROOT/evaluator),
        child_record_path=str(child_record.relative_to(ROOT)),child_record_sha256=file_sha256(child_record),
        child_inputs_path=str(inputs.relative_to(ROOT)),child_inputs_sha256=file_sha256(inputs),
        launch_command=command,launcher=launcher,parent_observed_process_id=process.pid,started_utc=started,finished_utc=now())
    result=dict(raw,case_isolation='fresh_simulator_process_per_episode',case_process_receipt=receipt)
    check=case_process_check(result,variant,lock['files'])
    if not check['accepted']:raise ValueError('child process evidence: '+str(check['problems']))
    write(final_path,result)
    for video in child.glob('*.mp4'):
        import shutil
        shutil.copy2(video,directory/video.name)
    update_group(split,variant);verify_lock(ROOT,lock)
    tolerances=json.loads((BASE/'acceptance_tolerances.json').read_text())
    for other_variant in VARIANTS:
        if other_variant==variant:continue
        other_path=group_path(split,other_variant)/(name+'.json')
        if not other_path.exists():continue
        other=json.loads(other_path.read_text())
        parity_check=compare_conditions(result,other,tolerances) if split=='test' else public_parity(result,other,tolerances)
        if not parity_check['accepted']:
            write(BASE/'initial_condition_fail_fast.json',dict(split=split,case=name,a=variant,b=other_variant,check=parity_check))
            (ROOT/'runtime/framework/success_r09_STOP').write_text('initial condition parity failed; no tolerance relaxation\n')
            raise RuntimeError('paired initial conditions failed: '+str(parity_check['problems']))
    print(json.dumps(dict(split=split,variant=variant,case=name,success=raw['official_task_success'])),flush=True)

def run_group(split,variant,lock,task=None):
    for row in manifest(split):
        if task is None or row['task']==task:run_case(split,variant,row,lock)
    update_group(split,variant)

def public_parity(a,b,tolerances):
    problems=[];x,y=a['public_initial_conditions'],b['public_initial_conditions']
    if x['information_setting']!=y['information_setting'] or x['sensor_protocol']!=y['sensor_protocol']:problems.append('sensor/information')
    problems += ['a_'+p for p in sensor_problems(x)]+['b_'+p for p in sensor_problems(y)]
    if not np.allclose(x['robot_joints'],y['robot_joints'],atol=tolerances['robot_joints_rad'],rtol=0):problems.append('robot_joints')
    px,py=np.array(x['robot_pose']),np.array(y['robot_pose'])
    if np.linalg.norm(px[:3]-py[:3])>tolerances['robot_position_m']:problems.append('robot_position')
    if (Rotation.from_quat(px[3:]).inv()*Rotation.from_quat(py[3:])).magnitude()>tolerances['robot_orientation_rad']:problems.append('robot_orientation')
    if set(x['cameras'])!=set(y['cameras']):problems.append('cameras')
    for name in x['cameras'].keys()&y['cameras'].keys():
        for field in ('intrinsics','camera_to_world'):
            if not np.allclose(x['cameras'][name][field],y['cameras'][name][field],atol=tolerances['camera_calibration_component'],rtol=0):problems.append(name+'_'+field)
    xm={s['name']:s for s in x['static_map']};ym={s['name']:s for s in y['static_map']}
    if xm.keys()!=ym.keys():problems.append('static_map_names')
    for name in xm.keys()&ym.keys():
        if xm[name]['mesh_sha256']!=ym[name]['mesh_sha256']:problems.append('mesh_'+name)
        if not np.allclose(xm[name]['pose'],ym[name]['pose'],atol=tolerances['fixture_pose_component'],rtol=0):problems.append('fixture_'+name)
    return dict(accepted=not problems,problems=problems,rgbd_bitwise_identical=x['rgbd_sha256']==y['rgbd_sha256'])

def parity(split,variants):
    tolerances=json.loads((BASE/'acceptance_tolerances.json').read_text());expected={(r['task'],r['raw_episode_id']) for r in manifest(split)}
    data={v:{(r['task'],r['raw_episode_id']):r for r in records(split,v)} for v in variants};checks=[];errors=[]
    for variant,values in data.items():
        if set(values)!=expected:errors.append(variant+' coverage')
    for a,b in itertools.combinations(variants,2):
        for key in sorted(expected&data[a].keys()&data[b].keys()):
            check=compare_conditions(data[a][key],data[b][key],tolerances) if split=='test' else public_parity(data[a][key],data[b][key],tolerances)
            checks.append(dict(a=a,b=b,key=key,**check))
            if not check['accepted']:errors.append(dict(a=a,b=b,key=key,problems=check['problems']))
    result=dict(created_utc=now(),passed=not errors,errors=errors,checks=checks,split=split,tolerances_unchanged=True)
    write(BASE/(split+'_conditions.json'),result);return result

def selection_usage(split):
    subprocess.run([sys.executable,'scripts/analyze_success_usage_r09.py','--split',split],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    return json.loads((BASE/f'success_selection_usage_{split}.json').read_text())

def acceptance():
    usage=selection_usage('validation');failures=[];details={}
    certificate=json.loads((BASE/'geometry_fallback_equivalence_review.json').read_text())
    for predictor in ('cv','context'):
        report=json.loads((BASE/f'{predictor}_success/report.json').read_text());details[predictor]={}
        for task in TASKS:
            g=sum(r['official_task_success'] for r in records('validation',predictor+'_geometry') if r['task']==task)
            v=sum(r['official_task_success'] for r in records('validation',predictor+'_value') if r['task']==task)
            cross=report['calibration_crossfit'][task];task_usage=usage['groups']['validation_'+predictor+'_value']['per_task'][task]
            overrides=task_usage['actual_override_selections']
            trace_complete=(task_usage['matching_issue_count']==0 and task_usage['unmatched_attempts']==0
                and task_usage['unknown_override_status_selections']==0
                and task_usage['matched_selection_events']==task_usage['candidate_attempts'])
            equivalent=(overrides==0 and trace_complete and certificate.get('conditional_selection_equivalence_review_passed',False))
            details[predictor][task]=dict(geometry_successes=g,value_successes=v,episodes=20,actual_learned_overrides=overrides,
                geometry_equivalence_used=equivalent,calibration_crossfit=cross,
                SR_difference_attributable_to_critic=False if equivalent else 'not proven by this small development sample')
            if not cross['valid'] or cross['metrics']['ece_5_bins']>.15:failures.append(predictor+' '+task+' calibration error')
            if cross['metrics']['brier']>cross['raw_metrics']['brier']+.01:failures.append(predictor+' '+task+' calibration Brier')
            if not equivalent and v<g-1:failures.append(predictor+' '+task+' selection VAL regression')
    if not usage['diagnostics_valid'] or not usage['coverage_complete']:failures.append('selection diagnostics incomplete/invalid')
    value=dict(passed=not failures,failures=failures,results=details,formal_test_used=False,
        gate_preregistered_before_fresh_VAL=True,interpretation='development acceptance; zero overrides imply no selection change and no SR gain claim; independent native variation is not a critic effect')
    write(BASE/'success_acceptance.json',value);return value

def report():subprocess.run([sys.executable,'scripts/report_aligned_r09.py'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)

def main():
    BASE.mkdir(parents=True,exist_ok=True)
    try:
        verify_lock(ROOT,json.loads((ROOT/'artifacts/framework/aligned_r08/frozen_inputs.json').read_text()))
        lock=freeze('frozen_inputs.json')
        status('fresh_case_validation','R08模型保持冻结；四组各40次VAL，每回合独立进程与固定重置种子，严格检查初始条件。')
        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(run_group,'validation',variant,lock) for variant in VARIANTS[:-1]]):future.result()
        condition=parity('validation',VARIANTS[:-1]);gate=acceptance()
        if not condition['passed'] or not gate['passed']:
            status('success_or_reset_needs_attention','完整VAL尚未通过模型/初始条件验收；正式TEST保持停止。',acceptance_failures=gate['failures'],condition_errors=condition['errors']);report();return
        status('baseline_validation','四组VAL通过；PhysMani发布策略在相同fresh-case设置完成40次VAL。')
        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(run_group,'validation','physmani',lock,task) for task in TASKS]):future.result()
        condition=parity('validation',VARIANTS)
        if not condition['passed']:
            status('baseline_reset_needs_attention','基线和四组VAL初始条件未通过严格配对，TEST停止。',condition_errors=condition['errors']);report();return
        comparison=freeze('comparison_v6_inputs.json',acceptance=True);verify_lock(ROOT,comparison)
        status('official_TEST','TRAIN与fresh-case完整VAL验收完成；五组共400次正式TEST，每回合独立进程，记录两段视频。')
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures=[pool.submit(run_group,'test',variant,comparison) for variant in VARIANTS[:-1]]
            futures += [pool.submit(run_group,'test','physmani',comparison,task) for task in TASKS]
            for future in as_completed(futures):future.result()
        condition=parity('test',VARIANTS);selection_usage('test')
        subprocess.run([sys.executable,'scripts/summarize_aligned_r09_v6.py'],cwd=ROOT,check=True)
        if not condition['passed']:raise RuntimeError('final full pairwise TEST initial conditions failed')
        verify_lock(ROOT,comparison)
        status('complete','TRAIN覆盖、完整VAL、400次正式TEST与两段视频完成，全部严格协议检查通过。',report='artifacts/framework/aligned_r09/final_report.json');report()
    except Exception as exc:
        status('needs_attention',str(exc),error=repr(exc));raise

if __name__=='__main__':main()
