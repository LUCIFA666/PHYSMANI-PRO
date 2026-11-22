"""Read-only current full3000 fit diagnosis, never a production rule or release.

Consumes already strictly audited canonical acquisitions; independently hashes
canonical/raw child/cache/commands and validates actual arrays/adapter events.
Historical full source closure is referenced, not re-declared as freshly audited.
Alternative endpoint coordinates are hypotheses only, not full curve refits.
"""
import argparse, ast, hashlib, json, sys, time
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.contracts import Pose
from our_mani.multitask import acquisition_lineage as a, artifact_fitting as fitting, interaction
from our_mani.multitask.public_roles import ColourFitConfig
from our_mani.multitask.training_view import validate_arrays
from diagnose_TRAIN_reference_times_r10 import endpoint_crossfit, stats, relative, pose_error, frame_at

BASE=ROOT/'artifacts/framework/multitask_r10'
NATIVE=ROOT.parent/'phsimani/PhysMani/third_party/RLBench/rlbench/tasks'

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def translation_kernel(task,body):
    """Execute only reviewed original arithmetic AST against nonnative fake IO."""
    source=(NATIVE/(task+'.py')).read_text();tree=ast.parse(source)
    module=next(n.module for n in tree.body if isinstance(n,ast.ImportFrom)
        and any(x.name=='compute_target_position' for x in n.names))
    helper=NATIVE/(module+'.py');ht=ast.parse(helper.read_text())
    names={'compute_target_position','cross_boundary'}
    if 'handle_boundary' in source:names.add('handle_boundary')
    nodes=[n for n in ht.body if isinstance(n,ast.FunctionDef) and n.name in names]
    a.require(len(nodes)==len(names),'actual arithmetic owners missing')
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(x,ast.FunctionDef) and x.name=='step' for x in n.body))
    step=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='init_task')
    area=[]
    for n in init.body:
        if not isinstance(n,ast.Assign) or len(n.targets)!=1:continue
        t=n.targets[0]
        if ((isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self' and t.attr=='area')
                or isinstance(t,ast.Name) and t.id=='target_size_xy'):area.append(n)
    a.require(len(area)==3 and not any(isinstance(n,(ast.Call,ast.Import,ast.ImportFrom)) for row in area for n in ast.walk(row)),
              'unknown pure area expressions')
    ns=dict(np=np,deepcopy=deepcopy,List=List,Tuple=Tuple,Shape=object)
    exec(compile(ast.Module(body=nodes+[step],type_ignores=[]),'<readonly-original-translation-step>','exec'),ns)
    holder=SimpleNamespace();exec(compile(ast.Module(body=area,type_ignores=[]),'<readonly-original-area>','exec'),dict(self=holder))
    return ns,holder.area,helper


def replay_translation(task,arrays,initial,body,kernel):
    ns,area,_=kernel;j=list(arrays['body_ids']).index(body);positions=arrays['target_positions'][:,j]
    class Body:
        def __init__(self,p):self.value=np.asarray(p,dtype=np.float32)
        def get_position(self):return self.value.astype(float).tolist()
        def set_position(self,p):self.value=np.asarray(p,dtype=np.float32)
    class Tip:
        index=0
        def get_position(self):return arrays['measured_tcp_xyzw'][self.index,:3].tolist()
    tip=Tip();dt=float(np.float32(.05));target=Body(positions[0])
    fake=SimpleNamespace(pyrep=SimpleNamespace(get_simulation_timestep=lambda:dt),area=area,
        target_state_list=[deepcopy(initial)],_bool_expert=False,t=0.,step_id=0,
        robot=SimpleNamespace(arm=SimpleNamespace(get_tip=lambda:tip)),_tip_z_history=[],
        goal_condition=SimpleNamespace(condition_met=lambda:(False,False)))
    setattr(fake,'target_button' if body=='button' else 'bin',target)
    states=[];maxerr=0.;flips=[]
    for i,p in enumerate(positions):
        error=float(np.linalg.norm(np.asarray(target.get_position())-p));maxerr=max(maxerr,error)
        a.require(error<=1e-6,f'original arithmetic translation replay failed {task} frame{i} error{error}')
        states.append((fake.t,deepcopy(fake.target_state_list[-1])))
        if i+1<len(positions):
            count=len(fake.target_state_list);tip.index=i;ns['step'](fake)
            if len(fake.target_state_list)>count:flips.append(i)
    return states,maxerr,flips


def summary_endpoint(rows,config):
    result=endpoint_crossfit(rows,config)
    for key in ('source_to_fold','residuals','parent_terminal_residuals'):result.pop(key,None)
    return result


def run(out):
    started=time.perf_counter();out=a.local(ROOT,out);a.require(not out.exists(),'fresh output required');out.mkdir(parents=True)
    files={}
    def checked(path,expected=None):
        path=(ROOT/Path(path)).resolve();a.require(path.is_relative_to(ROOT) or path.is_relative_to(NATIVE),'diagnostic input outside explicit readonly source roots');digest=a.file_sha(path)
        a.require(expected is None or digest==expected,'diagnosis source bytes changed: '+str(path))
        old=files.get(str(path));a.require(old is None or old==digest,'input SHA conflict');files[str(path)]=digest;return path
    report_path=checked(BASE/'TRAIN_artifacts_full_goalv4_semanticv6_v1/report.json')
    manifest_path=checked(report_path.parent/'TRAIN_fit_manifest.json');report=a.read(report_path);manifest=a.read(manifest_path)
    a.require(report['artifact_fitting_execution_complete'] is True and report['native_TRAIN_rollouts']==3000
        and report['native_TRAIN_parent_sources']==1400 and report['TEST_used'] is False,'actual full3000 fit required')
    config_path=checked(a.local(ROOT,manifest['fitting_configuration']['path']),manifest['fitting_configuration']['sha256'])
    config=a.read(config_path);ic=interaction.InteractionFitConfig(**config['interaction']);cc=ColourFitConfig(**config['colour'])
    a.require(ic.minimum_completed_parent_sources==8 and ic.maximum_position_tolerance_m==.03
        and ic.maximum_orientation_tolerance_rad==.15 and cc.minimum_sources_per_colour==8,'original gates changed')
    bundle_path=checked(BASE/'TRAIN_primary2800_aux200_bundle_v7/bundle.json');bundle=a.read(bundle_path)
    checked(BASE/'TRAIN_primary2800_aux200_bundle_v7/report.json')
    records={};entry_by_source={e['source_id']:e for e in manifest['entries']}
    a.require(len(entry_by_source)==3000,'duplicate fit entries')
    for ptr in (bundle['primary_manifest'],bundle['auxiliary_teacher_manifest']):
        m=a.read(checked(ptr['path'],ptr['sha256']))
        for row in m.get('case_receipts',m.get('completed_acquisitions',[])):
            rawptr=row.get('canonical_receipt',row)
            path=checked(rawptr['path'],rawptr['sha256']);r=a.read(path);entry=r['entry'];sid=entry['source_id']
            if 'actual_plan_sha256' in row:a.require(r['plan_sha256']==row['actual_plan_sha256'],'actual plan identity changed')
            if 'parent_receipt' in row:checked(row['parent_receipt']['path'],row['parent_receipt']['sha256'])
            a.require(sid in entry_by_source and entry==entry_by_source[sid] and sid not in records,'fit/acquisition identity changed')
            a.require(r['status']=='completed' and r['split']=='train' and r['case_id']==row['case_id'],'noncompleted canonical')
            process=r['parent_process_receipt'];child=a.read(checked(process['child_case_path'],process['child_case_sha256']))
            a.require(child=={k:v for k,v in r.items() if k!='parent_process_receipt'},'raw/canonical mismatch')
            records[sid]=(path,r)
    a.require(len(records)==3000,'full3000 canonical roster missing')
    pending=[t for t,r in report['per_task'].items() if r['interaction_status']=='pending_insufficient_or_invalid_TRAIN_supervision']
    a.require(len(pending)==8,'expected unchanged eight pending task identities')
    for module in (a,fitting,interaction):checked(module.__file__)
    checked(__file__);checked(ROOT/'scripts/diagnose_TRAIN_reference_times_r10.py')
    source_evidence={}
    for task in pending:
        p=checked(NATIVE/(task+'.py'));source_evidence[task]=dict(path=str(p),sha256=files[str(p)],
            current_readonly_derivation_not_historical_closure=True)
    checked(NATIVE/'place_cups_on_rotating_frame.py')
    kernels={}
    for t in ('push_moving_button','push_moving_button_high_speed','put_rubbish_in_moving_bin_high_speed'):
        kernels[t]=translation_kernel(t,'button' if t.startswith('push_') else 'bin');checked(kernels[t][2])
    tasks={};all_case_proofs=[]
    for task in pending:
        fitted=report['per_task'][task]['completed_skill_fitting'];admitted={(r['source_id'],r['command_id']) for r in fitted['curation'] if r['included_in_BC']}
        entries=[e for e in manifest['entries'] if e['task']==task and e['teacher_used']]
        a.require(len(entries)==len({e['parent_source_group'] for e in entries})==100,'all100 original parents required')
        comparisons=defaultdict(lambda:defaultdict(list));events=[];case_rows=[];public=defaultdict(list);forecast_errors=[];replay_frames=0
        reference_identity_mismatch=0;body_nearest_agree=0;replay_errors=[]
        for entry in entries:
            path,record=records[entry['source_id']];cache=checked(entry['cache'],entry['cache_sha256'])
            command_path=checked(entry['teacher_command_receipt'],record['files_sha256'][entry['teacher_command_receipt']])
            with np.load(cache,allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
            validate_arrays(arrays,entry,record)
            commands=a.read(command_path);case=dict(path=path,record=record,entry=entry,arrays=arrays,command_receipt=commands,
                command_receipt_sha256=files[str(command_path)],command_receipt_verified=True)
            source,audit=fitting.interaction_source(case);ledger={c['command_id']:c for c in commands['commands']};ids=list(arrays['body_ids'])
            initial=record['native_reset_receipt']['native_initial_snapshot']
            a.require(initial['native_task_source_sha256']==source_evidence[task]['sha256'],'original task source differs from readonly arithmetic owner')
            replay=None
            if task in kernels:
                replay,err,flips=replay_translation(task,arrays,initial['active_motion_schedule'],'button' if task.startswith('push_') else 'bin',kernels[task])
                replay_frames+=len(arrays['times']);replay_errors.append(err)
                for c in commands['commands']:
                    if task.startswith('push_') and c['raw_phase']=='press':delay=.25 if task.endswith('high_speed') else .5;body='button'
                    elif task.startswith('put_') and c['raw_phase']=='deposit':
                        previous=commands['commands'][c['command_id']-1] if c['command_id']>0 else None
                        delay=.5 if c['desired_opening']>=.5 else (0. if previous and previous['raw_stage']=='wp2' else 1.);body='bin'
                    else:continue
                    i=frame_at(source.times,c['generated_at']);tt,state=replay[i]
                    predicted=np.asarray(kernels[task][0]['compute_target_position'](t=tt+delay,t0=state['t0'],x0=state['x'],v0=state['v'],a0=state['a'],dt=float(np.float32(.05))))
                    unclipped=predicted.copy()
                    if task=='push_moving_button_high_speed':predicted=np.asarray(kernels[task][0]['handle_boundary'](predicted.tolist(),kernels[task][1]))
                    if task.startswith('put_') and delay==0.:predicted=arrays['target_positions'][i,ids.index('bin')].copy()
                    error=float(np.linalg.norm(predicted[:2]-np.asarray(c['raw_waypoint_xyzw'])[:2]))
                    if delay>0.:a.require(error<=1e-6,f'original forecast goal formula mismatch {task} {record["case_id"]} command{c["command_id"]} {error}')
                    forecast_errors.append(dict(case_id=record['case_id'],parent=source.grouping_source,command_id=c['command_id'],
                        delay_seconds=delay,desired_opening=c['desired_opening'],xy_error_m=error,raw_command_preserved=True,
                        branch_selected_from_original_prior_command_stage_not_outcome=True,
                        goal_is_source_arithmetic_forecast=(delay>0.),
                        current_waypoint_has_canonical_offset_from_body=(delay==0.),
                        boundary_clipping_distance_m=float(np.linalg.norm(predicted[:2]-unclipped[:2])) if task=='push_moving_button_high_speed' else 0.))
                    c['_diagnostic_forecast_position']=predicted;c['_diagnostic_delay']=delay
            for event in source.completion_events:
                c=ledger[event.origin_command_id];binding=event.binding;start=frame_at(source.times,event.attempt_start_time);end=frame_at(source.times,event.attempt_end_time)
                terminal=Pose.from_xyzw(event.terminal_tcp_xyzw);goal=Pose.from_xyzw(event.committed_world_waypoint_xyzw)
                current=Pose.from_xyzw(event.terminal_reference_xyzw);raw_key='|'.join((event.phase,binding.anchor_id,binding.reference_role,binding.reference_kind,binding.manipulated_role,event.reference_mode,binding.reference_time_basis))
                matches=[k for k in fitted['phase_fits'] if k.rsplit('|',1)[0]==raw_key.rsplit('|',1)[0]]
                a.require(len(matches)==1,'raw event has no unique actual fit partition');key=matches[0]
                issue=Pose(np.zeros(3)) if binding.reference_kind=='world' else Pose.from_xyzw(event.origin_measured_tcp_xyzw) if binding.reference_kind=='phase_origin' else Pose.from_xyzw(c['issued_reference_native_pose_xyzw'])
                if binding.reference_kind=='translation_only':issue=Pose(issue.position)
                options=dict(current=relative(current,terminal),issue=relative(issue,terminal),goal_relative_issue=relative(issue,goal),
                    actual_tracking=relative(goal,terminal),goal_relative_current=relative(current,goal))
                ev=dict(case_id=record['case_id'],source_id=source.source_id,parent=source.grouping_source,command_id=event.origin_command_id,
                    phase=event.phase,raw_stage=c['raw_stage'],reference_id=binding.reference_id,reference_kind=binding.reference_kind,
                    reference_mode=event.reference_mode,partition=key,raw_reference_time_basis=binding.reference_time_basis,originally_admitted_in_fit=(source.source_id,event.origin_command_id) in admitted,
                    duration_seconds=event.attempt_end_time-event.attempt_start_time,terminal_to_committed_goal=pose_error(terminal,goal),
                    completion_evidence=event.evidence,raw_command_completed=('command_completed_at' in c),
                    actual_arrival_observed=('measured_arrival_at' in c),
                    reference_issue_to_terminal=pose_error(current,issue),origin_and_terminal_frames=[start,end])
                if 'measured_arrival_at' in c:
                    arrival_index=frame_at(source.times,c['measured_arrival_at']);arrival=Pose.from_xyzw(arrays['measured_tcp_xyzw'][arrival_index])
                    ev['actual_measured_arrival_to_committed_goal']=pose_error(arrival,goal)
                    ev['actual_arrival_to_phase_terminal_motion']=pose_error(terminal,arrival)
                    ev['arrival_to_phase_terminal_seconds']=event.attempt_end_time-c['measured_arrival_at']
                    ev['joint_limit_recovery_ticks_after_arrival']=int(arrays['teacher_joint_limit_recovery'][arrival_index:end+1].sum())
                    ev['executed_after_arrival_stages']=dict(Counter(map(str,arrays['teacher_execution_stage'][arrival_index:end+1])))
                if '_diagnostic_forecast_position' in c:
                    forecast=Pose(c['_diagnostic_forecast_position']);options['source_declared_forecast_intent']=relative(forecast,terminal)
                    options['goal_relative_source_declared_forecast']=relative(forecast,goal);ev['source_declared_delay_seconds']=c['_diagnostic_delay']
                if binding.reference_id in ids:
                    j=ids.index(binding.reference_id);pv=bool(arrays['valid'][end,j]);po=bool(pv and arrays['rotation_observable'][end,j])
                    ev['public_reference_valid_at_terminal']=pv;ev['public_fullR_at_terminal']=po
                    if pv:public[event.phase+'_position'].append(float(np.linalg.norm(arrays['positions'][end,j]-arrays['target_positions'][end,j])))
                    if po:public[event.phase+'_orientation'].append(float(Rotation.from_matrix(arrays['rotations'][end,j]@arrays['target_rotations'][end,j].T).magnitude()))
                    public[event.phase+'_observable'].append(float(po));public[event.phase+'_valid'].append(float(pv))
                if task.startswith('remove_cups'):
                    cups=[x for x in ids if x.startswith('cup')];spokes=[x for x in ids if x.startswith('spoke')]
                    nearest=min(cups,key=lambda x:np.linalg.norm(arrays['target_positions'][start,ids.index(x)]-arrays['measured_tcp_xyzw'][start,:3]))
                    manipulated=c['manipulated_body'];body_nearest_agree+=int(nearest==manipulated);reference_identity_mismatch+=int(nearest!=manipulated)
                    cupi=ids.index(manipulated)
                    nearspoke=min(spokes,key=lambda x:np.linalg.norm(arrays['target_positions'][start,ids.index(x)]-arrays['target_positions'][start,cupi]))
                    si=ids.index(nearspoke);sr=Pose(arrays['target_positions'][start,si],arrays['target_rotations'][start,si])
                    options['nearest_spoke_issue']=relative(sr,terminal);options['goal_relative_nearest_spoke_issue']=relative(sr,goal)
                    framei=ids.index('frame');fr=Pose(arrays['target_positions'][start,framei],arrays['target_rotations'][start,framei])
                    options['frame_issue']=relative(fr,terminal);ev['nearest_cup_at_issue']=nearest;ev['nearest_spoke_to_manipulated_cup_at_issue']=nearspoke
                    if event.phase in ('approach','detach'):
                        previous=ledger.get(c['command_id']-1)
                        delay=(1. if previous and previous['raw_stage']=='wp0' else .5) if event.phase=='approach' else .5
                        yaw=float(np.asarray(initial['active_motion_schedule']['yaw_speed']).reshape(-1)[0])
                        delta=fr.rotation@Rotation.from_euler('z',np.deg2rad(yaw)*delay).as_matrix()@fr.rotation.T
                        def shifted(pose):return Pose(fr.position+delta@(pose.position-fr.position),delta@pose.rotation)
                        future_issue=shifted(issue)
                        degoal=Pose(fr.position+delta.T@(goal.position-fr.position),delta.T@goal.rotation)
                        options['hypothetical_rigid_frame_rotated_issue_reference']=relative(future_issue,terminal)
                        options['goal_relative_hypothetical_rigid_frame_rotated_issue']=relative(future_issue,goal)
                        options['hypothetical_deforecast_goal_relative_issue']=relative(issue,degoal)
                        ev['hypothetical_original_forecast_seconds']=delay
                        ev['waypoint_parent_hierarchy_not_recorded_so_rotation_hypothesis_not_source_proof']=True
                ev['coordinate_hypotheses']={name:dict(position=value['position'].tolist(),rotation_xyzw=Rotation.from_matrix(value['rotation']).as_quat().tolist()) for name,value in options.items()}
                events.append(ev)
                if ev['originally_admitted_in_fit']:
                    for name,value in options.items():comparisons[key][name].append(dict(source=source.grouping_source,case_id=record['case_id'],command_id=event.origin_command_id,progress=1.,terminal_event=True,opening=float(source.expert_opening[max(start,end-1)]),**value))
            row=dict(case_id=record['case_id'],parent=source.grouping_source,source_id=source.source_id,canonical_receipt=a.pointer(ROOT,path),
                cache=a.pointer(ROOT,cache),command_receipt=a.pointer(ROOT,command_path),actual_plan_sha256=record['plan_sha256'],
                all_raw_commands=len(commands['commands']),raw_completion_events=len(commands['phase_completion_events']),
                adapter_completion_events=len(source.completion_events),excluded_events=list(source.excluded_completion_events),
                commands_without_completion=len(commands['commands'])-len({e.origin_command_id for e in source.completion_events}),
                public_source_fields_are_not_interaction_fit_inputs=True,original_failed_censored_or_noevent_source_preserved=True)
            case_rows.append(row);all_case_proofs.append(row)
        partitions={key:dict(original_full_curve_fit=fitted['phase_fits'][key],
            endpoint_hypotheses_only={name:summary_endpoint(rows,ic) for name,rows in variants.items()}) for key,variants in comparisons.items()}
        tasks[task]=dict(actual_original100_teacher_cases=len(case_rows),all_failed_censored_sources_retained=True,
            original_failed_fit_keys=fitted['failed_fit_keys'],partitions=partitions,
            public_pose_vs_native_terminal={k:stats(v) for k,v in public.items()},
            source_arithmetic_translation_replay_frames=replay_frames,
            source_arithmetic_maximum_actual_track_replay_error_m=max(replay_errors,default=0.),
            replay_is_diagnostic_numerical_comparison_not_new_production_permission=True,
            original_forecast_goal_formula_checks=dict(commands=len(forecast_errors),error_distribution_m=stats([x['xy_error_m'] for x in forecast_errors if x['goal_is_source_arithmetic_forecast']]),
                current_waypoint_offset_xy_distribution_m=stats([x['xy_error_m'] for x in forecast_errors if not x['goal_is_source_arithmetic_forecast']]),
                arithmetic_forecast_commands=sum(x['goal_is_source_arithmetic_forecast'] for x in forecast_errors),
                delay_counts=dict(Counter(str(x['delay_seconds']) for x in forecast_errors)),all_raw_commands_preserved=True,
                formula_is_private_TRAIN_expert_intention_not_observed_future_dynamics=True),
            remove_nearest_cup_identity_matches=body_nearest_agree,remove_nearest_cup_identity_mismatches=reference_identity_mismatch)
        write(out/task/'events.json',events);write(out/task/'cases.json',case_rows);write(out/task/'translation_formula_checks.json',forecast_errors)
        write(out/task/'summary.json',tasks[task]);print(task,len(case_rows),'parents',flush=True)
    # All3000 current RGBD annotations, replicas remain parent grouped.
    color_cases=[]
    for path,record in records.values():
        names=[p for p in record['files_sha256'] if Path(p).name=='TRAIN_color_annotations.json'];a.require(len(names)==1,'one current colour annotation required')
        checked(names[0],record['files_sha256'][names[0]]);color_cases.append(dict(path=path,record=record,entry=record['entry']))
    samples,excluded=fitting.colour_samples(ROOT,color_cases,cc)
    support=defaultdict(set);bytask=defaultdict(lambda:defaultdict(set));byrole=defaultdict(lambda:defaultdict(set))
    for sample in samples:
        support[sample.colour_name].add(sample.source_group);bytask[sample.colour_name][sample.task_name].add(sample.source_group);byrole[sample.colour_name][sample.role_id].add(sample.source_group)
    previous=report['colour_parent_crossfit'];a.require({k:sorted(v) for k,v in support.items()}==previous['public_colour_class_parent_support'] and excluded==report['excluded_colour_samples'],'current colour support changed since full14 fit')
    color=dict(actual3000_annotation_cases=len(color_cases),current_RGBD_samples=len(samples),excluded=excluded,
        distinct_parent_support={k:len(v) for k,v in support.items()},
        support_by_task={k:{t:len(v) for t,v in d.items()} for k,d in bytask.items()},
        support_by_role={k:{t:len(v) for t,v in d.items()} for k,d in byrole.items()},
        actual_fold_fit_support=[x['fit_parent_support_by_colour'] for x in previous['folds']],
        hard_minimum_parents=2*cc.minimum_sources_per_colour,
        mathematically_insufficient_classes=[k for k,v in support.items() if len(v)<2*cc.minimum_sources_per_colour],
        original_valid=previous['valid'],original_passed=previous['passed'],original_macro_score=previous['macro_parent_correct_and_supported_fraction'],
        no_accuracy_claim_when_support_gate_prevents_cv=True,same_parent_extra_replicas_do_not_increase_independent_support=True,
        original_parent_fold_assignment_unchanged=True)
    # Identify whether yellow's7/9 is a purely feasible local whole-parent exchange.
    folds=previous['parent_to_fold'];labels=defaultdict(set)
    for name,parents in support.items():
        for parent in parents:labels[parent].add(name)
    counts=[dict(x['fit_parent_support_by_colour']) for x in previous['folds']]
    candidates=[]
    for parent in sorted(support.get('yellow',set())):
        old=folds[parent];new=1-old;updated=[dict(c) for c in counts]
        for label in labels[parent]:updated[old][label]+=1;updated[new][label]-=1
        oldfails=sum(v<8 for c in counts for k,v in c.items() if k!='orange');newfails=sum(v<8 for c in updated for k,v in c.items() if k!='orange')
        if newfails<oldfails:candidates.append(dict(parent=parent,from_fold=old,to_fold=new,all_parent_labels=sorted(labels[parent]),
            hypothetical_fit_support_yellow=[c['yellow'] for c in updated],no_nonorange_class_below8=not newfails))
    color['readonly_feasible_yellow_whole_parent_move_candidates']=candidates
    write(out/'colour.json',color)
    # Every byte directly read in this local diagnosis must remain unchanged.
    for p,sha in files.items():a.require(a.file_sha(p)==sha,'diagnosis input changed at final direct rehash: '+p)
    value=dict(schema='r10_full14_pending_reference_readonly_diagnosis_v1',started_utc=datetime.now(timezone.utc).isoformat(),
        actual_full3000_fit=a.pointer(ROOT,report_path),actual_fit_manifest=a.pointer(ROOT,manifest_path),
        complete3000_audit_referenced_not_repeated=a.pointer(ROOT,BASE/'TRAIN_primary2800_aux200_bundle_v7/report.json'),
        tasks=tasks,colour=color,source_evidence=source_evidence,verified_files_sha256=files,
        target_tasks=8,actual_original_teacher_cases=800,original_parent_sources=800,
        endpoint_hypotheses_are_not_full_curve_fits=True,production_rules_or_sources_changed=False,
        native_or_GPU_started=False,VAL_or_TEST_used=False,formal_model_or_policy_release=False,
        thresholds_or_parents_or_outliers_relaxed=False,seconds=time.perf_counter()-started)
    write(out/'report.json',value);print('REPORT',out/'report.json',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();run(args.out)
