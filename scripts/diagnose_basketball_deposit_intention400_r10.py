"""TRAIN-only source-declared hoop intention/full-curve diagnostic on all400."""
import argparse
import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
from typing import List,Tuple
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.contracts import Pose
from our_mani.multitask import acquisition_lineage as a,artifact_fitting as adapter,interaction


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(interaction._json(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def kernel(source,helper):
    """Retain the original hoop-only independent statements, without native."""
    tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)
        and any(isinstance(m,ast.FunctionDef) and m.name=='step' for m in n.body))
    step=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
    index=next(i for i,n in enumerate(step.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple)
        and [v.id for v in n.targets[0].elts]==['hoop_target_position','hoop_target_velocity'])
    suffix=[n for n in step.body if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Attribute)
        and n.target.attr in ('step_id','t')]
    selected=step.body[:3]+step.body[index:index+3]+suffix
    replay=ast.FunctionDef(name='step',args=step.args,body=selected,decorator_list=[],returns=step.returns)
    ast.fix_missing_locations(replay)
    helpers=[n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.FunctionDef)
        and n.name in ('compute_target_position','cross_boundary')]
    a.require(len(helpers)==2 and len(suffix)==2,'source independent hoop/time kernel changed')
    expert=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_expert_info')
    wp3calls=[n for n in ast.walk(expert) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
        and n.func.id=='compute_target_position' and any(k.arg=='x0' and isinstance(k.value,ast.Subscript)
            and ast.unparse(k.value)=="hoop_target_state_dict['wp3_x']" for k in n.keywords)]
    a.require(len(wp3calls)==2 and all(any(k.arg=='t' and ast.unparse(k.value)=='task.t + t_delay' for k in c.keywords)
        for c in wp3calls),'both closed/open deposit branches must use original wp3 future intention')
    namespace=dict(np=np,deepcopy=deepcopy,List=List,Tuple=Tuple,Shape=object)
    exec(compile(ast.Module(body=helpers+[replay],type_ignores=[]),'<source-independent-hoop-TRAIN-kernel>','exec'),namespace)
    return namespace,dict(source_AST_hoop_statements_sha256=a.canonical_sha([ast.dump(n,include_attributes=False) for n in selected]),
        closed_and_open_deposit_formula='wp3_x + v*(task.t+0.5-t0) + 0.5*a*(task.t+0.5-t0)**2',
        actual_future_body_observation_used_as_intention_label=False,source_only_pure_hoop_step=True,
        omitted_ball_branch_has_no_effect_on_hoop_driver=True)


def replay(arrays,initial,namespace):
    j=list(arrays['body_ids']).index('hoop');positions=arrays['target_positions'][:,j]
    ball,hoop=deepcopy(initial);dt=float(np.float32(.05));offset=np.asarray(hoop['wp3_x'])-positions[0]
    class Hoop:
        value=np.asarray(positions[0],np.float32)
        def get_position(self):return self.value.astype(float).tolist()
        def set_position(self,v):self.value=np.asarray(v,np.float32)
    obj=Hoop()
    waypoint=SimpleNamespace(get_position=lambda:(obj.value.astype(float)+offset).astype(np.float32).astype(float).tolist())
    area=[.3,-.5,.8,.6,.5,.8];size=[.15,.05]
    area=[area[0]+size[0]/2,area[1]+size[1]/2,area[2],area[3]-size[0]/2,area[4]-size[1]/2,area[5]]
    fake=SimpleNamespace(pyrep=SimpleNamespace(get_simulation_timestep=lambda:dt),target_state_list=[(ball,hoop)],
        hoop=obj,wp3=waypoint,hoop_area=area,t=0.,step_id=0)
    states=[];maximum=0.
    for i in range(len(positions)):
        error=float(np.linalg.norm(np.asarray(obj.get_position())-positions[i]));maximum=max(maximum,error)
        a.require(error==0.,'hoop pure source replay differs from actual frame '+str(i))
        states.append((fake.t,deepcopy(fake.target_state_list[-1][1])))
        if i+1<len(positions):namespace['step'](fake)
    return states,maximum


def numeric(rows,durations,config):
    grid=np.linspace(0,1,config.progress_knots);parents=sorted({r['source'] for r in rows});folds={p:i%2 for i,p in enumerate(parents)}
    a.require(len(parents)>=8,'draft cannot relax original eight-parent support')
    errors=[];term=[]
    for fold in (0,1):
        train=[r for r in rows if folds[r['source']]!=fold];held=[r for r in rows if folds[r['source']]==fold]
        p,r,o=interaction._fit_completed_curve(train,grid)
        for v in held:
            x=np.asarray([np.interp(v['progress'],grid,p[:,k]) for k in range(3)])
            rot=Slerp(grid,Rotation.from_matrix(r))([v['progress']]).as_matrix()[0]
            pe=float(np.linalg.norm(x-v['position']));re=float(Rotation.from_matrix(rot@v['rotation'].T).magnitude())
            errors.append((v['source'],pe**2+re**2))
            if v['terminal_event']:term.append(dict(source=v['source'],attempt_id=v['attempt_id'],position=pe,orientation=re))
    byparent={p:{k:float(np.median([r[k] for r in term if r['source']==p])) for k in ('position','orientation')} for p in parents}
    pt=max(config.minimum_position_tolerance_m,float(np.quantile([r['position'] for r in byparent.values()],config.tolerance_quantile)))
    rt=max(config.minimum_orientation_tolerance_rad,float(np.quantile([r['orientation'] for r in byparent.values()],config.tolerance_quantile)))
    p,r,o=interaction._fit_completed_curve(rows,grid);d=[float(np.median([v for group,v in durations if group==parent])) for parent in parents]
    return dict(passed=pt<=.03 and rt<=.15,completed_parent_count=len(parents),completed_attempts=len(term),actual_curve_rows=len(rows),
        source_to_fold=folds,terminal_parent_residuals=byparent,position_tolerance_m=pt,orientation_tolerance_rad=rt,
        grouped_full_curve_CV=float(np.mean([np.mean([x for parent,x in errors if parent==p]) for p in parents])),
        duration_seconds=float(np.median(d)),duration_budget_seconds=max(float(np.median(d)),float(np.quantile(d,config.duration_quantile))),
        fitted_curve=dict(progress=grid,positions=p,rotations=r,opening=o),lag_seconds=0.,numeric_gates_unchanged_8_03_15=True)


def run(fit_path,consumer_path,output):
    begun=time.perf_counter();output=a.local(ROOT,output);a.require(not output.exists(),'fresh400 deposit diagnostic required');output.mkdir()
    files={};fit=a.read(a.checked_pointer(ROOT,a.pointer(ROOT,a.local(ROOT,fit_path)),files))
    consumer=a.read(a.checked_pointer(ROOT,a.pointer(ROOT,a.local(ROOT,consumer_path)),files))
    a.require(consumer['actual_teacher_sources_rebuilt']==400 and consumer['producer_consumer_all_source_proofs_identical'] is True,
        'completed strict actual400 consumer proof required before numerical intention diagnostic')
    manifest=a.read(a.checked_pointer(ROOT,fit['prefit400_source_proof'],files));configuration=a.read(a.checked_pointer(ROOT,fit['configuration'],files))
    config=interaction.InteractionFitConfig(**configuration['interaction']);native=Path('/home/ubuntu/桌面/phsimani/PhysMani/third_party/RLBench/rlbench/tasks')
    helper=native/'reach_single_moving_target_on_the_table_high_speed.py';archived={};kernels={};formulas={}
    for task in fit['tasks']:
        for p in (native/(task+'.py'),helper):
            if str(p) in archived:continue
            q=output/'readonly_current_source'/p.name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
            archived[str(p)]=dict(archive=a.pointer(ROOT,q),sha256=a.file_sha(p),historical_execution_dependency_claimed=False)
        kernels[task],formulas[task]=kernel(a.local(ROOT,archived[str(native/(task+'.py'))]['archive']['path']),
            a.local(ROOT,archived[str(helper)]['archive']['path']))
    groups={t:dict(rows=[],durations=[],commands=[],sources=[],frames=0,max_goal_error=0.) for t in fit['tasks']}
    for row in manifest['case_proofs']:
        path=a.checked_pointer(ROOT,row['canonical_receipt'],files);record=a.read(path);task=record['entry']['task'];group=groups[task]
        with np.load(a.checked_pointer(ROOT,row['cache'],files),allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        commands=a.read(a.checked_pointer(ROOT,row['command_receipt'],files))
        case=dict(path=path,record=record,entry=record['entry'],arrays=arrays,command_receipt=commands,
            command_receipt_verified=True,command_receipt_sha256=row['command_receipt']['sha256'])
        source=adapter.interaction_source(case)[0];namespace=kernels[task]
        states,error=replay(arrays,record['native_reset_receipt']['native_initial_snapshot']['active_motion_schedule'],namespace)
        group['frames']+=len(states);group['sources'].append(dict(source_id=source.source_id,parent=source.grouping_source,kind=row['kind'],
            raw_binding=row['canonical_receipt'],all_raw_events_retained=True,source_replay_error_m=error))
        events={e.origin_command_id:e for e in source.completion_events if e.phase=='deposit'}
        for command in source.command_ledger:
            if command['raw_phase']!='deposit':continue
            cid=command['command_id'];i=int(round(command['generated_at']/.05));t,state=states[i]
            wp3=np.asarray(namespace['compute_target_position'](t=t+.5,t0=state['t0'],x0=state['wp3_x'],v0=state['v'],a0=state['a']))
            forecast=np.asarray(namespace['compute_target_position'](t=t+.5,t0=state['t0'],x0=state['x'],v0=state['v'],a0=state['a']))
            goal_error=float(np.linalg.norm(wp3-np.asarray(command['raw_waypoint_xyzw'])[:3]));group['max_goal_error']=max(group['max_goal_error'],goal_error)
            a.require(goal_error<1e-6,'source-declared closed/open wp3 goal does not match actual committed command')
            event=events.get(cid);entry=dict(source_id=source.source_id,parent=source.grouping_source,command_id=cid,
                opening=command['desired_opening'],issued_at=command['generated_at'],native_task_time=t,raw_goal_error_m=goal_error,
                predicted_body_reference_xyz=forecast.tolist(),original_committed_goal_xyz=command['raw_waypoint_xyzw'][:3],
                explicit_source_intention_not_actual_future_observation=True,original_current_terminal_metric=None if event is None else event.terminal_reference_xyzw.tolist(),
                completed_event=event is not None,right_censored=event is None,executed_rows=0)
            group['commands'].append(entry)
            if event is None:continue
            binding=event.binding;duration=event.attempt_end_time-event.attempt_start_time;attempt=source.source_id+':command'+str(cid)
            indices=[int(j) for j in np.flatnonzero(source.command_ids==cid) if source.references[j]==binding
                and event.attempt_start_time-1e-8<=source.times[j]<event.attempt_end_time-1e-8]
            rows=[]
            for j in indices:
                try:source.body_tracks[binding.reference_id].at(source.times[j],orientation_required=False,
                    same_mode_as=event.reference_mode,not_before=source.times[j])
                except interaction.InteractionArtifactError:continue
                tcp=Pose.from_xyzw(source.curve_tcp_xyzw[j]);rows.append(dict(source=source.grouping_source,attempt_id=attempt,
                    progress=float((source.times[j]-event.attempt_start_time)/duration),position=tcp.position-forecast,
                    rotation=tcp.rotation,opening=source.expert_opening[j],terminal_event=False))
            a.require(rows,'completed deposit has no valid same-mode raw curve')
            tcp=Pose.from_xyzw(event.terminal_tcp_xyzw);rows.append(dict(source=source.grouping_source,attempt_id=attempt,progress=1.,
                position=tcp.position-forecast,rotation=tcp.rotation,opening=source.expert_opening[indices[-1]],terminal_event=True))
            entry['executed_rows']=len(rows)-1;group['rows'].extend(rows);group['durations'].append((source.grouping_source,duration))
    results={}
    for task,group in groups.items():
        a.require(len(group['sources'])==200 and len({r['parent'] for r in group['sources']})==100,'all200 rollouts/100original parents per task required')
        numeric_result=numeric(group['rows'],group['durations'],config)
        old=a.read(a.checked_pointer(ROOT,fit['tasks'][task]['detailed_fit_report'],files))
        previous=next(v for k,v in old['phase_fits'].items() if k.startswith('deposit|'))
        a.require(set(numeric_result['terminal_parent_residuals'])==set(previous['terminal_parent_residuals']),
            'all newly failing/tail parents must remain in the intention fit')
        detail=output/task/'deposit_full_curve_intention_fit.json';write(detail,numeric_result)
        sourceproof=output/task/'all200_sources_and_all_deposit_commands.json';write(sourceproof,dict(sources=group['sources'],commands=group['commands'],formula=formulas[task]))
        results[task]=dict(real_raw_rollouts=200,original_parent_groups=100,all_hoop_frames=group['frames'],exact_hoop_track_error_m=0.,
            all_deposit_commands=len(group['commands']),source_goal_maximum_error_m=group['max_goal_error'],
            original_current_metric_cap_m=previous['position_tolerance_m'],original_completed_parents=previous['completed_parent_count'],
            intention_fit={k:v for k,v in numeric_result.items() if k!='fitted_curve'},full_curve_report=a.pointer(ROOT,detail),
            all_source_command_proof=a.pointer(ROOT,sourceproof),no_tail_or_failed_source_removed=True)
    for name,item in archived.items():a.require(a.file_sha(name)==item['sha256'],'readonly source changed during diagnostic')
    report=dict(schema='r10_basketball_original200_aux200_deposit_source_intention_draft_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter()-begun,original400_fit=a.pointer(ROOT,a.local(ROOT,fit_path)),strict400_consumer=a.pointer(ROOT,a.local(ROOT,consumer_path)),
        source=archived,tasks=results,all400_sources_and_original200_parent_groups_retained=True,
        only_TRAIN_recorded_initial_driver_and_original_committed_goals_used=True,actual_future_observations_not_intention_labels=True,
        runtime_forecast_accuracy_or_closed_loop_success_proven=False,production_goal_core_or_profile_modified=False,
        native_GPU_or_collection_started=False,production_permission_or_library_exported=False,verified_files_sha256=files)
    path=output/'report.json';write(path,report)
    return dict(report=a.pointer(ROOT,path),tasks={t:{k:v for k,v in r.items() if k!='intention_fit'}|dict(intention_fit={k:v for k,v in r['intention_fit'].items() if k not in ('terminal_parent_residuals','source_to_fold')}) for t,r in results.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit-report',required=True);p.add_argument('--consumer-report',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();print(json.dumps(run(args.fit_report,args.consumer_report,args.out),ensure_ascii=False))
