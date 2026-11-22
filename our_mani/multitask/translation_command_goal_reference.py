"""Explicit TRAIN-only button approach and high-bin command intention.

The original task/math files are readonly current derivation snapshots. They
are separately checked against actual task archives, never retroactively
claimed as a complete historical arithmetic closure. No native imports occur.
"""
import ast
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List,Tuple

import numpy as np

from ..contracts import Pose
from . import acquisition_lineage as a

SCHEMA='r10_TRAIN_translation_command_intention_v1'
PERMISSION_SCHEMA='r10_TRAIN_translation_command_intention_permission_v1'
TASKS=('push_moving_button','push_moving_button_high_speed','put_rubbish_in_moving_bin_high_speed')
HIGH_BIN='put_rubbish_in_moving_bin_high_speed'
FORECAST='original_expert_high_bin_unreflected_forecast'
SELECTOR='original_expert_high_bin_current_then_closed1_open_half_forecast'


def require(value,message):
    if not value:raise ValueError('translation command-goal: '+message)


@dataclass(frozen=True)
class TranslationPermission:
    root:str
    pointer:dict
    kernel_sources:dict
    verified_files_sha256:dict
    production_fitting_enabled:bool=False


def load_permission(root,pointer,*,purpose):
    require(purpose in ('CPU_diagnostic','production_fitting'),'TRAIN permission purpose required')
    root=Path(root).resolve();files={};value=a.read(a.checked_pointer(root,pointer,files))
    require(set(value)=={'schema','TRAIN_only','VAL_used','TEST_used','production_fitting_enabled','tasks','helper',
        'kernel_sources','readonly_current_source_review','actual_original100_evidence'},'unknown permission field')
    require(value['schema']==PERMISSION_SCHEMA and value['TRAIN_only'] is True and value['VAL_used'] is False
        and value['TEST_used'] is False and value['tasks']==list(TASKS)
        and value['production_fitting_enabled'] is (purpose=='production_fitting'),'permission scope changed')
    require(a.checked_pointer(root,value['helper'],files)==Path(__file__).resolve(),'translation helper binding changed')
    keys=set(TASKS)|{'motion_helper','pyrep_getter','sim_getter','C_ABI_header','C_ABI_builder'}
    require(set(value['kernel_sources'])==keys,'complete explicit translation task/math/ABI source owners required')
    for ptr in value['kernel_sources'].values():a.checked_pointer(root,ptr,files)
    review=a.read(a.checked_pointer(root,value['readonly_current_source_review'],files))
    require(review['schema']=='r10_translation_readonly_source_derivation_review_v1'
        and review['kernel_sources']==value['kernel_sources'] and review['historical_execution_dependency_claimed'] is False
        and review['new_TRAIN_derivation_sources_explicitly_bound'] is True,'source snapshots must retain current derivation gap')
    evidence=a.read(a.checked_pointer(root,value['actual_original100_evidence'],files))
    require(evidence['schema']=='r10_full14_pending_reference_readonly_diagnosis_v1'
        and evidence['actual_original_teacher_cases']==800 and evidence['VAL_or_TEST_used'] is False
        and evidence['thresholds_or_parents_or_outliers_relaxed'] is False,'actual original100 per task evidence required')
    for task in TASKS:
        row=evidence['tasks'][task]
        require(row['actual_original100_teacher_cases']==100 and row['all_failed_censored_sources_retained'] is True
            and row['source_arithmetic_maximum_actual_track_replay_error_m']<=1e-6,'all100 numerical source replay required')
    high=evidence['tasks'][HIGH_BIN]['original_forecast_goal_formula_checks']
    require(high['commands']==739 and high['arithmetic_forecast_commands']==639
        and high['delay_counts']=={'0.0':100,'1.0':606,'0.5':33}
        and high['error_distribution_m']['maximum']==0.,'original739 current/1/.5 command proof required')
    header=a.local(root,value['kernel_sources']['C_ABI_header']['path']).read_text()
    builder=a.local(root,value['kernel_sources']['C_ABI_builder']['path']).read_text()
    require('typedef float simFloat;' in header and 'simFloat simGetSimulationTimeStep();' in header
        and 'typedef float simFloat;' in builder and 'simFloat simGetSimulationTimeStep();' in builder,'binary32 native clock ABI changed')
    return TranslationPermission(str(root),pointer,value['kernel_sources'],files,purpose=='production_fitting')


def _kernel(permission,task):
    require(task in TASKS,'unknown translation task')
    root=Path(permission.root);files={}
    tree=ast.parse(a.checked_pointer(root,permission.kernel_sources[task],files).read_text())
    ht=ast.parse(a.checked_pointer(root,permission.kernel_sources['motion_helper'],files).read_text())
    funcs=[n for n in ht.body if isinstance(n,ast.FunctionDef) and n.name in ('compute_target_position','cross_boundary')]
    require(len(funcs)==2,'exact original arithmetic owners required')
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(x,ast.FunctionDef) and x.name=='step' for x in n.body))
    step=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='init_task')
    area=[]
    for node in init.body:
        if not isinstance(node,ast.Assign) or len(node.targets)!=1:continue
        target=node.targets[0]
        if ((isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id=='self' and target.attr=='area')
                or isinstance(target,ast.Name) and target.id=='target_size_xy'):area.append(node)
    require(len(area)==3 and not any(isinstance(n,(ast.Call,ast.Import,ast.ImportFrom)) for row in area for n in ast.walk(row)),
        'original area arithmetic must stay pure')
    expert=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_expert_info')
    if task==HIGH_BIN:
        forecasts=[n for n in ast.walk(expert) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='t_delay' for t in n.targets)]
        require({float(n.value.value) for n in forecasts if isinstance(n.value,ast.Constant)}=={0.,.5,1.},'source delay set changed')
        branches=[n for n in expert.body if isinstance(n,ast.If)]
        forecasts=[]
        for branch in branches:
            while isinstance(branch,ast.If):
                calls=[n for row in branch.body for n in ast.walk(row) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='compute_target_position']
                if calls:
                    delays=[n.value.value for row in branch.body for n in ast.walk(row) if isinstance(n,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id=='t_delay' for t in n.targets) and isinstance(n.value,ast.Constant)]
                    openings=[n.value.value for row in branch.body for n in ast.walk(row) if isinstance(n,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id=='open' for t in n.targets) and isinstance(n.value,ast.Constant)]
                    forecasts.append((delays,openings,ast.unparse(branch.test)))
                branch=branch.orelse[0] if len(branch.orelse)==1 and isinstance(branch.orelse[0],ast.If) else None
        require(len(forecasts)==2 and forecasts[0][0]==[1.] and forecasts[0][1]==[0]
            and forecasts[1][0]==[.5] and forecasts[1][1]==[1]
            and 'dist_to_wp3 > th_wp3' in forecasts[0][2] and 'dist_to_wp3 <= th_wp3' in forecasts[1][2],
            'original closed1/open.5 source branches changed')
    else:
        # wp0 approach has no forecast; press is explicitly outside this rule.
        first=next(n for n in expert.body if isinstance(n,ast.If))
        require("stage == 'wp0'" in ast.unparse(first.test)
            and any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='t_delay' for t in n.targets)
                and isinstance(n.value,ast.Constant) and n.value.value==0. for n in first.body)
            and any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='eepose' for t in n.targets)
                and isinstance(n.value,ast.Name) and n.value.id=='wp0_pose' for n in first.body),'button approach source scope changed')
    ns=dict(np=np,deepcopy=deepcopy,List=List,Tuple=Tuple,Shape=object)
    exec(compile(ast.Module(body=funcs+[step],type_ignores=[]),'<bound-TRAIN-translation-arithmetic>','exec'),ns)
    holder=SimpleNamespace();exec(compile(ast.Module(body=area,type_ignores=[]),'<bound-TRAIN-area>','exec'),dict(self=holder))
    return ns,holder.area


def replay_states(case,permission):
    task=case['entry']['task'];arrays=case['arrays'];body='bin' if task==HIGH_BIN else 'button';j=list(arrays['body_ids']).index(body)
    ns,area=_kernel(permission,task);positions=arrays['target_positions'][:,j];initial=case['record']['native_reset_receipt']['native_initial_snapshot']
    require(initial['native_task_source_sha256']==permission.kernel_sources[task]['sha256'],'new arithmetic task source differs from actual acquisition archive')
    class Body:
        def __init__(self,p):self.p=np.asarray(p,dtype=np.float32)
        def get_position(self):return self.p.astype(float).tolist()
        def set_position(self,p):self.p=np.asarray(p,dtype=np.float32)
    class Tip:
        index=0
        def get_position(self):return arrays['measured_tcp_xyzw'][self.index,:3].tolist()
    tip=Tip();native=Body(positions[0]);dt=float(np.float32(.05))
    fake=SimpleNamespace(pyrep=SimpleNamespace(get_simulation_timestep=lambda:dt),target_state_list=[deepcopy(initial['active_motion_schedule'])],
        area=area,_bool_expert=False,t=0.,step_id=0,robot=SimpleNamespace(arm=SimpleNamespace(get_tip=lambda:tip)),
        _tip_z_history=[],goal_condition=SimpleNamespace(condition_met=lambda:(False,False)))
    setattr(fake,'bin' if task==HIGH_BIN else 'target_button',native)
    states=[];errors=[]
    for i,actual in enumerate(positions):
        error=float(np.linalg.norm(np.asarray(native.get_position())-actual));errors.append(error)
        require(error<=1e-6,'original source arithmetic disagrees with actual TRAIN body trajectory')
        states.append((fake.t,deepcopy(fake.target_state_list[-1])))
        if i+1<len(positions):tip.index=i;ns['step'](fake)
    return states,ns,dict(actual_body_frames=len(positions),maximum_body_replay_error_m=max(errors),
        body_replay_is_numerical_TRAIN_validation_not_actual_future_prediction=True)


def derive_source(case,source,rule,permission,*,original_verification=None):
    from . import command_goal_reference as goal
    from .interaction import TRAINReferenceBinding
    require(isinstance(permission,TranslationPermission) and source.task_name in TASKS,'explicit translation TRAIN permission required')
    original=original_verification or goal.verify_original_source(case,source,rule)
    require(isinstance(original,goal.TRAINCommandGoalEvidence) and not original.references
        and original.source_id==source.source_id and original.parent_source_group==source.grouping_source,'strict original raw source required first')
    arrays=case['arrays'];ids=list(arrays['body_ids']);task=source.task_name;body='bin' if task==HIGH_BIN else 'button';j=ids.index(body)
    states,ns,replay=replay_states(case,permission);references=[];pending=[];counts={};errors=[]
    events={e['origin_command_id']:e for e in case['command_receipt']['phase_completion_events']};ledger={c['command_id']:c for c in source.command_ledger}
    phase='deposit' if task==HIGH_BIN else 'approach'
    for command in source.command_ledger:
        if command['raw_phase']!=phase:continue
        cid=command['command_id'];found=np.flatnonzero(np.isclose(source.times,command['generated_at'],atol=1e-8,rtol=0));require(len(found)==1,'unique actual issue clock required');i=int(found[0])
        require(command['reference_body']==body and command['reference_kind']=='translation_only'
            and command['issued_reference_motion_mode']=='external','original translation binding or mode changed')
        if not bool(arrays['native_pose_valid'][i,j] and arrays['mode_valid'][i,j]):
            pending.append(dict(command_id=cid,reason='original issue pose/mode mask invalid',raw_source_preserved=True));continue
        issue=Pose(np.asarray(command['issued_reference_native_pose_xyzw'])[:3]);require(np.array_equal(issue.position,arrays['target_positions'][i,j]),'original issued translation differs from raw acquired pose')
        intent=issue;kind=goal.ISSUE;delay=0.;raw_goal=np.asarray(command['raw_waypoint_xyzw']);previous=ledger.get(cid-1)
        if task==HIGH_BIN:
            require(command['raw_stage']=='wp3','high-bin deposit must retain original wp3')
            delay=.5 if command['desired_opening']>=.5 else (0. if previous and previous['raw_stage']=='wp2' else 1.)
            if delay:
                tt,state=states[i];pred=np.asarray(ns['compute_target_position'](t=tt+delay,t0=state['t0'],x0=state['x'],v0=state['v'],a0=state['a'],dt=float(np.float32(.05))))
                error=float(np.linalg.norm(pred[:2]-raw_goal[:2]));require(error<=1e-6,'original high-bin forecast does not match committed raw goal');errors.append(error)
                intent=Pose(pred);kind=FORECAST
        else:
            require(command['raw_stage']=='wp0' and command['desired_opening']>=.5,'button approach must preserve source wp0/open')
            require(np.linalg.norm(raw_goal[:2]-issue.position[:2])<=1e-6,'button wp0 is not its original current body-relative goal')
        counts[str(delay)]=counts.get(str(delay),0)+1
        binding=TRAINReferenceBinding(command['reference_role'],body,'translation_only',command['anchor_id'],command['manipulated_role'])
        provenance=dict(schema=goal.SCHEMA,translation_derivation_schema=SCHEMA,TRAIN_only=True,
            original_events_masks_and_command_ids_preserved=True,intent_label_is_actual_future_dynamics_observation=False,
            runtime_native_driver_or_teacher_parameter_access_allowed=False,source_declared_forecast_seconds=delay,
            privileged_TRAIN_native_driver_label_used=bool(delay),future_boundary_reflection_applied=False,
            current_readonly_derivation_sources_not_historical_execution_dependencies=True,permission=permission.pointer,
            kernel_sources=permission.kernel_sources,rule=rule.pointer,actual_plan_sha256=case['record']['plan_sha256'],raw_bindings=original.raw_bindings,
            source_branch_selected_from_original_prior_stage_not_outcome=True,
            current_world_waypoint_offset_is_preserved_canonical_geometry=(delay==0.))
        references.append(goal.TRAINCommandGoalReference(task,source.source_id,source.grouping_source,cid,float(command['generated_at']),
            binding,'external',kind,intent.xyzw(),issue.xyzw(),events.get(cid),provenance))
    require(goal.fingerprint(arrays)==original.original_masks_sha256,'raw masks changed')
    require({x.command_id for x in references}|{x['command_id'] for x in pending}=={c['command_id'] for c in source.command_ledger if c['raw_phase']==phase},'all original configured commands must be retained')
    proof=dict(schema=SCHEMA,permission=permission.pointer,all_original_configured_commands_retained=True,delay_counts=counts,
        exact_forecast_goal_xy_maximum_error_m=max(errors,default=0.),source_arithmetic_replay=replay,
        original_actual_metric_preserved=True,runtime_private_forecast_access_allowed=False)
    files=dict(original.derivation_proof['verified_files_sha256']);files.update(permission.verified_files_sha256)
    return goal.TRAINCommandGoalEvidence(task,source.source_id,source.grouping_source,rule.sha256,tuple(references),tuple(pending),
        original.raw_bindings,original.original_masks_sha256,original.original_command_ledger_sha256,
        original.original_completion_events_sha256,dict(original.derivation_proof,verified_files_sha256=files,translation_source_intention=proof))
