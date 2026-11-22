"""TRAIN-only original hoop intention; no runtime teacher/dynamics access.

The task's closed and opening deposit commands both use its unreflected
source-declared t+0.5 waypoint. Current source snapshots are explicit new
derivation evidence, not retroactive acquisition dependencies.
"""
import ast
from copy import deepcopy
from dataclasses import dataclass,replace
from pathlib import Path
from types import SimpleNamespace
from typing import List,Tuple

import numpy as np

from ..contracts import Pose
from . import acquisition_lineage as a,command_goal_reference as goal

TASKS=('moving_basketball_in_hoop','moving_basketball_in_hoop_high_speed')
SCHEMA='r10_TRAIN_basketball_original_command_goal_intent_v1'
PERMISSION_SCHEMA='r10_TRAIN_basketball_exact_source_intent_permission_v1'
FORECAST='original_expert_hoop_unreflected_forecast'


def require(value,message):
    if not value:raise ValueError('basketball command-goal: '+message)


@dataclass(frozen=True)
class BasketballPermission:
    root:str
    pointer:dict
    kernel_sources:dict
    verified_files_sha256:dict
    production_fitting_enabled:bool=False


def load_permission(root,pointer,*,purpose):
    require(purpose in ('CPU_diagnostic','production_fitting'),'TRAIN fitting purpose required')
    root=Path(root).resolve();files={};value=a.read(a.checked_pointer(root,pointer,files))
    require(set(value)=={'schema','TRAIN_only','VAL_used','TEST_used','production_fitting_enabled','tasks',
        'helper','kernel_sources','readonly_current_source_review','exact_all400_replay','complete_aux200_audit'},
        'unknown permission fields')
    require(value['schema']==PERMISSION_SCHEMA and value['TRAIN_only'] is True and value['VAL_used'] is False
        and value['TEST_used'] is False and value['tasks']==list(TASKS)
        and value['production_fitting_enabled'] is (purpose=='production_fitting'),'permission scope differs')
    require(a.checked_pointer(root,value['helper'],files)==Path(__file__).resolve(),'helper source binding changed')
    keys=set(TASKS)|{'motion_helper','pyrep_getter','sim_getter','C_ABI_header','C_ABI_builder'}
    require(set(value['kernel_sources'])==keys,'complete explicit arithmetic/native ABI source set required')
    for item in value['kernel_sources'].values():a.checked_pointer(root,item,files)
    review=a.read(a.checked_pointer(root,value['readonly_current_source_review'],files))
    require(review['schema']=='r10_basketball_readonly_current_source_derivation_review_v1'
        and review['kernel_sources']==value['kernel_sources'] and review['historical_execution_dependency_claimed'] is False
        and review['new_TRAIN_derivation_sources_explicitly_bound'] is True,'new snapshots are not historical closure proof')
    proof=a.read(a.checked_pointer(root,value['exact_all400_replay'],files))
    require(proof['all400_sources_and_original200_parent_groups_retained'] is True
        and proof['only_TRAIN_recorded_initial_driver_and_original_committed_goals_used'] is True
        and proof['actual_future_observations_not_intention_labels'] is True
        and set(proof['tasks'])==set(TASKS),'complete TRAIN400 source replay evidence required')
    for task,row in proof['tasks'].items():
        require(row['real_raw_rollouts']==200 and row['original_parent_groups']==100
            and row['exact_hoop_track_error_m']==0. and row['source_goal_maximum_error_m']<1e-6
            and row['no_tail_or_failed_source_removed'] is True,'all real source rows/tails must remain')
        require(any(item['archive']['sha256']==value['kernel_sources'][task]['sha256']
            for item in proof['source'].values()),'draft replay used another task source')
        a.checked_pointer(root,row['all_source_command_proof'],files)
        a.checked_pointer(root,row['full_curve_report'],files)
    require(sum(row['all_hoop_frames'] for row in proof['tasks'].values())==74705
        and sum(row['all_deposit_commands'] for row in proof['tasks'].values())==675,
        'the fixed all400/675command replay scope differs')
    require(any(item['archive']['sha256']==value['kernel_sources']['motion_helper']['sha256']
        for item in proof['source'].values()),'draft replay used another motion helper')
    complete=a.read(a.checked_pointer(root,value['complete_aux200_audit'],files))
    require(complete['auxiliary_complete200'] is True and complete['original_Context_primary_K1_changed'] is False,
        'complete fixed auxiliary200 audit required')
    for key in ('C_ABI_header','C_ABI_builder'):
        text=a.local(root,value['kernel_sources'][key]['path']).read_text()
        require('typedef float simFloat;' in text and 'simFloat simGetSimulationTimeStep();' in text,
            'native timestep binary32 ABI changed')
    return BasketballPermission(str(root),pointer,value['kernel_sources'],files,purpose=='production_fitting')


def _kernel(permission,task_name):
    require(task_name in TASKS,'unknown hoop task')
    root=Path(permission.root)
    def source(key):return ast.parse(a.checked_pointer(root,permission.kernel_sources[key],{}).read_text())
    tree=source(task_name);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)
        and any(isinstance(m,ast.FunctionDef) and m.name=='step' for m in n.body))
    step=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
    index=next(i for i,n in enumerate(step.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple)
        and [v.id for v in n.targets[0].elts]==['hoop_target_position','hoop_target_velocity'])
    suffix=[n for n in step.body if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Attribute)
        and n.target.attr in ('step_id','t')]
    selected=step.body[:3]+step.body[index:index+3]+suffix
    require(len(suffix)==2,'source hoop clock changed')
    replay=ast.FunctionDef(name='step',args=step.args,body=selected,decorator_list=[],returns=step.returns)
    ast.fix_missing_locations(replay)
    helpers=[n for n in source('motion_helper').body if isinstance(n,ast.FunctionDef)
        and n.name in ('compute_target_position','cross_boundary')]
    require(len(helpers)==2,'source helper owner changed')
    expert=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_expert_info')
    calls=[n for n in ast.walk(expert) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
        and n.func.id=='compute_target_position' and any(k.arg=='x0'
            and ast.unparse(k.value)=="hoop_target_state_dict['wp3_x']" for k in n.keywords)]
    require(len(calls)==2 and all(any(k.arg=='t' and ast.unparse(k.value)=='task.t + t_delay'
        for k in c.keywords) for c in calls),'closed/open source0.5 intention changed')
    for call in calls:
        owners=[n for n in ast.walk(expert) if isinstance(n,ast.If)
            and any(call in tuple(ast.walk(statement)) for statement in n.body)]
        require(len(owners)==1,'original forecast branch owner is ambiguous')
        statements=owners[0].body;index=next(i for i,n in enumerate(statements) if call in tuple(ast.walk(n)))
        delays=[n for n in statements[:index] if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name)
            and t.id=='t_delay' for t in n.targets)]
        require(len(delays)==1 and isinstance(delays[0].value,ast.Constant) and delays[0].value.value==.5,
            'closed/open source0.5 intention changed')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='init_task');area=[]
    for i,node in enumerate(init.body):
        if not isinstance(node,ast.Assign) or len(node.targets)!=1:continue
        t=node.targets[0]
        if isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self' and t.attr=='hoop_area':
            if not area:area.append(init.body[i-1])
            area.append(node)
    require(len(area)==3 and not any(isinstance(n,(ast.Call,ast.Import,ast.ImportFrom)) for row in area for n in ast.walk(row)),
        'source hoop area is not pure arithmetic')
    namespace=dict(np=np,deepcopy=deepcopy,List=List,Tuple=Tuple,Shape=object)
    exec(compile(ast.Module(body=helpers+[replay],type_ignores=[]),'<bound-TRAIN-hoop-kernel>','exec'),namespace)
    holder=SimpleNamespace();exec(compile(ast.Module(body=area,type_ignores=[]),'<bound-TRAIN-hoop-area>','exec'),dict(self=holder))
    return namespace,holder.hoop_area


def replay_states(arrays,initial,permission,task_name):
    namespace,area=_kernel(permission,task_name);j=list(arrays['body_ids']).index('hoop')
    positions=np.asarray(arrays['target_positions'][:,j]);ball,hoop=deepcopy(initial)
    offset=np.asarray(hoop['wp3_x'])-positions[0]
    class Hoop:
        def __init__(self,v):self.value=np.asarray(v,np.float32)
        def get_position(self):return self.value.astype(float).tolist()
        def set_position(self,v):self.value=np.asarray(v,np.float32)
    obj=Hoop(positions[0]);waypoint=SimpleNamespace(get_position=lambda:
        (obj.value.astype(float)+offset).astype(np.float32).astype(float).tolist())
    fake=SimpleNamespace(pyrep=SimpleNamespace(get_simulation_timestep=lambda:float(np.float32(.05))),
        target_state_list=[(ball,hoop)],hoop=obj,wp3=waypoint,hoop_area=area,t=0.,step_id=0)
    states=[]
    for i in range(len(positions)):
        require(np.array_equal(np.asarray(obj.get_position()),positions[i]),'source replay differs from actual hoop frame '+str(i))
        states.append((fake.t,deepcopy(fake.target_state_list[-1][1])))
        if i+1<len(positions):namespace['step'](fake)
    return states,namespace


def derive_source(case,source,rule,permission,*,original_verification):
    require(isinstance(permission,BasketballPermission) and source.task_name in TASKS and source.split=='train',
        'explicit basketball TRAIN permission/source required')
    original=original_verification
    require(isinstance(original,goal.TRAINCommandGoalEvidence) and not original.references
        and (original.task_name,original.source_id,original.parent_source_group)==
            (source.task_name,source.source_id,source.grouping_source),'original strict raw verification must precede intention')
    arrays=case['arrays'];j=list(arrays['body_ids']).index('hoop')
    states,kernel=replay_states(arrays,case['record']['native_reset_receipt']['native_initial_snapshot']['active_motion_schedule'],
        permission,source.task_name)
    events={e['origin_command_id']:e for e in case['command_receipt']['phase_completion_events']};references=[];pending=[]
    for command in source.command_ledger:
        if command['raw_phase']!='deposit':continue
        cid=command['command_id'];indices=np.flatnonzero(np.isclose(source.times,command['generated_at'],atol=1e-8,rtol=0))
        require(len(indices)==1,'unique original issue frame required');i=int(indices[0])
        require(command['reference_body']=='hoop' and command['reference_kind']=='translation_only'
            and command['issued_reference_motion_mode']=='external','original hoop command binding/mode changed')
        if not bool(arrays['native_pose_valid'][i,j] and arrays['mode_valid'][i,j]):
            pending.append(dict(command_id=cid,reason='original issue hoop pose/mode mask invalid',raw_source_preserved=True));continue
        issue=Pose(np.asarray(command['issued_reference_native_pose_xyzw'])[:3])
        require(np.array_equal(issue.position,arrays['target_positions'][i,j]),'issued hoop pose differs from actual raw frame')
        t,state=states[i]
        def predict(key):return np.asarray(kernel['compute_target_position'](t=t+.5,t0=state['t0'],
            x0=state[key],v0=state['v'],a0=state['a']))
        require(np.linalg.norm(predict('wp3_x')-np.asarray(command['raw_waypoint_xyzw'])[:3])<1e-6,
            'closed/open original wp3 goal differs from source0.5 intention')
        binding=goal.interaction.TRAINReferenceBinding(command['reference_role'],'hoop','translation_only',
            command['anchor_id'],command['manipulated_role'])
        provenance=dict(schema=goal.SCHEMA,TRAIN_only=True,original_events_masks_and_command_ids_preserved=True,
            intent_label_is_actual_future_dynamics_observation=False,runtime_native_driver_or_teacher_parameter_access_allowed=False,
            source_declared_forecast_seconds=.5,privileged_TRAIN_native_driver_label_used=True,
            future_boundary_reflection_applied=False,current_readonly_derivation_sources_not_historical_execution_dependencies=True,
            closed_and_open_both_use_original_forecast=True,original_task_time=t,original_reflected_driver_state=state,
            rule=rule.pointer,permission=permission.pointer,kernel_sources=permission.kernel_sources,
            raw_bindings=original.raw_bindings,basketball_derivation_schema=SCHEMA)
        references.append(goal.TRAINCommandGoalReference(source.task_name,source.source_id,source.grouping_source,cid,
            float(command['generated_at']),binding,'external',FORECAST,Pose(predict('x')).xyzw(),issue.xyzw(),events.get(cid),provenance))
    require({r.command_id for r in references}|{r['command_id'] for r in pending}==
        {r['command_id'] for r in source.command_ledger if r['raw_phase']=='deposit'},'every original deposit command must remain')
    files=dict(original.derivation_proof['verified_files_sha256']);files.update(permission.verified_files_sha256)
    return replace(original,references=tuple(references),pending_commands=tuple(pending),
        derivation_proof=dict(original.derivation_proof,verified_files_sha256=files,basketball_source_intention=dict(
            schema=SCHEMA,permission=permission.pointer,actual_hoop_frames=len(states),exact_source_replay_error_m=0.,
            all_deposit_commands=len(references)+len(pending),historical_execution_dependency_claimed=False)))
