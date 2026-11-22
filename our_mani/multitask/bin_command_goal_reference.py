"""Explicit TRAIN bin intention labels; never a runtime dynamics predictor.

Readonly current source snapshots are new derivation evidence. They are not
represented as dependencies archived during the original acquisition.
"""
import ast
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List,Tuple

import numpy as np

from ..contracts import Pose
from . import acquisition_lineage as a,command_goal_reference as goal

TASK='put_rubbish_in_moving_bin'
SCHEMA='r10_TRAIN_bin_original_command_goal_intent_v1'
PERMISSION_SCHEMA='r10_TRAIN_bin_exact_source_intent_permission_v1'
FORECAST='original_expert_bin_unreflected_forecast'


def require(value,message):
    if not value:raise ValueError('bin command-goal: '+message)


@dataclass(frozen=True)
class BinIntention:
    task_name:str
    source_id:str
    parent_source_group:str
    command_id:int
    issued_at:float
    binding:object
    reference_mode:str
    intent_kind:str
    intent_reference_xyzw:np.ndarray
    original_issue_reference_xyzw:np.ndarray
    original_completion_metric:dict|None
    provenance:dict

    def __post_init__(self):
        require(self.task_name==TASK and self.binding.reference_kind=='translation_only'
            and self.binding.reference_id=='bin' and self.reference_mode=='external'
            and self.intent_kind in (goal.ISSUE,FORECAST),'invalid original bin intent scope')
        for key in ('intent_reference_xyzw','original_issue_reference_xyzw'):
            object.__setattr__(self,key,goal.frozen_pose(getattr(self,key)))
        require(self.provenance.get('schema')==SCHEMA and self.provenance.get('TRAIN_only') is True
            and self.provenance.get('intent_label_is_actual_future_dynamics_observation') is False
            and self.provenance.get('runtime_native_driver_or_teacher_parameter_access_allowed') is False,
            'missing intent/runtime separation')
        if self.intent_kind==goal.ISSUE:
            require(np.array_equal(self.intent_reference_xyzw,self.original_issue_reference_xyzw),
                'closed command must retain original current issue pose')
        else:require(self.provenance.get('source_declared_forecast_seconds')==.5
            and self.provenance.get('future_boundary_reflection_applied') is False,
            'opening command needs original unreflected source0.5 intention')


@dataclass(frozen=True)
class BinEvidence:
    task_name:str
    source_id:str
    parent_source_group:str
    rule_sha256:str
    references:tuple
    pending_commands:tuple
    raw_bindings:dict
    original_masks_sha256:dict
    original_command_ledger_sha256:str
    original_completion_events_sha256:str
    derivation_proof:dict


@dataclass(frozen=True)
class BinPermission:
    root:str
    pointer:dict
    kernel_sources:dict
    review:dict
    verified_files_sha256:dict
    production_fitting_enabled:bool=False


def load_permission(root,pointer,*,purpose):
    require(purpose in ('CPU_diagnostic','production_fitting'),'TRAIN fitting purpose required')
    root=Path(root).resolve();files={};path=a.checked_pointer(root,pointer,files);value=a.read(path)
    require(set(value)=={'schema','TRAIN_only','VAL_used','TEST_used','production_fitting_enabled',
        'task','helper','kernel_sources','readonly_current_source_review','exact_all100_replay'},'unknown permission fields')
    require(value['schema']==PERMISSION_SCHEMA and value['TRAIN_only'] is True
        and value['VAL_used'] is False and value['TEST_used'] is False and value['task']==TASK
        and value['production_fitting_enabled'] is (purpose=='production_fitting'),'permission scope changed')
    require(a.checked_pointer(root,value['helper'],files)==Path(__file__).resolve(),'helper source binding changed')
    keys={'task','motion_helper','pyrep_getter','sim_getter','C_ABI_header','C_ABI_builder'}
    require(set(value['kernel_sources'])==keys,'complete explicit new derivation source set required')
    for item in value['kernel_sources'].values():a.checked_pointer(root,item,files)
    review=a.read(a.checked_pointer(root,value['readonly_current_source_review'],files))
    require(review['schema']=='r10_bin_readonly_current_source_derivation_review_v1'
        and review['kernel_sources']==value['kernel_sources']
        and review['historical_execution_dependency_claimed'] is False
        and review['new_TRAIN_derivation_sources_explicitly_bound'] is True,'current snapshot is not historical execution proof')
    replay=a.read(a.checked_pointer(root,value['exact_all100_replay'],files))
    require(replay['original100_sources']==100 and replay['actual_body_frames']==11644
        and replay['maximum_actual_track_replay_error_m']==0.
        and replay['opening_forecast_commands']==90
        and replay['maximum_opening_forecast_raw_goal_xy_error_m']<1e-6,
        'all100 exact body-track and original forecast proof required')
    for key in ('task','motion_helper'):
        require(any(row['archive']['sha256']==value['kernel_sources'][key]['sha256']
            for row in replay['source'].values()),'replay uses a different arithmetic source')
    header=a.local(root,value['kernel_sources']['C_ABI_header']['path']).read_text()
    builder=a.local(root,value['kernel_sources']['C_ABI_builder']['path']).read_text()
    require('typedef float simFloat;' in header and 'simFloat simGetSimulationTimeStep();' in header
        and 'typedef float simFloat;' in builder and 'simFloat simGetSimulationTimeStep();' in builder,
        'native timestep binary32 ABI changed')
    return BinPermission(str(root),pointer,value['kernel_sources'],review,files,purpose=='production_fitting')


def _kernel(permission):
    root=Path(permission.root)
    def source(key):
        item=permission.kernel_sources[key];path=a.checked_pointer(root,item,{})
        return ast.parse(path.read_text())
    helper=source('motion_helper');functions=[node for node in helper.body
        if isinstance(node,ast.FunctionDef) and node.name in ('compute_target_position','cross_boundary')]
    require(len(functions)==2,'exact original motion helper owners required')
    task=source('task');cls=next(node for node in task.body if isinstance(node,ast.ClassDef))
    step=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='step')
    init=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='init_task')
    area=[]
    for node in init.body:
        if not isinstance(node,ast.Assign) or len(node.targets)!=1:continue
        target=node.targets[0]
        if ((isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name)
                and target.value.id=='self' and target.attr=='area')
                or isinstance(target,ast.Name) and target.id=='target_size_xy'):area.append(node)
    require(len(area)==3 and not any(isinstance(n,(ast.Call,ast.Import,ast.ImportFrom))
        for row in area for n in ast.walk(row)),'original area arithmetic must be pure source expressions')
    namespace=dict(np=np,deepcopy=deepcopy,List=List,Tuple=Tuple,Shape=object)
    exec(compile(ast.Module(body=functions+[step],type_ignores=[]),'<bound-TRAIN-bin-pure-kernel>','exec'),namespace)
    holder=SimpleNamespace();area_ns=dict(self=holder)
    exec(compile(ast.Module(body=area,type_ignores=[]),'<bound-TRAIN-bin-area>','exec'),area_ns)
    return namespace,holder.area


def replay_states(arrays,initial,permission):
    """Source arithmetic on raw TRAIN initial driver; no physics calls."""
    namespace,area=_kernel(permission);j=list(arrays['body_ids']).index('bin')
    positions=np.asarray(arrays['target_positions'][:,j]);dt=float(np.float32(.05))
    class Bin:
        def __init__(self,value):self.value=np.asarray(value,dtype=np.float32)
        def get_position(self):return self.value.astype(float).tolist()
        def set_position(self,value):self.value=np.asarray(value,dtype=np.float32)
    fake=SimpleNamespace(pyrep=SimpleNamespace(get_simulation_timestep=lambda:dt),target_state_list=[deepcopy(initial)],
        bin=Bin(positions[0]),area=area,_bool_expert=False,t=0.,step_id=0)
    states=[];flips=[]
    for i in range(len(positions)):
        require(np.array_equal(np.asarray(fake.bin.get_position()),positions[i]),
            'source arithmetic does not exactly reproduce actual bin frame '+str(i))
        states.append((fake.t,deepcopy(fake.target_state_list[-1])))
        if i+1<len(positions):
            count=len(fake.target_state_list);namespace['step'](fake)
            if len(fake.target_state_list)>count:flips.append(i)
    return states,namespace,flips


def derive_source(case,source,base_rule,permission,*,original_verification=None):
    require(isinstance(permission,BinPermission) and source.task_name==TASK and source.split=='train',
        'explicit bin TRAIN permission/source required')
    # The unchanged primary helper verifies actual parent/raw/archive/arrays/
    # clocks and preserves every original event. It attaches no bin basis.
    original=original_verification or goal.verify_original_source(case,source,base_rule)
    require(isinstance(original,goal.TRAINCommandGoalEvidence) and not original.references
        and (original.task_name,original.source_id,original.parent_source_group)==
            (source.task_name,source.source_id,source.grouping_source),
        'verified original raw source must precede bin derivation')
    arrays=case['arrays'];record=case['record'];ids=list(arrays['body_ids']);j=ids.index('bin')
    states,kernel,flips=replay_states(arrays,record['native_reset_receipt']['native_initial_snapshot']['active_motion_schedule'],permission)
    from .interaction import TRAINReferenceBinding
    event_map={row['origin_command_id']:row for row in case['command_receipt']['phase_completion_events']}
    references=[];pending=[];opening=closed=0
    for command in source.command_ledger:
        if command['raw_phase']!='deposit':continue
        cid=command['command_id'];indices=np.flatnonzero(np.isclose(source.times,command['generated_at'],atol=1e-8,rtol=0))
        require(len(indices)==1,'unique original issue frame required');i=int(indices[0])
        require(command['reference_body']=='bin' and command['reference_kind']=='translation_only'
            and command['issued_reference_motion_mode']=='external','original bin command binding/mode changed')
        if not bool(arrays['native_pose_valid'][i,j] and arrays['mode_valid'][i,j]):
            pending.append(dict(command_id=cid,reason='original issue reference mask invalid',raw_source_preserved=True));continue
        issue=Pose(np.asarray(command['issued_reference_native_pose_xyzw'])[:3])
        require(np.array_equal(issue.position,arrays['target_positions'][i,j]),'issued bin pose differs from raw frame')
        binding=TRAINReferenceBinding(command['reference_role'],'bin','translation_only',command['anchor_id'],command['manipulated_role'])
        provenance=dict(schema=SCHEMA,TRAIN_only=True,intent_label_is_actual_future_dynamics_observation=False,
            runtime_native_driver_or_teacher_parameter_access_allowed=False,original_events_masks_and_command_ids_preserved=True,
            source_declared_forecast_seconds=0.,privileged_TRAIN_native_driver_label_used=False,
            current_readonly_derivation_sources_not_historical_execution_dependencies=True,
            permission=permission.pointer,kernel_sources=permission.kernel_sources,actual_rule=base_rule.pointer,
            raw_bindings=original.raw_bindings)
        intent=issue;kind=goal.ISSUE
        if command['desired_opening']>=.5:
            t,state=states[i];predicted=np.asarray(kernel['compute_target_position'](t=t+.5,t0=state['t0'],
                x0=state['x'],v0=state['v'],a0=state['a'],dt=float(np.float32(.05))))
            require(np.linalg.norm(predicted[:2]-np.asarray(command['raw_waypoint_xyzw'])[:2])<1e-6,
                'opening raw goal differs from source-declared original forecast')
            # The expert overrides XY only. The physical bin reference height
            # stays at the actual issued body height; waypoint Z is geometry.
            intent=Pose(np.r_[predicted[:2],issue.position[2]]);kind=FORECAST;opening+=1
            provenance.update(source_declared_forecast_seconds=.5,privileged_TRAIN_native_driver_label_used=True,
                future_boundary_reflection_applied=False,original_task_time=t,original_reflected_driver_state=state,
                original_future_XY_only=True)
        else:closed+=1
        references.append(BinIntention(TASK,source.source_id,source.grouping_source,cid,float(command['generated_at']),
            binding,'external',kind,intent.xyzw(),issue.xyzw(),event_map.get(cid),provenance))
    require({r.command_id for r in references}|{r['command_id'] for r in pending}==
        {r['command_id'] for r in source.command_ledger if r['raw_phase']=='deposit'},'every original deposit command must remain')
    files=dict(original.derivation_proof['verified_files_sha256']);files.update(permission.verified_files_sha256)
    return BinEvidence(TASK,source.source_id,source.grouping_source,permission.pointer['sha256'],tuple(references),tuple(pending),
        original.raw_bindings,original.original_masks_sha256,original.original_command_ledger_sha256,
        original.original_completion_events_sha256,dict(actual_raw_source_verified=True,actual_bin_frames=len(states),
            exact_source_replay_error_m=0.,opening_forecast_commands=opening,closed_issue_commands=closed,
            boundary_state_update_frames=flips,verified_files_sha256=files,
            historical_execution_dependency_claimed=False,permission=permission.pointer))


def evidence_proof(evidence):
    require(isinstance(evidence,BinEvidence),'typed original bin proof required')
    value=dict(schema=SCHEMA,source_id=evidence.source_id,parent_source_group=evidence.parent_source_group,
        permission_sha256=evidence.rule_sha256,raw_bindings=evidence.raw_bindings,
        original_masks_sha256=evidence.original_masks_sha256,original_command_ledger_sha256=evidence.original_command_ledger_sha256,
        original_completion_events_sha256=evidence.original_completion_events_sha256,
        references=[dict(command_id=r.command_id,intent_kind=r.intent_kind,intent_reference_xyzw=r.intent_reference_xyzw.tolist(),
            original_issue_reference_xyzw=r.original_issue_reference_xyzw.tolist(),original_completion_metric=r.original_completion_metric,
            provenance=r.provenance) for r in evidence.references],pending_commands=list(evidence.pending_commands),
        replay={k:v for k,v in evidence.derivation_proof.items() if k!='verified_files_sha256'})
    return dict(value,derivation_sha256=a.canonical_sha(value))
