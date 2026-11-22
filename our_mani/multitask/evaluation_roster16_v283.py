"""Full16 source selection and explicit14/2 routing for a future frozen evaluation.

This module neither qualifies controllers nor launches evaluation. It replaces
the old560-source selector's coverage rule without pretending a14-task subset
is the complete16-task cohort. Original source rows remain unchanged.
"""
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from .specs import TASK_SPECS
from .joint16_tracks_v31 import BLOCKS
from .protocol import MultitaskProtocolError

METHODS=('cv_geometry','context_geometry','physmani')
TASKS=tuple(sorted(set(TASK_SPECS)|set(BLOCKS)))
if len(TASKS)!=16 or len(TASK_SPECS)!=14 or len(BLOCKS)!=2:raise RuntimeError('explicit original14+2 registry required')
TRAIN_COUNTS=MappingProxyType({task:100 for task in TASKS})
VAL_COUNTS=MappingProxyType({task:(18 if task in ('insert_onto_rotating_peg','put_rubbish_in_moving_bin_high_speed')
    else 19 if task=='insert_onto_rotating_peg_high_speed' else 20) for task in TASKS})
TEST_COUNTS=MappingProxyType({task:40 for task in TASKS})


@dataclass(frozen=True)
class Route:
    task: str
    method: str
    component: str
    GPU_required: bool


def route(task,method):
    if task not in TASKS or method not in METHODS:raise MultitaskProtocolError('exactly16 tasks and3 authorized methods required')
    return Route(task,method,'aligned_block' if task in BLOCKS else 'r10_multitask',method=='physmani')


def validate_roster(manifest,split,*,completed_VAL_audit=False):
    if split not in ('train','validation','test') or manifest.get('split')!=split:
        raise MultitaskProtocolError('exact declared evaluation source split required')
    if split=='test' and completed_VAL_audit is not True:
        raise MultitaskProtocolError('complete frozen-policy VAL audit required before TEST source selection')
    values=manifest.get('episodes',manifest.get('entries'))
    if not isinstance(values,list) or not values:raise MultitaskProtocolError('complete original source roster required')
    expected=TRAIN_COUNTS if split=='train' else VAL_COUNTS if split=='validation' else TEST_COUNTS
    keys=[]
    for row in values:
        if (not isinstance(row,dict) or row.get('split',split)!=split or row.get('task') not in TASKS
                or not isinstance(row.get('raw_episode_id'),str) or not row['raw_episode_id']
                or not isinstance(row.get('path'),str) or not row['path']):
            raise MultitaskProtocolError('original canonical task/raw_episode_id/path row required')
        keys.append((row['task'],row['raw_episode_id']))
    counts=Counter(task for task,_ in keys)
    if len(set(keys))!=len(values) or dict(counts)!=dict(expected):
        raise MultitaskProtocolError('full16 source coverage or uniqueness differs;14-task views are not full cohorts')
    if split=='test' and (manifest.get('mapping_status')!='verified' or manifest.get('episode_count')!=640
            or manifest.get('training_calibration_or_parameter_selection_used') is not False):
        raise MultitaskProtocolError('complete verified640-source author TEST mapping and no tuning required')
    return values


def select_source(manifest,split,task,raw,*,completed_VAL_audit=False):
    if task not in TASKS:raise MultitaskProtocolError('unknown full16 task')
    values=validate_roster(manifest,split,completed_VAL_audit=completed_VAL_audit)
    rows=[row for row in values if row['task']==task and row['raw_episode_id']==raw]
    if len(rows)!=1:raise MultitaskProtocolError('one exact original full16 source required')
    return (dict(rows[0]),rows[0]),values


def case_grid(manifest,split,*,completed_VAL_audit=False):
    values=validate_roster(manifest,split,completed_VAL_audit=completed_VAL_audit)
    return [dict(task=row['task'],raw_episode_id=row['raw_episode_id'],method=method,
        component=route(row['task'],method).component,source_row=dict(row)) for row in values for method in METHODS]


@contextmanager
def core14_selection_scope(evaluator,*,completed_VAL_audit=False):
    """Use the full cohort in the14-task core; blocks must take their own branch."""
    original=evaluator.select_source
    def select(manifest,split,task,raw):
        if task not in TASK_SPECS:raise MultitaskProtocolError('two block tasks must use explicit aligned_block route')
        return select_source(manifest,split,task,raw,completed_VAL_audit=completed_VAL_audit)
    evaluator.select_source=select
    try:yield
    finally:evaluator.select_source=original
