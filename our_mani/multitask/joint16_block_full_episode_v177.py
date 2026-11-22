"""TRAIN development bridge to the unchanged R09 geometric block executor.

Public motion-state forwarding and read-only counted joint-window monitoring
add no render or physics call. Each actual case must still run in a fresh
simulator process. This helper does not authorize formal VAL or TEST.
"""
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
import json
import numpy as np

from .joint16_block_context_v97 import Joint16BlockContext
from .joint16_tracks_v31 import BLOCKS
from .evaluation import digest,file_sha

SCHEMA='r10_joint16_block_full_counted_budget_TRAIN_geometric_bridge_v177'


from .joint16_block_episode_v98 import PublicCountedBlockEnvironment, shared_context_factory_scope
from .block_full_budget_v176 import finish_counted_budget


def run_TRAIN_development_episode(env,row,config,*,method,root):
    root=Path(root).resolve()
    if method not in ('cv_geometry','context_geometry') or row.get('task') not in BLOCKS:
        raise ValueError('only the two moving-block tasks and authorized geometry methods')
    roster=json.loads((root/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    first=[r for r in roster if r['task']==row['task']][:4]
    if row not in first:raise ValueError('fixed first four original TRAIN rows; no TEST or successful-source replacement')
    if (config.get('dt')!=.05 or config.get('simulation_seconds')!=10.
            or config.get('global_joint_acceleration')!=6. or config.get('warmup_ticks')!=5):
        raise ValueError('preserve the original 50ms/10s/six-unit/five-frame block protocol')
    from scripts import evaluate_aligned_r09 as evaluator
    from ..context_dynamics import ContextLatentDynamics as original_context_class
    if evaluator.ContextLatentDynamics is not original_context_class:
        raise ValueError('fresh isolated process required; nested block model scopes are forbidden')
    c=deepcopy(config);predictor=method.split('_')[0];context=None
    if predictor=='context':
        pointer=c.get('joint16_context_checkpoint')
        if not isinstance(pointer,dict) or set(pointer)!={'path','sha256'}:
            raise ValueError('explicit frozen shared joint16 checkpoint pointer required')
        checkpoint=(root/pointer['path']).resolve()
        if not checkpoint.is_relative_to(root) or file_sha(checkpoint)!=pointer['sha256']:
            raise ValueError('shared joint16 checkpoint path/hash differs')
        context=Joint16BlockContext(checkpoint,task=row['task'],root=root)
        c['dynamics_checkpoint']=str(checkpoint.relative_to(root))
    # The old geometry kernel reads these strings before choosing its branch;
    # no Value file is admitted, opened or instantiated.
    c['cv_local_value']=c['context_local_value']='';c['value_input_context']=False
    boundary=PublicCountedBlockEnvironment(env,context)
    with shared_context_factory_scope(evaluator,context):
        record,frames=evaluator.run_episode(boundary,row,c,predictor=predictor,
            ranking='geometry',official=False,video=False)
    finish_counted_budget(boundary, record)
    source_paths=('our_mani/multitask/joint16_block_full_episode_v177.py','our_mani/multitask/block_full_budget_v176.py','our_mani/multitask/joint16_block_episode_v98.py',
        'our_mani/multitask/joint16_block_context_v97.py','scripts/evaluate_aligned_r09.py',
        'our_mani/multitask/context_se3.py')
    binding=dict(schema=SCHEMA,method=method,task=row['task'],fixed_TRAIN_row_sha256=digest(row),
        original_geometric_component_sha256=record['controller_sha256'],
        sources_sha256={name:file_sha(root/name) for name in source_paths},
        shared_context=None if context is None else context.binding,Value_used=False,
        NGFF_used=False,FASTER_used=False,formal_VAL_TEST=False)
    record.update(legacy_geometric_component_controller_sha256=record['controller_sha256'],
        controller_sha256=digest(binding),joint16_block_execution=dict(binding,sha256=digest(binding)),
        all_stage_counted_joint_motion=boundary.receipt(),
        shared_context_receipt=None if context is None else context.receipt(),
        original_early_policy_failure_retained=True,original_early_policy_termination_retained=False,
        task_budget_fully_consumed_or_native_success=bool(record['official_task_success']
            or record.get('native_terminal_failure') or np.isclose(env.time,10.)),
        demo_only=True,development_demo_only=True,formal_validation_metric=False,
        formal_TEST_metric=False,old_R09_release_not_reused=True)
    return record,frames
