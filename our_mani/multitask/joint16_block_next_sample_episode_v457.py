"""TRAIN development bridge to the unchanged R09 geometric block executor.

Public motion-state forwarding and read-only counted joint-window monitoring
do not advance physics. Explicit fresh counted safety-tail observations are disclosed. Each actual case must still run in a fresh
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

SCHEMA='r10_joint16_block_same_next_actual_TRAIN_bridge_v457'


from .joint16_block_episode_v98 import shared_context_factory_scope
from .block_clip_brake_v315 import public_decision_scope,binding as held_binding
from .block_fresh_tail_v448 import measured_tracker_scope,finish_counted_budget,binding as fresh_binding
from .block_next_sample_v456 import SmoothBlockEnvironment as PublicCountedBlockEnvironment,binding as next_sample_binding


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
    if config.get('block_clip_brake_v315') != held_binding(root):
        raise ValueError('freeze actual held geometry and contact semantics')
    boundary=PublicCountedBlockEnvironment(env,context)
    with measured_tracker_scope(evaluator,boundary), shared_context_factory_scope(evaluator,context), public_decision_scope(evaluator,boundary):
        record,frames=evaluator.run_episode(boundary,row,c,predictor=predictor,
            ranking='geometry',official=False,video=False)
        finish_counted_budget(boundary, record)
    source_paths=('our_mani/multitask/block_next_sample_v456.py','our_mani/multitask/next_sample_actual_v455.py','our_mani/multitask/block_fresh_tail_v448.py','our_mani/multitask/block_clip_brake_v315.py','our_mani/multitask/block_held_support_v298.py','our_mani/multitask/block_exact_reference_v201.py','our_mani/multitask/joint16_block_next_sample_episode_v457.py','our_mani/multitask/joint16_block_reference_episode_v198.py','our_mani/multitask/joint16_block_reference_episode_v187.py','our_mani/multitask/block_reference_v197.py','our_mani/multitask/joint16_block_full_episode_v177.py','our_mani/multitask/block_full_budget_v176.py','our_mani/multitask/joint16_block_episode_v98.py',
        'our_mani/multitask/joint16_block_context_v97.py','scripts/evaluate_aligned_r09.py',
        'our_mani/multitask/context_se3.py')
    binding=dict(schema=SCHEMA,method=method,task=row['task'],fixed_TRAIN_row_sha256=digest(row),
        original_geometric_component_sha256=record['controller_sha256'],
        sources_sha256={name:file_sha(root/name) for name in source_paths},
        next_sample_actual=next_sample_binding(root),held_geometry_and_contact=held_binding(root),fresh_counted_tail=fresh_binding(root),shared_context=None if context is None else context.binding,Value_used=False,
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
