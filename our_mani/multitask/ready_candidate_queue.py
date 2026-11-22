"""Task-local TRAIN/CAL collection contracts for executed candidate critics.

This module never imports a native simulator or torch. The original full source
manifests stay intact; a task subset selects their exact rows. Development videos,
teachers, hold trajectories and unexecuted candidates are not critic labels.
"""
from collections import Counter
from pathlib import Path
import json

from .candidate_success import CONTINUATION_POLICY
from .candidate_training import MANIFEST_SCHEMA
from .evaluation import case_identity, digest, file_sha, validate_completed_case
from .specs import TASK_SPECS

VERSION = 'r10_task_ready_executed_candidate_queue_v1'
SCOPE = 'r10_task_ready_critic_TRAIN_CAL_v1'
METHODS = ('cv_geometry', 'context_geometry')
STAGES = {'train': 'train', 'cal': 'validation', 'geometry_val': 'validation'}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def pointer(root, path):
    path = (Path(root) / path).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'local receipt pointer leaves project')
    return dict(path=str(path.relative_to(Path(root).resolve())), sha256=file_sha(path))


def checked(root, value):
    require(isinstance(value, dict) and set(value) >= {'path', 'sha256'}, 'missing file pointer')
    path = (Path(root) / value['path']).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'receipt pointer leaves project')
    require(path.is_file() and file_sha(path) == value['sha256'], 'receipt bytes changed: ' + str(path))
    return path


def select_original_rows(manifest, split, tasks):
    """Select exact existing rows, checking all fourteen original task budgets."""
    require(split in ('train', 'validation'), 'critic collection is TRAIN/CAL only; no TEST')
    require(manifest.get('split') == split, 'original source split differs')
    values = manifest.get('episodes')
    require(isinstance(values, list), 'use original full episodes manifest')
    require(tasks and len(tasks) == len(set(tasks)) and set(tasks) <= set(TASK_SPECS),
            'unique declared remaining tasks required')
    rows = [row for row in values if row.get('task') in TASK_SPECS]
    counts = Counter(row['task'] for row in rows)
    expected = 1400 if split == 'train' else 275
    require(len(rows) == expected and set(counts) == set(TASK_SPECS),
            'preserve complete original fourteen-task source manifest')
    require(all(count == 100 for count in counts.values()) if split == 'train'
            else all(18 <= count <= 20 for count in counts.values()),
            'original per-task TRAIN/CAL parent coverage differs')
    keys = [(row['task'], row.get('raw_episode_id')) for row in rows]
    require(len(keys) == len(set(keys)) and all(isinstance(raw, str) and raw for _, raw in keys),
            'duplicate or missing original source parent')
    require(all(row.get('split', split) == split and isinstance(row.get('path'), str)
                and row['path'] for row in rows), 'source row split/path differs')
    return [dict(row) for row in rows if row['task'] in tasks]


def build_jobs(rows, stage, config_sha, lock_sha, seed):
    require(stage in STAGES, 'only declared TRAIN/CAL/geometry-VAL stages can create cases')
    split = STAGES[stage]
    jobs = []
    for row in rows:
        for method in METHODS:
            exploration = (int(digest([seed, row['task'], row['raw_episode_id'], 0, method, 'first'])[:8], 16)
                           if stage == 'train' else None)
            identity = case_identity(split=split, task=row['task'],
                raw_episode_id=row['raw_episode_id'] + '__policy_aug_r0', method=method,
                source_row=row, config_sha256=config_sha, inputs_sha256=lock_sha,
                parent_raw_episode_id=row['raw_episode_id'], replica=0, exploration_seed=exploration)
            jobs.append(dict(job_id=identity['sha256'][:20], identity=identity,
                source_raw_episode_id=row['raw_episode_id'], source_row=row,
                exploration_seed=exploration, replica=0, record_demo=False))
    require(len({job['job_id'] for job in jobs}) == len(jobs), 'duplicate executed candidate job')
    return jobs


def validate_raw_child(root, plan, job, path, config):
    """Keep original evaluator/base/hook and the actual wrapper independently bound."""
    record = read(path)
    validate_completed_case(record, job['identity'])
    require(not record.get('demo_only') and not record.get('development_demo_execution')
            and record.get('video') is None, 'development videos cannot be critic fitting sources')
    predictor = job['identity']['method'].split('_', 1)[0]
    task = job['identity']['task']
    library_identity = config['executed_candidate_collection_scope']['interaction_library_identities'][task]
    require(record.get('predictor') == predictor
            and record.get('continuation_sha256') == config['continuation_identities'][task][predictor]
            and record.get('interaction_library_sha256') == library_identity
            and record.get('continuation_selection_policy') == CONTINUATION_POLICY
            and record.get('downstream_selection_policy') == 'geometry'
            and record.get('unexecuted_candidates_labelled') is False,
            'labels differ from matching fitted predictor and frozen geometry continuation')
    selection = 'TRAIN_explore_once' if plan['stage'] == 'train' else 'geometry'
    require(record.get('first_selection_policy') == selection, 'first-selection collection distribution differs')
    require(record.get('entrypoint_sha256') == plan['original_evaluator']['sha256'],
            'original evaluator source identity differs')
    witness = record.get('task_execution_witness', {})
    require(witness.get('schema') == 'r10_task_formal_case_execution_contract_v1'
            and witness.get('task') == task and witness.get('split') == STAGES[plan['stage']]
            and witness.get('actual_wrapper') == plan['child_source']
            and witness.get('original_evaluator') == plan['original_evaluator']
            and witness.get('actual_process_id') == record['process_id'],
            'actual wrapper/process execution witness differs')
    base = read(checked(root, witness['base_evaluator_receipt']))
    require(base == {key: value for key, value in record.items() if key != 'task_execution_witness'},
            'trained wrapper changed original evaluator raw labels/receipts')
    hook = read(checked(root, witness['hook_receipt']))
    require(hook.get('identity') == job['identity'] and hook.get('complete') is True
            and hook.get('actual_wrapper') == plan['child_source']
            and hook.get('actual_process_id') == record['process_id']
            and hook.get('split') == STAGES[plan['stage']]
            and hook.get('shared_development_hooks_applied') is True
            and hook.get('automatic_retry') is False
            and hook.get('policy_failure_label_assigned') is False,
            'missing actual shared physical reset/RML cleanup evidence')
    return record


def critic_manifest(plan, jobs_and_paths, root):
    """Every fixed source enters the manifest, including zero-candidate failures."""
    require(plan['stage'] in ('train', 'cal'), 'formal geometry VAL cannot silently replace critic CAL')
    require(len(jobs_and_paths) == len(plan['jobs'])
            and {job['job_id'] for job, _ in jobs_and_paths}
                == {job['job_id'] for job in plan['jobs']}, 'critic manifest needs complete declared queue')
    entries = []
    for job, path in jobs_and_paths:
        identity = job['identity']
        entries.append(dict(split=identity['split'], task=identity['task'],
            predictor=identity['method'].split('_', 1)[0],
            raw_episode_id=identity['raw_episode_id'],
            parent_raw_episode_id=identity['parent_raw_episode_id'],
            parent_source_group=':'.join((identity['split'], identity['task'], identity['parent_raw_episode_id'])),
            **pointer(root, path)))
    return dict(schema=MANIFEST_SCHEMA, entries=entries, collection_stage=plan['stage'],
        queue_plan_sha256=plan['sha256'], exact_original_source_rows=True,
        fixed_replica=0, automatic_retry=False, unexecuted_candidates_labelled=False,
        teacher_hold_and_development_demo_sources_used=False, TEST_used_for_fitting=False,
        original_parent_counts=plan['per_task_original_parents'])
