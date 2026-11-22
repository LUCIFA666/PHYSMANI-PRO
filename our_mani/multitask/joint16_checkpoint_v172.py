"""Qualify the actual shared16 dependency checkpoint for development consumers."""
from collections import Counter
from pathlib import Path
import json
from .context_se3 import MultiBodyContextDynamics
from .evaluation import file_sha
from .frozen_subset_batch_v132 import checked_subset


def check_training_metadata_and_roster(metadata, entries, roster):
    parents = [(r['task'], r['raw_episode_id']) for r in roster]
    trained = [(r['task'], r['parent_raw_episode_id']) for r in entries]
    tasks = sorted({task for task, _ in parents})
    if (len(parents) != 1600 or len(set(parents)) != 1600 or len(tasks) != 16
            or any(n != 100 for n in Counter(t for t, _ in parents).values())
            or len(trained) != 1600 or len(set(trained)) != 1600 or set(trained) != set(parents)
            or any(r.get('split') != 'train' for r in entries)):
        raise ValueError('all exact original1600 TRAIN parents, once each, required')
    if (metadata.get('training_complete') is not True or metadata.get('completed_epochs') != 30
            or metadata.get('TEST_used_for_fitting') is not False
            or metadata.get('normalization_fit_split') != 'train' or metadata.get('training_fit_split') != 'train'
            or sorted(metadata.get('jointly_trained_tasks', [])) != tasks
            or metadata.get('training_source_manifest', {}).get('source_episode_count') != 1600
            or abs(metadata.get('maximum_training_horizon_seconds', -1.)-1.1) > 1e-8):
        raise ValueError('completed original shared16 training metadata required')
    buzz = [r for r in entries if r['task'] in ('beat_the_rotating_buzz', 'beat_the_rotating_buzz_high_speed')]
    if (len(buzz) != 200 or any(r.get('new_conditional_target_rotation_mask') is not True
            or r.get('derived_observed_rotation_channel') != 'r10_causal_public_two_center_no_action_hold_rotation_channel_v77'
            or r.get('context_primary_view') is not True for r in buzz)):
        raise ValueError('all200 original buzz TRAIN parents require the admitted conditional rotation channel')
    return dict(TRAIN_original_parents=1600, tasks=tasks, completed_epochs=30,
        buzz_parents_with_conditional_rotation=200, exact_roster_passed=True)


def verified_checkpoint(root, prepared):
    root = Path(root).resolve()
    protocol, files = prepared['config'], prepared['files']
    permission = protocol['joint16_development_checkpoint_v157']
    if permission.get('formal_release') is not False:
        raise ValueError('this checkpoint consumer is development only')
    acceptance_path = root/permission['acceptance']['path']
    if file_sha(acceptance_path) != permission['acceptance']['sha256']:
        raise ValueError('checkpoint CPU acceptance changed')
    acceptance = json.loads(acceptance_path.read_text())
    if acceptance['checkpoint'] != protocol['context_checkpoint'] or acceptance['executor_qualification_granted'] is not False:
        raise ValueError('actual checkpoint and outstanding executor qualification must be explicit')
    model = MultiBodyContextDynamics(root/protocol['context_checkpoint']['path'])
    if model.checkpoint_sha256 != protocol['context_checkpoint']['sha256']:
        raise ValueError('actual shared checkpoint differs')
    source_path = model.path.parent/model.metadata['training_source_manifest']['path_relative_to_checkpoint']
    source = json.loads(source_path.read_text())
    manifest_path = root/'artifacts/framework/multitask_r10/joint16_buzz_hold_channel_v77/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (file_sha(manifest_path) != model.metadata['track_manifest_sha256']
            or source['body_track_manifest_sha256'] != model.metadata['track_manifest_sha256']
            or manifest['TEST_read'] is not False):
        raise ValueError('shared16 original frozen manifest77 required')
    roster_path = root/manifest['original_sources']['train']['path']
    roster = json.loads(roster_path.read_text())['episodes']
    proof = check_training_metadata_and_roster(model.metadata, source['entries'], roster)
    subset = {str(Path(__file__)): file_sha(__file__), str(acceptance_path): permission['acceptance']['sha256'],
        str(model.path): model.checkpoint_sha256,
        str(source_path): model.metadata['training_source_manifest']['sha256'],
        str(manifest_path): model.metadata['track_manifest_sha256'],
        str(roster_path): manifest['original_sources']['train']['sha256']}
    for key in ('training_report', 'training_sources'):
        pointer = acceptance[key]
        subset[str(root/pointer['path'])] = pointer['sha256']
    for shard in manifest['shards']:
        subset[str(root/shard['path'])] = shard['sha256']
    subset.update(manifest['observation_extension']['inputs_sha256'])
    for row in source['entries']:
        subset[str(root/row['cache'])] = row['cache_sha256']
    checked_subset(root, files, subset)
    return dict(proof, checkpoint=protocol['context_checkpoint'],
        original_manifest_executor_precondition_still_outstanding=True,
        native_smoke_permission_only=True, formal_release=False)
