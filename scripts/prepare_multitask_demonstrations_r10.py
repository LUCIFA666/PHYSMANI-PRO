"""Extract small, grouped TRAIN/VAL demonstration arrays on CPU only.

No TEST manifest, TEST recording, RGB-D image payload, future world-model
prediction, simulator, torch or GPU is opened. Debug target positions are
offline labels with task-dependent semantics, never an online observation.
"""
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/framework/multitask_r10'
REPO = Path('/home/ubuntu/桌面/phsimani/PhysMani')
BENCHMARK = REPO / 'data/rmt/physmani_bench'
CAMERAS = ('front', '8', '16', '36')
EXCLUDED = {'pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed'}
NOMINAL_DT = .05
PACKAGER = REPO / 'packages/physmani_policy/physmani_policy/datasets/dataset_engine.py'
LOW_DIM_FIELDS = {
    'expert_trajectory': 'low_dim_obs/expert_trajectory',
    'expert_stage': 'low_dim_obs/expert_trajectory_stage',
    'expert_debug_target_position': 'low_dim_obs/expert_trajectory_target_position',
    'tcp_pose': 'low_dim_obs/gripper_pose',
    'joint_positions': 'low_dim_obs/joint_positions',
    'gripper_open': 'low_dim_obs/gripper_open',
}
PHASES = {
    'beat_the_rotating_buzz': {'wp0': 'approach', 'wp1': 'acquire', 'wp2': 'follow',
                             'wp3_': 'follow', 'wp3': 'end_release'},
    'insert_onto_rotating_peg': {'wp0': 'approach', 'wp1': 'pregrasp', 'wp1-grasp': 'grasp',
                               'wp2': 'lift', 'wp3': 'align', 'wp4': 'insert', 'wp4-open': 'release'},
    'moving_basketball_in_hoop': {'wp0': 'approach', 'wp1': 'grasp', 'wp2': 'lift', 'wp3': 'carry_release'},
    'place_cups_on_rotating_frame': {'wp0': 'approach', 'wp1': 'grasp', 'wp2': 'lift',
                                    'wp3': 'transport', 'wp4': 'align', 'wp5': 'hang_release'},
    'push_moving_button': {'wp0': 'approach', 'wp1': 'press'},
    'put_rubbish_in_moving_bin': {'wp0': 'approach', 'wp1': 'grasp', 'wp2': 'lift', 'wp3': 'moving_bin_release'},
    'remove_cups_from_rotating_frame': {'wp0': 'approach', 'wp1': 'pregrasp', 'wp2': 'grasp',
                                      'wp3': 'detach', 'wp4': 'transport', 'wp5': 'place_release'},
}
TARGETS = {
    'beat_the_rotating_buzz': dict(expression='task.wand.get_position()', role='wand',
        identity_rule='one named wand object', known_persistent_body=True,
        missing_information=['wand orientation/angular state', 'ring contact/attachment mode'],
        warning='A rotating body center XYZ can stay constant and does not describe rotating contact geometry.'),
    'insert_onto_rotating_peg': dict(expression='task._square_ring.get_position()', role='square_ring',
        identity_rule='one named manipulated square-ring object', known_persistent_body=True,
        missing_information=['rotating peg/frame trajectory', 'ring orientation', 'attachment/contact state'],
        warning='This label follows the manipulated ring, not the moving peg. Held/free/contact phases have different controlled dynamics.'),
    'moving_basketball_in_hoop': dict(expression='[task.ball.get_position(), task.hoop.get_position()]', role='ball',
        identity_rule='the packager retains only row 0 of [ball, hoop]', known_persistent_body=True,
        discarded_source_roles=['hoop'], missing_information=['hoop trajectory', 'ball attachment/free mode'],
        warning='The stored label remains ball in every stage, while the placement reference switches to the hoop in wp3. A hoop trajectory cannot be recovered from this field.'),
    'place_cups_on_rotating_frame': dict(expression='target_cup.get_position()', role='current_indexed_cup',
        target_definition='target_cup = task._cups[task._cups_placed]', known_persistent_body=False,
        identity_rule='cup index is task state and is not stored in these HDF5 fields',
        missing_information=['persistent cup ID', 'frame/spoke pose', 'attachment/contact mode', '_cups_placed counter'],
        warning='The selected cup can change across manipulation cycles. A stage label or its run does not certify persistent body identity.'),
    'push_moving_button': dict(expression='task.target_button.get_position()', role='target_button',
        identity_rule='one named moving target-button assembly', known_persistent_body=True,
        missing_information=['topPlate depression joint', 'contact state', 'button orientation'],
        warning='Target-button XYZ does not measure the pressed topPlate joint or native task success.'),
    'put_rubbish_in_moving_bin': dict(expression='task.bin.get_position()', role='bin',
        identity_rule='one named moving bin object', known_persistent_body=True,
        missing_information=['manipulated rubbish trajectory', 'release/contact state'],
        warning='The stored target is bin throughout. The action objective moves from grasping rubbish to releasing above the moving bin; the label is not a rubbish trajectory.'),
    'remove_cups_from_rotating_frame': dict(expression='target_cup.get_position()', role='nearest_cup_to_tcp',
        target_definition='target_cup = min(task.cups, key=lambda cup: norm(cup.position - tcp.position))',
        known_persistent_body=False, identity_rule='nearest cup is recomputed; identity may switch even within the same stage',
        missing_information=['persistent cup ID', 'frame/spoke pose', 'attachment/contact mode'],
        warning='Nearest-body XYZ can jump between cups without a stage transition. It must not be treated as one body dynamics trajectory.'),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def canonical_digest(arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(json.dumps(dict(name=name, shape=list(array.shape), dtype=array.dtype.str),
                                 sort_keys=True).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def source_stat(path):
    value = path.stat()
    return dict(size_bytes=value.st_size, mtime_ns=value.st_mtime_ns,
                inode=value.st_ino, device=value.st_dev)


def task_family(task):
    return task.removesuffix('_high_speed')


def build_semantics(tasks):
    source_files = {str(PACKAGER): sha(PACKAGER)}
    for relative in ('packages/physmani_bench/physmani_bench/logger.py',
                     'packages/physmani_bench/physmani_bench/contracts.py'):
        source_files[str(REPO / relative)] = sha(REPO / relative)
    definitions = {}
    for task in tasks:
        family = task_family(task)
        path = REPO / 'third_party/RLBench/rlbench/tasks' / (task + '.py')
        source = path.read_text()
        tree = ast.parse(source)
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'get_expert_info')
        target_nodes = [value for node in ast.walk(function) if isinstance(node, ast.Dict)
                        for key, value in zip(node.keys, node.values)
                        if isinstance(key, ast.Constant) and key.value == 'tar_position']
        if len(target_nodes) != 1:
            raise ValueError('debug target expression not unique: ' + task)
        expression = ast.unparse(target_nodes[0])
        if expression != TARGETS[family]['expression']:
            raise ValueError('task debug target source differs: ' + task + ': ' + expression)
        stages = {node.value.value for node in ast.walk(function)
                  if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                  and isinstance(node.value.value, str)
                  and any(isinstance(target, ast.Name) and target.id == 'stage' for target in node.targets)}
        if stages - set(PHASES[family]):
            raise ValueError('source contains unmapped stages: ' + task + ': ' + str(stages))
        definition = dict(family=family, source_path=str(path), source_sha256=sha(path),
            get_expert_info_line_range=[function.lineno, function.end_lineno],
            debug_expression_line_range=[target_nodes[0].lineno, target_nodes[0].end_lineno],
            debug_target_expression=expression, stored_debug_target=TARGETS[family],
            stage_to_phase=PHASES[family], source_assigned_stage_labels=sorted(stages),
            stage_debug_target_semantics={label: dict(action_phase=phase,
                stored_target_role=TARGETS[family]['role'], source_debug_expression=expression,
                identity_rule=TARGETS[family]['identity_rule']) for label,phase in PHASES[family].items()},
            source_stage_is='expert action-plan label, not a persistent body ID or an online policy observation',
            no_continuity_across_contact_modes=True,
            cannot_assume_identity_continuity_within_stage=not TARGETS[family]['known_persistent_body'],
            source_phase_review='aligned with task_adapter_audit TaskSpec phase review; exact source expressions checked by AST')
        definitions[task] = definition
        source_files[str(path)] = definition['source_sha256']
    return dict(version='r10_offline_stage_target_semantics_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), tasks=definitions, source_files_sha256=source_files,
        HDF5_packager_rule=dict(source_path=str(PACKAGER), source_sha256=source_files[str(PACKAGER)],
            line_range=[1242, 1251], source_expression="expert_info['debug_info']['tar_position']",
            retained_array='if shape is (2,3), keep row 0; otherwise keep the full 3-vector',
            output_dataset='low_dim_obs/expert_trajectory_target_position'),
        boundaries=['Debug target and expert actions are offline demonstration labels, not causal observations available to an online controller.',
            'The stored XYZ can name a manipulated body, a destination, or a dynamically chosen cup. It does not provide general object poses.',
            'Do not concatenate episodes, TRAIN/VAL splits, action stages, attachment/contact modes, or unverified identities as one body trajectory.',
            'Stage-run IDs below only identify consecutive equal stage labels. They are not body IDs, attachment labels or certified dynamics segments.',
            'No TEST motion, label, trajectory or reset file is read by this preparation program.'])


def extract_episode(row, split, *, output, semantics, old_recordings):
    task, episode = row['task'], row['raw_episode_id']
    path = Path(row['path']).resolve()
    expected_root = (BENCHMARK / ('train_package_compressed' if split == 'train' else 'val_package_compressed')).resolve()
    if not path.is_relative_to(expected_root) or path.name != 'data.h5':
        raise ValueError('source is outside the permitted TRAIN/VAL root: ' + str(path))
    before = source_stat(path)
    arrays = {}
    dataset_references = {}
    with h5py.File(path, 'r') as source:
        frames = int(source.attrs['num_frames'])
        if str(source.attrs['task']) != task:
            raise ValueError('source task attribute differs')
        variation = int(source.attrs['variation'])
        for name, key in LOW_DIM_FIELDS.items():
            # Only these six small arrays are read, never image/WM datasets.
            value = source[key][()]
            if name == 'expert_stage':
                raw = value.reshape(-1)
                value = np.asarray([stage.decode('utf-8') if isinstance(stage, bytes) else str(stage)
                                    for stage in raw], dtype=np.str_)
            arrays[name] = value
            dataset_references[name] = dict(source_dataset=key, shape=list(value.shape), dtype=value.dtype.str)
        camera_arrays = {}
        for field in ('intrinsics', 'extrinsics', 'near', 'far'):
            camera_arrays[field] = np.stack([source[f'cam_params/{camera}/{field}'][()] for camera in CAMERAS])
        arrays.update(camera_intrinsics=camera_arrays['intrinsics'], camera_extrinsics=camera_arrays['extrinsics'],
                      camera_near_far=np.column_stack((camera_arrays['near'], camera_arrays['far'])),
                      camera_names=np.asarray(CAMERAS))
    after = source_stat(path)
    if before != after:
        raise ValueError('source changed during the read: ' + str(path))
    expected_shapes = {'expert_trajectory': (frames, 1, 8), 'expert_stage': (frames,),
        'expert_debug_target_position': (frames, 3), 'tcp_pose': (frames, 7),
        'joint_positions': (frames, 7), 'gripper_open': (frames,)}
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError('unexpected shape for ' + name + ': ' + str(arrays[name].shape))
    for name, value in arrays.items():
        if value.dtype.kind not in ('U', 'S') and not np.isfinite(value).all():
            raise ValueError('nonfinite values: ' + str(path) + ': ' + name)
    original_digest = canonical_digest(arrays)
    gripper = np.column_stack((arrays['tcp_pose'], arrays['gripper_open'])).astype('<f4')
    gripper_sha = hashlib.sha256(gripper.tobytes()).hexdigest()
    prior = old_recordings.get((split, task, episode))
    if prior and (prior['frames'] != frames or prior['gripper_sha256'] != gripper_sha):
        raise ValueError('small-array trajectory differs from the existing read-only recording fingerprint: ' + str(path))
    stage = arrays['expert_stage']
    stage_change = np.r_[True, stage[1:] != stage[:-1]]
    phases = semantics['tasks'][task]['stage_to_phase']
    unknown = sorted(set(stage) - set(phases))
    arrays.update(expert_tcp_pose=arrays['expert_trajectory'][:, 0, :7].copy(),
                  expert_gripper_open=arrays['expert_trajectory'][:, 0, 7].copy(),
                  frame_index=np.arange(frames, dtype=np.int32),
                  timestamps_seconds=np.arange(frames, dtype=np.float64) * NOMINAL_DT,
                  nominal_dt_seconds=np.asarray(NOMINAL_DT),
                  stage_run_id=np.cumsum(stage_change).astype(np.int32)-1,
                  stage_transition=stage_change,
                  expert_phase=np.asarray([phases.get(label, 'UNRESOLVED') for label in stage]))
    destination = output / split / task / (episode + '.npz')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(destination)
    # Confirm exact round-trip for every array with object deserialization disabled.
    with np.load(destination, allow_pickle=False) as loaded:
        if set(loaded.files) != set(arrays) or any(not np.array_equal(loaded[name], value) for name, value in arrays.items()):
            raise ValueError('NPZ round-trip failed: ' + str(destination))
    record = dict(split=split, task=task, source_episode=episode,
        source_group=f'{split}:{task}:{episode}', source_path=str(path), source_file_stat=before,
        prior_declared_source_sha256=row.get('sha256', row.get('source_sha256')),
        whole_HDF5_sha256_recomputed=False, whole_HDF5_sha256=None,
        source_integrity_boundary='stat before/after plus canonical hashes of all extracted small arrays; whole HDF5 image/WM payload not hashed',
        frames=frames, variation=variation, npz_path=str(destination.relative_to(ROOT)),
        npz_sha256=sha(destination), npz_bytes=destination.stat().st_size,
        extracted_source_arrays_sha256=original_digest, observed_gripper_trajectory_sha256=gripper_sha,
        prior_recording_fingerprint_matched=bool(prior), source_dataset_references=dataset_references,
        camera_metadata_references={camera: {field: f'cam_params/{camera}/{field}'
                                            for field in ('intrinsics','extrinsics','near','far')} for camera in CAMERAS},
        timestamp_definition='relative frame_index * declared 0.05 s; reconstructed nominal clock, not recorded absolute or measured simulator time',
        stage_counts=dict(sorted(Counter(stage).items())), stage_runs=int(stage_change.sum()),
        unknown_stage_labels=unknown, stored_debug_target_role=semantics['tasks'][task]['stored_debug_target']['role'],
        persistent_body_identity_certified=False, body_dynamics_trajectory_ready=False,
        training_labels_are_privileged=True, formal_TEST_used=False,
        output_array_schema={name: dict(shape=list(value.shape), dtype=value.dtype.str) for name, value in arrays.items()})
    write(destination.with_suffix('.json'), record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='artifacts/framework/multitask_r10/demonstrations')
    args = parser.parse_args()
    output = (ROOT / args.out).resolve()
    if not output.is_relative_to(BASE.resolve()):
        raise ValueError('all new output must stay inside multitask_r10')
    paths = {split: ROOT / f'artifacts/framework/protocol/{split}_manifest.json'
             for split in ('train', 'validation')}
    manifest_hashes = {split: sha(path) for split, path in paths.items()}
    source_rows = {split: [row for row in json.loads(path.read_text())['episodes'] if row['task'] not in EXCLUDED]
                   for split, path in paths.items()}
    tasks = sorted({row['task'] for row in source_rows['train']})
    if len(tasks) != 14 or len(source_rows['train']) != 1400 or len(source_rows['validation']) != 275:
        raise ValueError('expected all 14 released TRAIN1400/VAL275 sources')
    if {row['task'] for row in source_rows['validation']} != set(tasks):
        raise ValueError('TRAIN/VAL task sets differ')
    semantics = build_semantics(tasks)
    write(BASE / 'stage_target_semantics.json', semantics)
    fingerprints_path = ROOT / 'artifacts/framework/protocol/recording_fingerprints.json'
    fingerprints = json.loads(fingerprints_path.read_text())
    # Only existing TRAIN/VAL entries are referenced; never access TEST rows.
    old_recordings = {(split, row['task'], row['raw_episode_id']): row
                      for split in ('train','validation') for row in fingerprints[split] if row['task'] in tasks}
    identity = dict(program_sha256=sha(Path(__file__)), manifests_sha256=manifest_hashes,
                    semantic_source_files_sha256=semantics['source_files_sha256'],
                    prior_recording_fingerprints_sha256=sha(fingerprints_path), nominal_dt_seconds=NOMINAL_DT)
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    previous = output / 'manifest.json'
    if previous.exists() and json.loads(previous.read_text()).get('inputs_sha256') != identity_sha:
        raise ValueError('preparation inputs changed; choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    report = dict(version='multitask_r10_grouped_small_demonstrations_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), inputs_sha256=identity_sha, inputs=identity,
        expected=dict(train=1400, validation=275), complete=False, episodes=rows,
        semantic_audit='artifacts/framework/multitask_r10/stage_target_semantics.json',
        TEST_trajectories_opened=False, TEST_used_for_training_calibration_or_selection=False,
        image_payloads_read=False, GPU_used=False, simulator_started=False,
        nominal_timestamp_warning='HDF5 has no actual timestamp field; frame-index relative nominal timestamps are reconstructed, explicitly not measured wall or simulator clocks.',
        interpretation='Prepared offline demonstration labels only; no dynamics or success model was trained, and cup/body identity or attachment labels were not fabricated.')
    try:
        for split in ('train','validation'):
            for task in tasks:
                selected = [row for row in source_rows[split] if row['task'] == task]
                for source_row in selected:
                    rows.append(extract_episode(source_row, split, output=output, semantics=semantics,
                                                old_recordings=old_recordings))
                report['completed'] = dict(Counter(row['split'] for row in rows))
                report['created_utc'] = datetime.now(timezone.utc).isoformat()
                write(output / 'manifest.json', report)
                print(json.dumps(dict(split=split, task=task, prepared=len(selected), total=len(rows))), flush=True)
        train_rows = [row for row in rows if row['split'] == 'train']
        val_rows = [row for row in rows if row['split'] == 'validation']
        train_paths = {row['source_path'] for row in train_rows}
        val_paths = {row['source_path'] for row in val_rows}
        train_signatures = {(row['task'], row['extracted_source_arrays_sha256']) for row in train_rows}
        val_signatures = {(row['task'], row['extracted_source_arrays_sha256']) for row in val_rows}
        checks = dict(source_groups_unique=len({row['source_group'] for row in rows}) == len(rows),
                      source_paths_disjoint=not train_paths & val_paths,
                      small_array_content_signatures_disjoint=not train_signatures & val_signatures,
                      all_prior_gripper_fingerprints_match=all(row['prior_recording_fingerprint_matched'] for row in rows),
                      all_observed_stages_source_mapped=all(not row['unknown_stage_labels'] for row in rows),
                      source_manifests_unchanged=all(sha(path) == manifest_hashes[split] for split,path in paths.items()),
                      author_semantic_sources_unchanged=all(sha(Path(path)) == digest for path,digest in semantics['source_files_sha256'].items()))
        report.update(complete=all(checks.values()) and len(train_rows)==1400 and len(val_rows)==275,
                      checks=checks, completed=dict(Counter(row['split'] for row in rows)),
                      total_frames=sum(row['frames'] for row in rows),
                      total_npz_bytes=sum(row['npz_bytes'] for row in rows),
                      per_task={task: {split: dict(episodes=len(local := [row for row in rows if row['task']==task and row['split']==split]),
                           frames=sum(row['frames'] for row in local),
                           stages=dict(sum((Counter(row['stage_counts']) for row in local), Counter())))
                           for split in ('train','validation')} for task in tasks})
        write(output / 'manifest.json', report)
        if not report['complete']:
            raise ValueError('preparation checks did not all pass: ' + str(checks))
        print(json.dumps(dict(complete=True, episodes=len(rows), frames=report['total_frames'],
                             npz_bytes=report['total_npz_bytes'], checks=checks)), flush=True)
    except Exception as error:
        report.update(complete=False, error=repr(error), completed=dict(Counter(row['split'] for row in rows)))
        write(output / 'manifest.json', report)
        raise


if __name__ == '__main__':
    main()
