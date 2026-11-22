"""Explicitly gated R10 TRAIN augmentation and non-teacher VAL body tracks.

Nothing native is imported by the planning, merging or CPU-test interfaces.
Every native case requires the completed 400-case R09 protocol acceptance.
The original HDF5 demonstrations and generated trajectories have independent
budgets and identities; generated motion is never called an archive replay.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from our_mani.multitask.mode_observer import (AttachmentModeObserver, PublicCommandSegments, MODE_IDS)
BASE = ROOT / 'artifacts/framework/multitask_r10'
PLAN_VERSION = 'r10_source_grouped_native_augmentation_plan_v1'
BEHAVIOR_PLAN_VERSION = 'r10_source_grouped_behavior_augmentation_plan_v2'
TRACK_VERSION = 'r10_causal_body_tracks_v1'
TEACHER_BEHAVIOR='TRAIN_teacher_committed_waypoint'
TRAIN_HOLD_BEHAVIOR='TRAIN_hold_motion_identification'
VAL_HOLD_BEHAVIOR='VAL_hold_motion_identification'
COLLECTION_STOP = ROOT/'runtime/framework/multitask_r10_STOP'


class CollectionError(ValueError):
    pass


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def destination(path):
    path = Path(path).resolve()
    if not path.is_relative_to(BASE.resolve()):
        raise CollectionError('all collection outputs must stay in desktop our-mani/multitask_r10')
    return path


def write_json(path, value, *, overwrite=False):
    path = destination(path)
    if path.exists() and not overwrite:
        raise CollectionError('refusing to overwrite an existing artifact: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def optional_geometry_input_files(protocol):
    """Bind enabled frozen geometry and all declared proof/source dependencies."""
    from our_mani.multitask.specs import TASK_SPECS
    paths = {}
    def add(name, expected):
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != expected:
            raise CollectionError('enabled geometry artifact/source dependency changed: ' + str(path))
        if str(path) in paths and paths[str(path)] != expected:
            raise CollectionError('enabled geometry dependencies disagree: ' + str(path))
        paths[str(path)] = expected
        return path
    visited = set()
    def artifact(pointer):
        if not isinstance(pointer, dict) or not {'path', 'sha256'} <= set(pointer):
            raise CollectionError('enabled geometry requires an exact frozen path and SHA256')
        path = add(pointer['path'], pointer['sha256'])
        if path in visited:
            return
        visited.add(path)
        value = json.loads(path.read_text())
        for field in ('inputs_sha256', 'construction_sources_sha256'):
            for name, expected in value.get(field, {}).items():
                add(name, expected)
        for field in ('source_asset', 'source_certificate', 'analytic_proof', 'engineering_inflation_proof'):
            if field in value:
                item = value[field]
                if not isinstance(item, dict) or not {'path', 'sha256'} <= set(item):
                    # Earlier evidence reports also use these names for
                    # ordinary text paths/descriptions. Their declared file
                    # hashes were already bound above; they are not pointers.
                    continue
                # Text proofs are still immutable inputs but are not JSON.
                nested = add(item['path'], item['sha256'])
                if nested.suffix == '.json':
                    artifact(item)
    for field in ('axis_swept_assets', 'finite_rotation_symmetries'):
        groups = protocol.get(field, {})
        if not isinstance(groups, dict):
            raise CollectionError('enabled geometry permissions require explicit task dictionaries')
        for task, roles in groups.items():
            if task not in TASK_SPECS or not isinstance(roles, dict):
                raise CollectionError('enabled geometry permission names an unknown task')
            for pointer in roles.values():
                artifact(pointer)
    for task,permission in protocol.get('cup_feature_priors', {}).items():
        if task not in TASK_SPECS:
            raise CollectionError('cup feature permission names an unknown task')
        artifact(permission)
    return tuple(Path(name) for name in paths)


def require_r09_release(path):
    """Fail before any native import; failed VAL screen is not a bypass."""
    path = Path(path).resolve()
    if not path.is_file():
        raise CollectionError('R09 final_report.json is missing; native R10 execution is locked')
    report = json.loads(path.read_text())
    if (report.get('complete') is not True or report.get('protocol_acceptance_passed') is not True
            or report.get('total_test_runs') != 400 or report.get('expected_test_runs') != 400
            or report.get('acceptance_errors')):
        raise CollectionError('native R10 execution requires complete R09 400/400 and hard protocol acceptance')
    return dict(path=str(path), sha256=sha(path), complete=True,
                protocol_acceptance_passed=True, total_test_runs=400,
                development_performance_screen_passed=report.get('development_performance_screen_passed'),
                benefit_validated_by_this_gate=False)


def source_identity(row):
    return row['split'], row['task'], row.get('source_episode', row.get('raw_episode_id'))


def demonstration_registry(path):
    """Read metadata only; never read VAL expert arrays during planning."""
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if (value.get('version') != 'multitask_r10_grouped_small_demonstrations_v1'
            or value.get('complete') is not True or not value.get('checks')):
        raise CollectionError('completed, audited small-demonstration manifest required')
    rows = value.get('episodes', [])
    identities = [source_identity(row) for row in rows]
    if (len(set(identities)) != len(rows) or any(split not in ('train', 'validation') for split, _, _ in identities)
            or Counter(split for split, _, _ in identities) != {'train': 1400, 'validation': 275}):
        raise CollectionError('need the real 1400 TRAIN / 275 VAL sources, preserving archive shortfalls')
    for row in rows:
        split, task, raw = source_identity(row)
        source = Path(row['source_path']).resolve()
        package = 'train_package_compressed' if split == 'train' else 'val_package_compressed'
        if package not in source.parts or source.name != 'data.h5' or not raw or not task:
            raise CollectionError('source split/path identity differs')
        cache = (ROOT / row['npz_path']).resolve()
        if not cache.is_relative_to(BASE.resolve()) or not cache.is_file():
            raise CollectionError('small-demonstration cache missing/outside new R10 artifacts')
    return value, rows


def validate_collection_protocol(protocol):
    from our_mani.multitask.protocol import validate_protocol
    validate_protocol(protocol)
    from our_mani.multitask.reference_overrides import validate_overrides,ReferenceOverrideError
    try:validate_overrides(protocol,root=ROOT)
    except ReferenceOverrideError as error:raise CollectionError(str(error)) from error
    if 'dynamics_target_label_schema' in protocol:
        from our_mani.multitask.motion_labels import SCHEMA
        if protocol['dynamics_target_label_schema']!=SCHEMA:
            raise CollectionError('unknown explicit dynamics target label schema')
    if (protocol.get('shared_semantic_instance_binding') is not True
            or protocol.get('shared_TRAIN_CAD_geometry') is not True
            or protocol.get('semantic_binding_available_to_baseline') is not True):
        raise CollectionError('declare semantic mask-role bindings and TRAIN CAD geometry for both methods')
    distance = protocol.get('attachment_association_distance_m')
    if not isinstance(distance, (int, float)) or not np.isfinite(distance) or distance <= 0:
        raise CollectionError('declare controller attachment association distance; ambiguous IDs will be masked')
    if protocol.get('TRAIN_teacher_opening_gate_enabled') is True:
        validate_teacher_timing(protocol.get('gripper_timing_config'))
    if 'collection_behavior_budgets' in protocol:
        behavior_profiles(protocol,'train');behavior_profiles(protocol,'validation')
        margin=protocol.get('joint_limit_recovery_margin_rad')
        if type(margin) not in (int,float) or not np.isfinite(margin) or margin<=0:
            raise CollectionError('explicit positive public joint-limit recovery margin required')
        settling=protocol.get('TRAIN_teacher_release_settle_ticks')
        if type(settling) is not int or not 0<=settling<=200:
            raise CollectionError('explicit counted release settling budget required')


def behavior_profiles(protocol,split):
    profiles=protocol.get('collection_behavior_budgets',{}).get(split)
    expected={TEACHER_BEHAVIOR:(200,True,False,True),TRAIN_HOLD_BEHAVIOR:(60,False,True,False)} if split=='train' else {
        VAL_HOLD_BEHAVIOR:(60,False,True,False)}
    if not isinstance(profiles,dict) or set(profiles)!=set(expected):
        raise CollectionError('declare fixed teacher+hold TRAIN and nonteacher hold VAL behavior inventories')
    for name,(ticks,teacher,context,interaction) in expected.items():
        row=profiles[name]
        if (set(row)!={'replicas','replica_indices','maximum_ticks','teacher_used','context_primary_view','interaction_eligible'}
                or row['replicas']!=1 or type(row['replicas']) is not int or row['maximum_ticks']!=ticks
                or type(row['maximum_ticks']) is not int
                or row['replica_indices']!=[0] or any(type(index) is not int for index in row['replica_indices'])
                or row['teacher_used'] is not teacher or row['context_primary_view'] is not context
                or row['interaction_eligible'] is not interaction):
            raise CollectionError('behavior budget/teacher/context/interaction eligibility differs: '+name)
    return profiles


def validate_teacher_timing(configuration):
    names={'close_position_tolerance_m','close_orientation_tolerance_rad'}
    if (not isinstance(configuration,dict) or set(configuration) != names
            or any(type(configuration[name]) not in (int,float) or not np.isfinite(configuration[name])
                   or configuration[name] <= 0 for name in names)):
        raise CollectionError('explicit finite positive TRAIN/public gripper waypoint timing tolerances required')
    return dict(configuration)


def build_plan(demonstrations_path, protocol_path, output, *, split='train', replicas=1,
               tasks=(), validation_mode='hold', template_assets=None, frozen_policy=None, rigid_layouts=None,
               offline_priors_path=None, sources_per_task=None):
    from our_mani.multitask.specs import get_task_spec
    if split not in ('train', 'validation') or type(replicas) is not int or replicas < 1:
        raise CollectionError('TRAIN/VAL split and positive integer replica budget required')
    manifest, all_rows = demonstration_registry(demonstrations_path)
    protocol = json.loads(Path(protocol_path).read_text())
    validate_collection_protocol(protocol)
    behavioral='collection_behavior_budgets' in protocol
    profiles=behavior_profiles(protocol,split) if behavioral else None
    if behavioral and replicas!=1:
        raise CollectionError('v5 preregisters one replica per behavior; do not override via generic K')
    if validation_mode not in ('hold', 'frozen_policy'):
        raise CollectionError('VAL accepts only hold or explicitly frozen causal policy')
    rows = [row for row in all_rows if row['split'] == split and (not tasks or row['task'] in tasks)]
    if sources_per_task is not None:
        if type(sources_per_task) is not int or sources_per_task < 1:
            raise CollectionError('sources_per_task must be a positive explicit subset budget')
        seen = Counter()
        selected = []
        for row in rows:
            if seen[row['task']] < sources_per_task:
                selected.append(row); seen[row['task']] += 1
        rows = selected
    if protocol.get('execution_scope') == 'TRAIN_smoke_only' and (
            split != 'train' or sources_per_task != 1 or replicas != 1):
        raise CollectionError('unverified-prior smoke config permits one TRAIN source/replica per task only')
    if not rows:
        raise CollectionError('no requested source cases')
    template_assets = {} if template_assets is None else template_assets
    rigid_layouts = {} if rigid_layouts is None else rigid_layouts
    if split == 'validation' and set(row['task'] for row in rows) - set(template_assets):
        raise CollectionError('VAL requires pre-existing TRAIN template assets for every task')
    if split == 'validation' and validation_mode == 'frozen_policy' and not frozen_policy:
        raise CollectionError('frozen-policy VAL needs an explicit SHA-bound policy factory')
    input_paths = [Path(demonstrations_path).resolve(), Path(protocol_path).resolve(), Path(__file__).resolve()]
    input_paths.append(ROOT/'scripts/run_local_sim.sh')
    input_paths += [ROOT / 'our_mani/multitask' / name for name in
                    ('perception.py', 'training_geometry.py', 'simulation.py', 'native_tasks.py', 'specs.py',
                     'maps.py', 'protocol.py', 'mode_observer.py', 'forecasting.py', 'context_se3.py',
                     'safe_native_mesh.py', 'component_flags.py', 'collision_geometry.py', 'reference_overrides.py',
                     'axis_evidence.py', 'depth_pose_consistency.py', 'rotation_equivalence.py',
                     'bin_swept_geometry.py', 'bin_swept_assets.py', 'cup_features.py', 'factory.py')]
    input_paths.extend(optional_geometry_input_files(protocol))
    if 'TRAIN_reference_kind_overrides' in protocol:
        from our_mani.multitask.reference_overrides import validate_overrides
        _,reference_inputs=validate_overrides(protocol,root=ROOT)
        input_paths.append(ROOT/'our_mani/multitask/reference_overrides.py')
        input_paths.extend(Path(path) for path in reference_inputs)
    if behavioral:
        input_paths.append(ROOT/'our_mani/multitask/train_teacher.py')
        if protocol.get('dynamics_target_label_schema') is not None:
            input_paths.append(ROOT/'our_mani/multitask/motion_labels.py')
        input_paths.extend((ROOT/'our_mani/contracts.py',ROOT/'our_mani/reachability.py'))
        experiment_path=Path(protocol.get('experiment_plan_path','')).resolve()
        if (not experiment_path.is_relative_to(ROOT) or not experiment_path.is_file()
                or sha(experiment_path)!=protocol.get('experiment_plan_sha256')):
            raise CollectionError('v5 behaviors require the explicit SHA-bound R10 full experiment plan')
        input_paths.append(experiment_path)
        native=Path(protocol['baseline_repo'])/'third_party/PyRep/pyrep'
        input_paths.extend(native/name for name in ('robots/configuration_paths/arm_configuration_path.py',
            'robots/configuration_paths/configuration_path.py','robots/arms/arm.py',
            'robots/end_effectors/gripper.py','const.py','errors.py'))
    for task, path in template_assets.items():
        asset = json.loads(Path(path).read_text())
        if asset.get('split') != 'train' or asset.get('task') != task:
            raise CollectionError('VAL geometry must be a TRAIN artifact for this same task')
        input_paths.append(Path(path).resolve())
    for task, path in rigid_layouts.items():
        load_rigid_constraints(path, task)
        input_paths.append(Path(path).resolve())
    if offline_priors_path is not None:
        priors = json.loads(Path(offline_priors_path).read_text())
        if (priors.get('version') != 'r10_original_TRAIN_action_phase_priors_v1'
                or priors.get('split') != 'train' or priors.get('sources_read') != 1400
                or priors.get('demonstration_manifest_sha256') != sha(demonstrations_path)):
            raise CollectionError('offline priors must bind all 1400 original TRAIN sources')
        input_paths.append(Path(offline_priors_path).resolve())
    if frozen_policy:
        if set(frozen_policy) != {'module', 'factory', 'source_path', 'source_sha256', 'artifact_path', 'artifact_sha256'}:
            raise CollectionError('frozen policy needs factory code and fitted artifact SHA receipts')
        for prefix in ('source', 'artifact'):
            path = Path(frozen_policy[prefix+'_path']).resolve()
            if sha(path) != frozen_policy[prefix+'_sha256']:
                raise CollectionError('frozen policy file changed')
            input_paths.append(path)
    annotations = protocol.get('body_orientation_observability', {})
    cases = []
    for row in rows:
        _, task, raw = source_identity(row)
        spec = get_task_spec(task)
        if behavioral:
            native_task=Path(protocol['baseline_repo'])/spec.native_source_relative_path
            if sha(native_task)!=spec.native_source_sha256:raise CollectionError('author task source differs: '+task)
            input_paths.append(native_task)
        physical = {role.name for role in spec.roles if role.kind == 'body'} - set(spec.audit_only_goal_roles)
        if set(annotations.get(task, {})) != physical or any(
                mode not in ('full', 'axis', 'none') for mode in annotations[task].values()):
            raise CollectionError('explicit per-body symmetry annotation required for ' + task)
        axes=protocol.get('body_symmetry_axes',{}).get(task,{})
        if set(axes)!={body for body,annotation in annotations[task].items() if annotation=='axis'}:
            raise CollectionError('explicit symmetry axes must match exactly axis-annotated bodies: '+task)
        if any(np.asarray(axis).shape!=(3,) or not np.isfinite(axis).all()
               or abs(np.linalg.norm(axis)-1)>1e-6 for axis in axes.values()):
            raise CollectionError('TRAIN axis prior must be a finite unit vector: '+task)
        inventory=[(behavior,profile,replica) for behavior,profile in profiles.items()
                   for replica in profile['replica_indices']] if behavioral else [(None,None,r) for r in range(replicas)]
        for behavior,profile,replica in inventory:
            augmentation_id=raw+f'__{behavior}_r{replica}' if behavioral else raw+f'__native_aug_r{replica}'
            identity=dict(split=split,task=task,raw=raw,replica=replica)
            if behavioral:identity['behavior']=behavior
            key=canonical_sha(identity)[:16]
            cases.append(dict(case_id=key, split=split, task=task, raw_episode_id=augmentation_id,
                parent_raw_episode_id=raw, parent_source_group=f'{split}:{task}:{raw}', replica=replica,
                source=dict(task=task, raw_episode_id=raw, path=row['source_path'], variation=row['variation']),
                offline_demonstration=dict(path=row['npz_path'], sha256=row['npz_sha256'],
                    frames=row['frames'], source_file_stat=row['source_file_stat'],
                    whole_HDF5_sha256_recomputed=row['whole_HDF5_sha256_recomputed']),
                template_asset=str(Path(template_assets[task]).resolve()) if task in template_assets else None))
            cases[-1]['rigid_layout'] = str(Path(rigid_layouts[task]).resolve()) if task in rigid_layouts else None
            if behavioral:cases[-1].update(behavior=behavior,behavior_replica=replica,
                maximum_collection_ticks=profile['maximum_ticks'],teacher_used=profile['teacher_used'],
                context_primary_view=profile['context_primary_view'],interaction_eligible=profile['interaction_eligible'],
                native_task_source_sha256=spec.native_source_sha256,
                paired_reset_seed_rule='same source-derived reset seed/variation/camera for equal parent and replica across behaviors')
    payload = dict(version=BEHAVIOR_PLAN_VERSION if behavioral else PLAN_VERSION, created_utc=utc(), split=split, cases=cases,
        inputs={str(path): sha(path) for path in sorted(set(input_paths))},
        protocol_path=str(Path(protocol_path).resolve()), protocol=protocol,
        original_demonstration_budget=dict(train_sources=1400, validation_sources=275,
            TRAIN_usage='original source-grouped expert TCP/gripper/stage supervision retained as offline priors; not body SE3 labels',
            original_sources_fitted_into_interaction_library=False,
            offline_action_phase_priors=str(Path(offline_priors_path).resolve()) if offline_priors_path else None,
            offline_action_phase_priors_fitted=offline_priors_path is not None,
            original_body_SE3_unavailable=True, validation_expert_actions_used=False,
            source_integrity='small-cache SHA and original manifest stat/array hashes; original full HDF5 not rehashed by planner'),
        augmentation_budget=dict(selected_parent_sources=len(rows), replicas=replicas,
            sources_per_task_limit=sources_per_task,
            planned_native_cases=len(cases), maximum_ticks_per_case=200, maximum_seconds_per_case=10.,
            completed_native_cases=0, native_started=False, original_HDF5_motion_replayed=False),
        validation_mode=validation_mode if split == 'validation' else None, frozen_policy=frozen_policy,
        execution_gate='complete R09 final_report 400/400, hard protocol acceptance before native import',
        grouping_warning='multiple replicas share parent_source_group; split/bootstrap by that parent, not independent augmentation IDs',
        information_setting='shared current RGB-D/current instance masks + semantic instance bindings + TRAIN native CAD mesh + shared static map; stronger than raw RGB-D',
        teacher_scope='TRAIN only: author get_expert_info(bool_return_path=False); hidden schedules/waypoints only in teacher supervision, never policy history',
        no_native_or_GPU_work_performed=True)
    if behavioral:
        payload['augmentation_budget'].update(version='r10_fixed_behavior_budget_v1',replicas=None,
            behaviors=profiles,selected_parent_sources=len(rows),planned_native_cases=len(cases),
            planned_cases_by_behavior={name:len(rows)*profile['replicas'] for name,profile in profiles.items()},
            maximum_physics_ticks_per_parent=sum(profile['maximum_ticks']*profile['replicas'] for profile in profiles.values()),
            parent_source_count_does_not_double_across_behaviors=True)
        payload['context_primary_behavior']=TRAIN_HOLD_BEHAVIOR if split=='train' else VAL_HOLD_BEHAVIOR
        payload['teacher_command_semantics']='committed waypoint/path supervision; held frames are not independent expert decisions'
        payload['teacher_scope']='TRAIN only get_expert_info(bool_return_path=False) plus owned pure arm.get_path with declared author family parameters; complete original arm path, no task.get_path/modify_path/hidden stepping'
        payload['experiment_plan']=dict(path=str(experiment_path),sha256=sha(experiment_path))
    payload['plan_sha256'] = canonical_sha(payload)
    write_json(output, payload)
    return payload


def prepare_offline_priors(demonstrations_path, output):
    """Fit small empirical action/phase priors using every original TRAIN file.

    These world-frame command/proprio summaries are offline supervision, not a
    completed body-relative skill or a body-dynamics model. No VAL expert,
    image, TEST, simulator or future world-model payload is read.
    """
    from our_mani.multitask.specs import get_task_spec
    _, rows = demonstration_registry(demonstrations_path)
    train = [row for row in rows if row['split'] == 'train']
    accumulated, sources = defaultdict(lambda: defaultdict(list)), []
    for row in train:
        _, task, raw = source_identity(row)
        path = ROOT/row['npz_path']
        if sha(path) != row['npz_sha256']:
            raise CollectionError('original TRAIN demonstration cache changed')
        stage_phases = {stage: phase.name for phase in get_task_spec(task).phase_skills
                        for stage in phase.TRAIN_expert_stages}
        with np.load(path, allow_pickle=False) as cache:
            stages = np.array(cache['expert_stage'])
            desired, measured = np.array(cache['expert_tcp_pose']), np.array(cache['tcp_pose'])
            opening = np.array(cache['expert_gripper_open'])
            phases = np.asarray([stage_phases[str(stage)] for stage in stages])
            for phase in sorted(set(phases)):
                selected = phases == phase
                delta = desired[selected, :3]-measured[selected, :3]
                desired_r = Rotation.from_quat(desired[selected, 3:]).as_matrix()
                measured_r = Rotation.from_quat(measured[selected, 3:]).as_matrix()
                angular = Rotation.from_matrix(desired_r @ np.swapaxes(measured_r, -1, -2)).as_rotvec()
                accumulated[task][phase].append(dict(source_group=row['source_group'],
                    frames=int(selected.sum()), nominal_seconds=float(selected.sum()*.05),
                    mean_delta=delta.mean(0), median_distance=float(np.median(np.linalg.norm(delta, axis=1))),
                    mean_angular_delta=angular.mean(0), opening_fraction=float((opening[selected] >= .5).mean())))
        sources.append(dict(source_group=row['source_group'], path=row['npz_path'], sha256=row['npz_sha256'],
            original_HDF5_source_path=row['source_path'], original_HDF5_whole_file_hashed=False,
            original_source_stat=row['source_file_stat']))
    per_task = {}
    for task, phases in accumulated.items():
        per_task[task] = {}
        for phase, samples in phases.items():
            per_task[task][phase] = dict(source_count=len(samples), frames=sum(s['frames'] for s in samples),
                weighting='equal original source per phase',
                nominal_duration_quantiles_seconds=np.quantile([s['nominal_seconds'] for s in samples], [.1,.5,.9]).tolist(),
                mean_command_minus_measured_TCP_world_m=np.mean([s['mean_delta'] for s in samples], axis=0).tolist(),
                median_source_command_distance_m=float(np.median([s['median_distance'] for s in samples])),
                mean_command_relative_rotation_vector_world_rad=np.mean([s['mean_angular_delta'] for s in samples], axis=0).tolist(),
                mean_source_open_fraction=float(np.mean([s['opening_fraction'] for s in samples])))
    payload = dict(version='r10_original_TRAIN_action_phase_priors_v1', created_utc=utc(), split='train',
        sources_read=len(sources), demonstration_manifest_path=str(Path(demonstrations_path).resolve()),
        demonstration_manifest_sha256=sha(demonstrations_path), sources=sources, per_task=per_task,
        actual_HDF5_sources_used_as_original_demonstrations=1400,
        source_arrays='previous exact low-dimensional TRAIN extraction, SHA-verified on this read',
        body_relative_interaction_library_fitted=False, body_SE3_labels_available=False,
        model_or_policy_training_performed=False, VAL_expert_arrays_read=False, TEST_read=False,
        future_debug_target_XYZ_used=False, simulator_started=False, GPU_used=False,
        timing='nominal archive frame index * 0.05 s; not a measured simulator clock',
        scope='empirical source-weighted offline action/phase prior statistics; not a learned successful interaction policy')
    payload['artifact_sha256'] = canonical_sha(payload)
    write_json(output, payload)
    return payload


def verify_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get('version') not in (PLAN_VERSION,BEHAVIOR_PLAN_VERSION) or plan.get('plan_sha256') != canonical_sha(
            {key: value for key, value in plan.items() if key != 'plan_sha256'}):
        raise CollectionError('collection plan identity changed')
    validate_collection_protocol(plan['protocol'])
    for name, expected in plan['inputs'].items():
        if sha(name) != expected:
            raise CollectionError('collection input changed: ' + name)
    return plan


def body_row(states, metadata, body_ids, modes, invalid_modes):
    b = len(body_ids)
    row = dict(positions=np.zeros((b, 3)), rotations=np.tile(np.eye(3), (b, 1, 1)),
        linear_velocities=np.zeros((b, 3)), angular_velocities=np.zeros((b, 3)),
        confidence=np.zeros(b), valid=np.zeros(b, bool), modes=np.zeros(b, np.int8),
        rotation_observable=np.zeros(b, bool))
    index = {body: i for i, body in enumerate(body_ids)}
    for body, mode in modes.items():
        row['modes'][index[body]] = MODE_IDS[mode]
    for state in states:
        if state.object_id not in index:
            raise CollectionError('tracker emitted an undeclared body identity')
        i = index[state.object_id]
        diagnostic = metadata['bodies'][state.object_id]
        # Indirect rigid/occluded estimates are useful diagnostics, not fresh
        # directly visible observations for the dynamics training cache.
        valid = state.visible and diagnostic['registration_valid'] and state.object_id not in invalid_modes
        row['positions'][i], row['rotations'][i] = state.pose.position, state.pose.rotation
        row['linear_velocities'][i], row['angular_velocities'][i] = state.linear_velocity, state.angular_velocity
        row['confidence'][i], row['valid'][i] = state.confidence, valid
        row['rotation_observable'][i] = valid and bool(diagnostic['orientation_observable'])
    return row


def context_window_eligibility(arrays, steps=22, stride=2):
    """Report data usability without fitting or changing acquisition outcomes."""
    times = arrays['times']
    per_body = {}
    for body, name in enumerate(map(str, arrays['body_ids'])):
        valid, modes = arrays['valid'][:, body], arrays['modes'][:, body]
        usable, full = 0, 0
        for anchor in range(4, len(times)-1, stride):
            mode = modes[anchor]
            if mode == 1 or not valid[anchor]:
                continue
            begin = anchor
            while (begin > max(0, anchor-19) and valid[begin-1] and modes[begin-1] == mode
                   and times[begin]-times[begin-1] <= .075):
                begin -= 1
            if anchor-begin+1 < 5:
                continue
            end = min(len(times), anchor+steps+1)
            boundary = np.maximum.accumulate(~arrays['target_valid'][anchor:end, body] |
                                              (modes[anchor:end] != mode))
            mask = ~boundary & np.isclose(times[anchor:end], times[anchor]+np.arange(end-anchor)*.05, atol=1e-6)
            mask[0] = False
            usable += int(mask.any())
            full += int(len(mask) == steps+1 and mask[-1])
        per_body[name] = dict(usable_context_windows=usable, full_horizon_windows=full)
    return dict(usable_context_windows=sum(row['usable_context_windows'] for row in per_body.values()),
        full_horizon_windows=sum(row['full_horizon_windows'] for row in per_body.values()),
        by_body=per_body, steps=steps, history_minimum_frames=5,
        outcome_used_for_eligibility=False, genuine_short_terminal_is_infrastructure_failure=False,
        automatic_resampling=False)


def sensor_digest(observation):
    digest = hashlib.sha256()
    for name in sorted(observation['observation/depths']):
        for channel in ('observation/images', 'observation/depths', 'observation/instance_masks'):
            if channel not in observation:
                continue
            values = np.ascontiguousarray(observation[channel][name])
            digest.update(channel.encode()); digest.update(name.encode()); digest.update(values.tobytes())
    return digest.hexdigest()


class TRAINWaypointOpeningAdapter:
    """TRAIN-only scheduling of a waypoint teacher's requested opening.

    This receives only the current teacher command and current measured TCP.
    It neither queries attachment/object GT nor executes native paths/steps.
    Native grasp remains once per actual open->close command transition.
    Each pending close is checked against the newly returned waypoint; an old
    target is never latched across teacher stage/body/target changes.
    """
    def __init__(self,configuration):
        self.configuration=validate_teacher_timing(configuration)
        self.applied_opening=None

    def apply(self,label,robot):
        from our_mani.contracts import Pose
        desired=float(label['opening'])
        if not np.isfinite(desired) or not 0 <= desired <= 1:
            raise CollectionError('invalid raw teacher desired opening')
        goal=Pose.from_xyzw(label['tcp_xyzw'])
        position_error=float(np.linalg.norm(robot.pose.position-goal.position))
        rotation_error=float(Rotation.from_matrix(goal.rotation @ robot.pose.rotation.T).magnitude())
        if self.applied_opening is None:
            self.applied_opening=float(robot.gripper_open)
        before=self.applied_opening
        ready=(position_error <= self.configuration['close_position_tolerance_m']
            and rotation_error <= self.configuration['close_orientation_tolerance_rad'])
        if desired >= .5:
            # Author release branches already test proximity to the *current*
            # release waypoint. Their returned pose may predict a future moving
            # target (or retreat), so a second pose gate would delay release.
            self.applied_opening=1.
            reason=('author_open_release_stage' if label['phase'] in ('release','deposit','hang','place')
                    else 'author_open_approach_or_reposition')
        elif before < .5:
            reason='maintain_applied_close_without_retry'
        elif label['phase'] != 'acquire':
            reason='close_refused_outside_acquire'
        elif ready:
            self.applied_opening=0.
            reason='acquire_waypoint_reached_apply_close'
        else:
            reason='wait_for_current_acquire_waypoint'
        return self.applied_opening,dict(enabled=True,desired_opening=desired,
            applied_opening=self.applied_opening,position_error_m=position_error,
            orientation_error_rad=rotation_error,close_waypoint_ready=bool(ready),
            opening_transition=before != self.applied_opening,
            close_delayed=desired < .5 and self.applied_opening >= .5,
            reason=reason,configuration=self.configuration,
            native_grasp_retry_or_GT_association_used=False)


def phase_completion_event(origin,current,native,body_ids,*,evidence,to_phase):
    """Private TRAIN completion label, independent of old-world command_done."""
    from our_mani.contracts import Pose
    reference=origin['reference_body'];pose=None;kind=origin['reference_kind']
    if reference in body_ids:
        index=body_ids.index(reference)
        if native['valid'][index]:
            pose=np.r_[native['positions'][index],Rotation.from_matrix(native['rotations'][index]).as_quat()].tolist()
            if kind=='translation_only':pose[3:]=[0.,0.,0.,1.]
    if kind=='phase_origin':pose=origin['issued_measured_tcp_xyzw']
    elif kind=='world':pose=Pose(np.zeros(3)).xyzw().tolist()
    return dict(version='r10_TRAIN_phase_completion_event_v1',phase=origin['raw_phase'],
        from_phase=origin['raw_phase'],to_phase=to_phase,evidence=evidence,
        origin_command_id=origin['command_id'],eligible_command_ids=[origin['command_id']],
        origin_executed_tick_count=int(origin['executed_tick_count']),
        origin_command_completed='command_completed_at' in origin,
        attempt_start_time=origin['generated_at'],attempt_end_time=float(current['times']),
        origin_reference_body=reference,origin_reference_role=origin['reference_role'],
        origin_reference_kind=origin['reference_kind'],origin_anchor_id=origin['anchor_id'],
        origin_manipulated_body=origin['manipulated_body'],origin_manipulated_role=origin['manipulated_role'],
        origin_reference_motion_mode=origin['issued_reference_motion_mode'],
        origin_manipulated_motion_mode=origin['issued_manipulated_motion_mode'],
        current_measured_tcp_xyzw=np.asarray(current['measured_tcp_xyzw']).tolist(),
        current_reference_native_pose_xyzw=pose,
        current_reference_native_pose_valid=pose is not None,
        origin_measured_tcp_xyzw=origin['issued_measured_tcp_xyzw'],
        committed_world_waypoint_xyzw=origin['raw_waypoint_xyzw'],
        label_namespace='TRAIN_metric_only',
        completion_is_current_moving_body_label=kind in ('observed_se3','translation_only') and pose is not None,
        committed_world_command_residual_is_current_body_residual=False,
        forward_transition_is_not_an_independent_successful_episode=True)


def collect_episode(env, tracker, *, split, maximum_ticks, mode_observer, targets,
                    teacher=None, policy=None, opening_adapter=None, behavior=None,
                    follower=None, reference_executor=None, gripper_measurement=None,
                    phase_order=(),final_required_phase=None,phase_completion_events=None):
    """Dependency-injected counted collector; mocks need no native modules.

    targets and teacher results are stored after tracker observation and are
    never passed to tracker/policy/mode association. Teacher controls TRAIN
    actions only. VAL policy receives causal observations and proprioception.
    """
    hold_behavior=behavior in (TRAIN_HOLD_BEHAVIOR,VAL_HOLD_BEHAVIOR)
    has_teacher=teacher is not None or follower is not None
    if type(maximum_ticks) is not int or not 1<=maximum_ticks<=200:
        raise CollectionError('explicit integer counted collection budget 1..200 required')
    if (split not in ('train','validation') or (split=='validation' and has_teacher)
            or (split=='train' and not has_teacher and not hold_behavior)):
        raise CollectionError('TRAIN teacher or explicit hold behavior required; teacher forbidden on VAL')
    if (behavior==TRAIN_HOLD_BEHAVIOR and split!='train') or (behavior==VAL_HOLD_BEHAVIOR and split!='validation'):
        raise CollectionError('hold behavior split differs')
    if hold_behavior and (has_teacher or policy is not None or opening_adapter is not None):
        raise CollectionError('fixed hold may not silently use teacher/policy actions')
    if follower is not None and (teacher is not None or opening_adapter is not None or reference_executor is None):
        raise CollectionError('committed TRAIN follower needs one direct executor and no legacy adapter')
    if split == 'train' and policy is not None:
        raise CollectionError('TRAIN teacher collection cannot silently mix a VAL policy')
    if opening_adapter is not None and (split != 'train' or teacher is None):
        raise CollectionError('waypoint opening adapter is TRAIN teacher only')
    body_ids = tuple(tracker.templates)
    rows, diagnostics, previous = [], [], []
    phase_completion_events=[] if phase_completion_events is None else phase_completion_events
    command_segments = PublicCommandSegments(body_ids)
    from our_mani.contracts import Pose
    for _ in range(maximum_ticks+1):
        observation, robot = env.observe()
        before = (env.physics_time(), float(env.task.t), env.ledger.ticks)
        modes, invalid = mode_observer.observe(previous, robot)
        states, metadata = tracker.observe(observation, modes=modes)
        current = body_row(states, metadata, body_ids, modes, invalid)
        current.update(times=float(observation['timestamp']), measured_tcp_xyzw=robot.pose.xyzw(),
            joint_positions=np.asarray(robot.joints), robot_gripper_open=bool(robot.gripper_open),
            robot_contact=bool(robot.contact), sensor_sha256=sensor_digest(observation))
        fingers=(np.asarray(gripper_measurement(),float) if gripper_measurement is not None
                 else np.full(2,float(robot.gripper_open)))
        current.update(teacher_command_id=-1,teacher_command_is_new=False,
            teacher_command_generated_at=0.,teacher_command_age_seconds=0.,
            teacher_execution_stage='no_teacher',teacher_path_done=False,
            teacher_measured_waypoint_arrival=False,teacher_gripper_actuation_done=False,
            teacher_command_completed=False,teacher_execution_valid=False,teacher_joint_limit_recovery=False,
            applied_joint_reference=np.asarray(robot.joints).copy(),gripper_open_amount=fingers.copy(),
            gripper_open_amount_after=fingers.copy(),actuator_feedback_timestamp=float(robot.timestamp))
        # Native targets have a different namespace from all policy history.
        native = targets(body_ids)
        for key in ('positions', 'rotations', 'linear_velocities', 'angular_velocities', 'valid', 'rotation_observable'):
            current['target_'+key] = native[key]
        current['mode_valid'] = np.asarray([body not in invalid for body in body_ids], dtype=bool)
        current['native_pose_valid'] = np.array(native['valid'], dtype=bool, copy=True)
        current['target_valid'] = current['target_valid'] & current['mode_valid']
        terminal = bool(env.native_outcome.terminate)
        can_act = not terminal and env.ledger.ticks < maximum_ticks
        action = None;follower_result=None
        opening_diagnostic=dict(enabled=opening_adapter is not None,desired_opening=None,applied_opening=None,
            position_error_m=0.,orientation_error_rad=0.,close_waypoint_ready=False,opening_transition=False,
            close_delayed=False,reason='no_teacher_action',native_grasp_retry_or_GT_association_used=False)
        command_kind = 'hold' if can_act else 'no_action_terminal' if terminal else 'budget_stop'
        if can_act and has_teacher:
            if follower is not None:
                follower_result=follower.command(robot,env.robot_model.lower,env.robot_model.upper)
                label=follower_result['label']
                current.update(teacher_command_id=follower_result['command_id'],
                    teacher_command_is_new=follower_result['command_is_new'],
                    teacher_command_generated_at=follower_result['command_generated_at'],
                    teacher_command_age_seconds=follower_result['command_age_seconds'],
                    teacher_execution_stage=follower_result['execution_state'],
                    teacher_path_done=follower_result['path_done'],
                    teacher_measured_waypoint_arrival=follower_result['measured_waypoint_arrival'],
                    teacher_execution_valid=follower_result['teacher_execution_valid'],
                    teacher_joint_limit_recovery=follower_result['joint_limit_recovery_action'])
                if follower_result['command_is_new']:
                    binding=follower.commands[-1]
                    reference=label['reference_body']
                    binding['issued_reference_motion_mode']='unknown' if reference in invalid else modes.get(reference,'not_a_body')
                    binding['issued_manipulated_motion_mode']=('unknown' if label['manipulated_body'] in invalid
                        else modes.get(label['manipulated_body'],'unknown'))
                    if reference in body_ids:
                        index=body_ids.index(reference)
                        binding['issued_reference_native_pose_xyzw']=np.r_[native['positions'][index],
                            Rotation.from_matrix(native['rotations'][index]).as_quat()].tolist()
                    if len(follower.commands)>1:
                        prior=follower.commands[-2]
                        binding['raw_teacher_stage_transition']=prior['raw_stage']!=binding['raw_stage']
                        binding['previous_command_id']=prior['command_id']
                        binding['raw_stage_transition_is_phase_success']=False
                        old,new=prior['raw_phase'],binding['raw_phase']
                        if ('command_completed_at' in prior and old in phase_order and new in phase_order
                                and phase_order.index(new)==phase_order.index(old)+1):
                            phase_completion_events.append(phase_completion_event(prior,current,native,body_ids,
                                evidence='completed_command_then_ordered_teacher_phase_transition',to_phase=new))
            else:label = teacher()
            if label is None:
                # A public q-bound recovery can precede the first raw decision.
                label=dict(tcp_xyzw=robot.pose.xyzw(),opening=follower_result['opening'],stage='',phase='',
                    manipulated_body='',manipulated_role='',reference_body='',reference_role='',reference_kind='',anchor_id='')
            desired_opening=float(label['opening'])
            applied_opening=follower_result['opening'] if follower is not None else desired_opening
            if follower is not None:
                goal=Pose.from_xyzw(label['tcp_xyzw'])
                opening_diagnostic.update(enabled=True,desired_opening=desired_opening,
                    applied_opening=applied_opening,position_error_m=float(np.linalg.norm(goal.position-robot.pose.position)),
                    orientation_error_rad=float(Rotation.from_matrix(goal.rotation@robot.pose.rotation.T).magnitude()),
                    close_waypoint_ready=follower_result['measured_waypoint_arrival'],
                    close_delayed=desired_opening<.5 and applied_opening>=.5,
                    reason='committed_RML_'+follower_result['execution_state'])
            elif opening_adapter is not None:
                applied_opening,opening_diagnostic=opening_adapter.apply(label,robot)
            else:
                opening_diagnostic.update(desired_opening=desired_opening,applied_opening=applied_opening,
                    reason='explicit_legacy_CPU_test_passthrough')
            current.update(teacher_tcp_xyzw=label['tcp_xyzw'], teacher_opening=applied_opening,
                teacher_desired_opening=desired_opening,teacher_applied_opening=applied_opening,
                teacher_stage=label['stage'], teacher_phase=label['phase'],
                teacher_action_valid=bool(label['stage']) and (follower is None or follower_result['teacher_execution_valid']),
                teacher_manipulated_body=label['manipulated_body'],
                teacher_manipulated_role=label.get('manipulated_role', label['manipulated_body']),
                teacher_reference_body=label['reference_body'],
                teacher_reference_role=label['reference_role'], teacher_reference_kind=label['reference_kind'],
                teacher_anchor_id=label['anchor_id'])
            if follower is None:action = Pose.from_xyzw(label['tcp_xyzw']), applied_opening
            # This records the command the TRAIN teacher/controller actually
            # issues. It does not expose the native task's hidden phase to a
            # learned policy or to tracker/history features.
            command_kind = ('hold' if follower_result is not None and
                (follower_result['failure'] is not None or follower_result['joint_limit_recovery_action']
                    or follower_result['execution_state']=='release_settle')
                else 'press' if label['phase']=='press' else 'move')
        else:
            last_applied=(follower.opening if follower is not None else opening_adapter.applied_opening if opening_adapter is not None
                          and opening_adapter.applied_opening is not None else float(robot.gripper_open))
            current.update(teacher_tcp_xyzw=np.r_[np.zeros(3), [0., 0., 0., 1.]], teacher_opening=last_applied,
                teacher_desired_opening=1.,teacher_applied_opening=last_applied,
                teacher_stage='', teacher_phase='', teacher_action_valid=False, teacher_manipulated_body='',
                teacher_manipulated_role='',
                teacher_reference_body='', teacher_reference_role='', teacher_reference_kind='', teacher_anchor_id='')
        if can_act and policy is not None:
            # No observation['state'] extension, teacher label, target arrays,
            # native receipt, env/task handle or future trajectory is passed.
            issued = policy.act(states=states, robot=robot, diagnostics=metadata,
                                remaining_seconds=(maximum_ticks-env.ledger.ticks)*env.ledger.dt)
            if not isinstance(issued, tuple) or len(issued) not in (2, 3) or not isinstance(issued[0], Pose):
                raise CollectionError('frozen causal policy must return (Pose, opening[, public_command])')
            action = issued[:2]
            public = issued[2] if len(issued) == 3 else None
            if public is not None:
                if not isinstance(public, dict) or set(public) != {'kind'}:
                    raise CollectionError('public command metadata must contain only its issued kind')
                command_kind = public['kind']
                if command_kind not in ('move', 'hold', 'press'):
                    raise CollectionError('an issued policy action needs public move/hold/press kind')
            else:
                command_kind = 'unspecified'
            if 'plate' in body_ids and command_kind == 'unspecified':
                raise CollectionError('plate VAL policy must explicitly log public move/hold/press commands')
        segment_valid = command_segments.issue(command_kind)
        current.update(teacher_opening_gate_enabled=opening_adapter is not None or follower is not None,
            teacher_opening_gate_delayed=bool(opening_diagnostic['close_delayed']),
            teacher_opening_gate_reason=opening_diagnostic['reason'],
            teacher_opening_gate_position_error_m=opening_diagnostic['position_error_m'],
            teacher_opening_gate_orientation_error_rad=opening_diagnostic['orientation_error_rad'])
        current['public_command_kind'] = command_kind
        current['dynamics_segment_valid'] = segment_valid
        current['valid'] &= segment_valid
        current['rotation_observable'] &= segment_valid
        current['target_valid'] &= segment_valid
        if before != (env.physics_time(), float(env.task.t), env.ledger.ticks):
            raise CollectionError('observation/tracker/teacher/target read advanced the counted clock')
        rows.append(current)
        # Typed current-observation capabilities never survive serialization.
        # The separately stored JSON receipt audits evidence without granting
        # a replayed diagnostic a runtime geometry/reference permission.
        diagnostic_bodies = {role: {key: value for key, value in body.items()
            if key != 'rotation_equivalence_evidence'} for role, body in metadata['bodies'].items()}
        diagnostics.append(dict(timestamp=current['times'], bodies=diagnostic_bodies,
            rigid_groups=metadata.get('rigid_groups', {}), mode_identity_ambiguous=sorted(invalid),
            public_command_kind=command_kind, non_exogenous_command_bodies=sorted(command_segments.non_exogenous),
            TRAIN_teacher_opening_timing=opening_diagnostic))
        if not can_act:
            if (follower is not None and follower.commands and final_required_phase is not None
                    and bool(getattr(env.native_outcome,'success',False))):
                origin=follower.commands[-1]
                if (origin['raw_phase']==final_required_phase and origin['executed_tick_count']>0
                        and len(rows)>=2 and rows[-2]['teacher_execution_valid']
                        and rows[-2]['teacher_command_id']==origin['command_id']):
                    phase_completion_events.append(phase_completion_event(origin,current,native,body_ids,
                        evidence='TRAIN_native_success_during_executed_final_required_phase',to_phase='native_success'))
            break
        count = env.ledger.ticks
        if follower_result is not None:
            feedback=reference_executor(follower_result['joint_reference'],follower_result['opening'],
                phase='TRAIN_teacher_'+follower_result['execution_state'])
            feedback['joint_limit_recovery_action']=follower_result['joint_limit_recovery_action']
            current.update(applied_joint_reference=np.asarray(feedback['applied_joint_reference']),
                gripper_open_amount=np.asarray(feedback['gripper_open_amount_before']),
                gripper_open_amount_after=np.asarray(feedback['gripper_open_amount_after']),
                actuator_feedback_timestamp=float(feedback['timestamp']),
                teacher_gripper_actuation_done=bool(feedback['gripper_actuation_done']),
                teacher_command_completed=follower.after_tick(feedback))
            if current['teacher_command_completed'] and follower.raw['reference_body'] in body_ids:
                reference=follower.raw['reference_body'];index=body_ids.index(reference)
                completion_native=targets(body_ids)
                follower.commands[-1]['command_terminal_reference_native_pose_xyzw']=np.r_[
                    completion_native['positions'][index],Rotation.from_matrix(completion_native['rotations'][index]).as_quat()].tolist()
                follower.commands[-1]['command_terminal_reference_metric_time']=float(feedback['timestamp'])
                follower.commands[-1]['reference_pose_is_private_TRAIN_label']=True
        elif action is None:
            env.hold_tick()
        else:
            env.advance(action[0], opening=float(action[1]))
        if env.ledger.ticks != count+1:
            raise CollectionError('actuator must execute exactly one common physical tick')
        previous = states
    if follower is not None:
        follower.finish('native_terminal' if env.native_outcome.terminate else 'fixed_tick_budget',rows[-1]['times'])
    if not rows or np.any(np.diff([row['times'] for row in rows]) <= 0):
        raise CollectionError('invalid causal observation clock')
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    arrays['body_ids'] = np.asarray(body_ids)
    return arrays, diagnostics


def read_native_targets(env, body_ids):
    """Metric/future-label only; never return this object to a policy."""
    from our_mani.multitask.native_tasks import resolve_native_roles
    from our_mani.contracts import Pose
    before = env.physics_time(), float(env.task.t)
    roles = resolve_native_roles(env.task, env.spec, scene_object_resolver=env.resolve_scene_object)
    positions, rotations, linear, angular = [], [], [], []
    for body in body_ids:
        pose = Pose.from_xyzw(roles[body].get_pose())
        v, w = roles[body].get_velocity()
        positions.append(pose.position); rotations.append(pose.rotation); linear.append(v); angular.append(w)
    if before != (env.physics_time(), float(env.task.t)):
        raise CollectionError('native metric-label read advanced the clock')
    return dict(positions=np.asarray(positions), rotations=np.asarray(rotations),
        linear_velocities=np.asarray(linear), angular_velocities=np.asarray(angular),
        valid=np.ones(len(body_ids), bool), rotation_observable=np.ones(len(body_ids), bool))


def teacher_reference_labels(env):
    """Capture teacher-selected identities BEFORE its mutable waypoint call.

    Privileged TRAIN supervision only. No goal sensor is queried and these
    choices are not reused as controller attachment identities.
    """
    family, task = env.spec.family, env.task
    if family == 'place_cups_on_rotating_frame':
        cup = task._cups[task._cups_placed]
        tip = np.asarray(task.robot.arm.get_tip().get_position())
        spoke = min(task._spokes, key=lambda value: np.linalg.norm(np.asarray(value.get_position())-tip))
        return dict(manipulated_body='cup'+str(task._cups.index(cup)), selected_reference='spoke'+str(task._spokes.index(spoke)))
    if family == 'remove_cups_from_rotating_frame':
        tip = np.asarray(task.robot.arm.get_tip().get_position())
        cup = min(task.cups, key=lambda value: np.linalg.norm(np.asarray(value.get_position())-tip))
        spoke = min(task.spokes, key=lambda value: np.linalg.norm(np.asarray(value.get_position())-np.asarray(cup.get_position())))
        return dict(manipulated_body='cup'+str(task.cups.index(cup)), selected_reference='spoke'+str(task.spokes.index(spoke)))
    if family == 'insert_onto_rotating_peg':
        # Variation is not the pillar ID: native TRAIN initialization samples
        # target_spoke independently. This driver field is privileged teacher
        # identity supervision only, never a policy feature or goal pose.
        states = task.var2target_state_list
        variation = env.collection_variation
        selected = states[variation] if variation in states else states[str(variation)]
        name = selected['target_spoke']
        if name not in ('pillar0', 'pillar1', 'pillar2'):
            raise CollectionError('TRAIN peg teacher target identity is unknown')
        return dict(manipulated_body='ring', selected_reference='peg'+name[-1])
    return dict(manipulated_body=env.spec.primary_manipulated_roles[0], selected_reference='')


def train_color_annotations(env, observation, instance_bindings, source_group):
    """Current TRAIN RGB mask evidence plus separately privileged color labels.

    A selected peg index/color mapping is episode-local teacher supervision,
    never a lookup table for a future VAL/TEST target. Vocabulary fitting must
    aggregate TRAIN examples; runtime resolves color from current RGB-D.
    """
    from our_mani.multitask.perception import decode_instance_mask
    if env.development_split != 'train' or not source_group.startswith('train:'):
        raise CollectionError('native color annotation is TRAIN-only')
    labels = {}
    module = importlib.import_module('rlbench.tasks.'+env.spec.task_name)
    if env.spec.family == 'insert_onto_rotating_peg':
        states, variation = env.task.var2target_state_list, env.collection_variation
        selected = states[variation] if variation in states else states[str(variation)]
        chosen = str(selected['target_spoke'])
        labels['peg'+chosen[-1]] = (selected['color_name'], selected['color_rgb'])
        other = [name for name in ('pillar0', 'pillar1', 'pillar2') if name != chosen]
        for name, color_index in zip(other, selected['color_choices']):
            labels['peg'+name[-1]] = module.colors[int(color_index)]
    elif env.spec.family == 'push_moving_button':
        labels['button'] = module.colors[0]
    evidence = {}
    for role, components in instance_bindings.items():
        ids = [value for values in components.values() for value in values]
        pixels, views = [], []
        for camera, image in observation['observation/images'].items():
            mask = decode_instance_mask(observation['observation/instance_masks'][camera])
            selection = np.isin(mask, ids)
            values = np.asarray(image)[selection]
            if len(values):
                pixels.append(values.astype(float)); views.append(camera)
        count = sum(len(values) for values in pixels)
        pooled = np.concatenate(pixels) if pixels else np.zeros((0, 3))
        sample_index = np.linspace(0, max(0, count-1), min(512, count), dtype=int)
        record = dict(observed_at=float(observation['timestamp']), visible=count >= 12,
            pixel_count=count, cameras=views, provenance='current_rgbd',
            component_names=list(components), rgb_pixels_255=pooled[sample_index].tolist(),
            pixel_sampling='deterministic evenly-spaced current masked RGB pixels, at most 512 per role',
            source_split='train', source_group=source_group,
            color_rgb_median_255=np.median(np.concatenate(pixels), axis=0).tolist() if pixels else None,
            color_name=None, teacher_color_annotation=None)
        if role in labels:
            name, rgb = labels[role]
            record['teacher_color_annotation'] = dict(color_name=str(name), color_rgb=np.asarray(rgb).tolist(),
                source_split='train', source_group=source_group,
                evidence='TRAIN_teacher_color_annotation_and_current_RGBD',
                label_is_runtime_policy_observation=False)
        evidence[role] = record
    return dict(version='r10_TRAIN_role_color_supervision_v1', split='train', task=env.spec.task_name,
        source_group=source_group, role_color_labels=evidence,
        geometry_color_role_assignment_is_permanent=False, shared_with_baseline=True,
        inference_requirement='estimate role colors from current RGB-D using a frozen TRAIN color vocabulary/calibrator; never reuse this episode selected-peg identity')


def reference_for_phase(spec, phase, selected, protocol=None):
    from our_mani.multitask.reference_overrides import structural_overrides,ReferenceOverrideError
    family, manipulated = spec.family, selected['manipulated_body']
    manipulated_role = 'selected_cup' if family == 'remove_cups_from_rotating_frame' else manipulated
    kind, reference, role = 'observed_se3', manipulated, manipulated
    if family == 'remove_cups_from_rotating_frame':
        role = 'selected_cup'
    if phase == 'lift':
        kind, reference, role = 'phase_origin', '', 'phase_origin'
    elif family == 'beat_the_rotating_buzz' and phase in ('follow', 'release'):
        reference = role = 'frame'
    elif family == 'insert_onto_rotating_peg' and phase in ('align', 'insert', 'release'):
        reference, role = selected['selected_reference'], 'selected_peg'
    elif family == 'moving_basketball_in_hoop':
        if phase in ('approach', 'acquire'):
            kind = 'translation_only'
        elif phase == 'deposit':
            reference = role = 'hoop'
    elif family == 'place_cups_on_rotating_frame' and phase in ('transport', 'align', 'hang'):
        reference, role = selected['selected_reference'], 'selected_spoke'
    elif family == 'put_rubbish_in_moving_bin' and phase == 'deposit':
        reference = role = 'bin'
    elif family == 'remove_cups_from_rotating_frame':
        if phase == 'detach':
            reference, role = selected['selected_reference'], 'selected_spoke'
        elif phase in ('transport', 'place'):
            kind, reference, role = 'world', '', 'world'
    override=structural_overrides(protocol).get(getattr(spec,'task_name',None),{}).get(phase)
    if override is not None:
        if override['reference_role']!=role:
            raise ReferenceOverrideError('reference-kind override cannot mutate the original semantic role')
        kind=override['reference_kind']
    return dict(reference_body=reference, reference_role=role, reference_kind=kind,
                manipulated_body=manipulated, manipulated_role=manipulated_role,
                anchor_id=f'{family}:{phase}:{role}:{manipulated_role}')


def native_teacher(env):
    if env.development_split != 'train':
        raise CollectionError('author teacher is forbidden outside TRAIN')
    from our_mani.multitask.reference_overrides import validate_overrides
    protocol=getattr(env,'protocol',{})
    validate_overrides(protocol,root=ROOT)
    module = importlib.import_module('rlbench.tasks.'+env.spec.task_name)
    function = module.get_expert_info
    stages = {stage: phase.name for phase in env.spec.phase_skills for stage in phase.TRAIN_expert_stages}

    def act():
        if env._native_terminal_latched or getattr(env.task, '_bool_expert', None) is not False:
            raise CollectionError('teacher cannot run after terminal or with native autonomous expert enabled')
        before = env.physics_time(), float(env.task.t), env.ledger.ticks
        selected = teacher_reference_labels(env)
        info = function(env.task, bool_return_path=False)
        if 'path' in info or before != (env.physics_time(), float(env.task.t), env.ledger.ticks):
            raise CollectionError('teacher requested a hidden native path or advanced physics')
        trajectory = info['trajectory']
        if hasattr(trajectory, 'detach'):
            trajectory = trajectory.detach().cpu().numpy()
        trajectory = np.asarray(trajectory, dtype=float)
        if trajectory.shape != (1, 1, 8) or not np.isfinite(trajectory).all():
            raise CollectionError('teacher must return one TCP/gripper action')
        stage = str(info['stage'])
        if stage not in stages:
            raise CollectionError('unmapped TRAIN teacher stage: '+stage)
        # Same teacher state bookkeeping as author expert_plan(), without
        # its path execution, gripper loops or uncounted physics stepping.
        env.task.stage = stage
        if env.spec.family == 'place_cups_on_rotating_frame':
            env.task._cups_placed = info['task_cup_placed']
        label = dict(tcp_xyzw=trajectory[0, 0, :7].copy(), opening=float(trajectory[0, 0, 7]),
                     stage=stage, phase=stages[stage], **reference_for_phase(env.spec, stages[stage], selected,protocol=protocol))
        if not 0 <= label['opening'] <= 1:
            raise CollectionError('invalid teacher opening')
        return label
    return act


def fit_rigid_layout(arrays, *, task, source_group, minimum_samples=5,
                     position_tolerance=.002, orientation_tolerance=.02):
    """TRAIN native-label calibration, never fitted on VAL or TEST.

    A relation must remain constant through the recorded motion. Unvalidated
    or absent relations are rejected individually and never silently forced.
    This output can be frozen for a later collector/policy; it is not used to
    retroactively change the current episode's causal tracker estimates.
    """
    if not source_group.startswith('train:'):
        raise CollectionError('rigid-layout fitting accepts TRAIN source groups only')
    ids = list(map(str, arrays['body_ids']))
    pairs = [('frame', child) for child in ids if child.startswith(('peg', 'spoke'))]
    if 'button' in ids and 'wrap' in ids:
        pairs.append(('button', 'wrap'))
    accepted, rejected = [], []
    for parent, child in pairs:
        p, c = ids.index(parent), ids.index(child)
        valid = arrays['target_valid'][:, p] & arrays['target_valid'][:, c]
        if valid.sum() < minimum_samples:
            rejected.append(dict(parent=parent, child=child, reason='insufficient TRAIN samples'))
            continue
        pr, cr = arrays['target_rotations'][valid, p], arrays['target_rotations'][valid, c]
        relative_p = np.einsum('tji,tj->ti', pr, arrays['target_positions'][valid, c]-arrays['target_positions'][valid, p])
        relative_r = np.einsum('tji,tjk->tik', pr, cr)
        center, rotation = np.median(relative_p, axis=0), Rotation.from_matrix(relative_r).mean().as_matrix()
        position_error = float(np.linalg.norm(relative_p-center, axis=1).max())
        angle_error = float(np.linalg.norm(Rotation.from_matrix(relative_r @ rotation.T).as_rotvec(), axis=1).max())
        record = dict(parent=parent, child=child, samples=int(valid.sum()), position_error_max_m=position_error,
                      orientation_error_max_rad=angle_error)
        if position_error > position_tolerance or angle_error > orientation_tolerance:
            rejected.append(dict(record, reason='TRAIN relative pose changed; not a permanent rigid relation'))
        else:
            accepted.append(dict(record, child_to_parent_position=center.tolist(), child_to_parent_rotation=rotation.tolist()))
    payload = dict(version='r10_TRAIN_observed_rigid_layout_v1', task=task, split='train',
        source_group=source_group, source='TRAIN native SE3 future-label arrays only',
        thresholds=dict(minimum_samples=minimum_samples, position_tolerance_m=position_tolerance,
                        orientation_tolerance_rad=orientation_tolerance),
        arrays_sha256=canonical_sha({key: np.asarray(arrays[key]).tolist() for key in
            ('body_ids', 'times', 'target_positions', 'target_rotations', 'target_valid')}),
        accepted=accepted, rejected=rejected, shared_with_baseline=True,
        changes_current_episode_history=False)
    payload['evidence_sha256'] = canonical_sha(payload)
    return payload


def load_rigid_constraints(path, task):
    from our_mani.multitask.perception import PriorProvenance, RigidChildConstraint
    from our_mani.contracts import Pose
    asset = json.loads(Path(path).read_text())
    if (asset.get('version') != 'r10_TRAIN_observed_rigid_layout_v1' or asset.get('split') != 'train'
            or asset.get('task') != task or asset.get('shared_with_baseline') is not True
            or asset.get('evidence_sha256') != canonical_sha({k:v for k,v in asset.items() if k != 'evidence_sha256'})):
        raise CollectionError('invalid/future/non-TRAIN rigid-layout artifact')
    provenance = PriorProvenance('train_calibrated_rigid_layout', str(Path(path).resolve()),
        'child frame expressed in parent native TRAIN CAD frame', True, 'train',
        (asset['source_group'],), asset['evidence_sha256'])
    return tuple(RigidChildConstraint(row['parent'], row['child'],
        Pose(row['child_to_parent_position'], row['child_to_parent_rotation']), provenance) for row in asset['accepted'])


def case_directory(output, case):
    return destination(Path(output)/'cases'/case['split']/case['task']/case['case_id'])


def validate_case_result(path, plan, case):
    record = json.loads(Path(path).read_text())
    if (record.get('case_id') != case['case_id'] or record.get('plan_sha256') != plan['plan_sha256']
            or record.get('status') != 'completed' or record.get('case_isolation') != 'fresh_simulator_process_per_episode'):
        raise CollectionError('case receipt identity/status differs')
    if plan['version']==BEHAVIOR_PLAN_VERSION:
        entry=record.get('entry',{})
        for field in ('behavior','behavior_replica','maximum_collection_ticks','teacher_used',
                      'context_primary_view','interaction_eligible'):
            if entry.get(field)!=case[field]:raise CollectionError('case behavior receipt differs: '+field)
        if (record.get('teacher_actions_used') is not case['teacher_used']
                or record.get('teacher_paths_requested') is not case['teacher_used']
                or record['alignment']['clock']['ticks']>case['maximum_collection_ticks']):
            raise CollectionError('actual behavior teacher/path/budget differs')
    for filename, expected in record['files_sha256'].items():
        if sha(ROOT/filename) != expected:
            raise CollectionError('completed case file changed: '+filename)
    parent = record.get('parent_process_receipt')
    if parent is not None:
        child_path = ROOT/parent['child_case_path']
        if sha(child_path) != parent['child_case_sha256']:
            raise CollectionError('raw native child receipt changed')
        child = json.loads(child_path.read_text())
        if (child.get('case_id') != record['case_id'] or child.get('entry') != record['entry']
                or child.get('process_id') != record.get('process_id')
                or sha(ROOT/parent['launcher']) != parent['launcher_sha256']):
            raise CollectionError('native parent/child identity or launcher differs')
    return record


def collect_native_case(plan_path, case_id, output, guard_report):
    plan = verify_plan(plan_path)
    if plan['protocol'].get('draft') is True or plan['protocol'].get('native_execution_authorized') is False:
        raise CollectionError('draft configuration is planning-only; root must finalize a new source-bound configuration')
    selected = [case for case in plan['cases'] if case['case_id'] == case_id]
    if len(selected) != 1:
        raise CollectionError('case ID must select exactly one planned source/replica')
    case = selected[0]
    behavioral=plan['version']==BEHAVIOR_PLAN_VERSION
    uses_teacher=case.get('teacher_used',case['split']=='train')
    if uses_teacher and plan['protocol'].get('TRAIN_teacher_opening_gate_enabled') is not True:
        raise CollectionError('new native TRAIN collection requires explicit waypoint timing configuration; legacy v3 inputs are historical')
    opening_adapter=(TRAINWaypointOpeningAdapter(plan['protocol']['gripper_timing_config'])
                     if uses_teacher and not behavioral else None)
    out = case_directory(output, case)
    if (out/'case.json').exists():
        return validate_case_result(out/'case.json', plan, case)
    gate = require_r09_release(guard_report)
    cache = ROOT/case['offline_demonstration']['path']
    if sha(cache) != case['offline_demonstration']['sha256']:
        raise CollectionError('original small-demonstration cache changed')
    source_stat = Path(case['source']['path']).stat()
    if dict(size_bytes=source_stat.st_size, mtime_ns=source_stat.st_mtime_ns,
            inode=source_stat.st_ino, device=source_stat.st_dev) != case['offline_demonstration']['source_file_stat']:
        raise CollectionError('original HDF5 stat differs from the audited small-cache source')
    rigid_layout = case.get('rigid_layout')
    # Guard is deliberately before importing any native runtime or torch.
    from our_mani.multitask.simulation import AlignedDevelopmentSimulation
    from our_mani.multitask.training_geometry import (export_train_geometry, declared_instance_bindings,
        body_templates_from_asset, register_collision_asset)
    from our_mani.multitask.perception import MultiBodySE3Tracker, RegistrationConfig
    from our_mani.multitask.factory import load_finite_rotation_symmetries,load_cup_feature_models
    started, env, follower = utc(), None, None
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    try:
        env = AlignedDevelopmentSimulation(case['task'], case['source']['path'], source=case['source'],
            split=case['split'], protocol=plan['protocol'], seed=plan['protocol'].get('seed', 0),
            replica=case['replica'], servo_lead=plan['protocol'].get('local_servo_lead', 1.4))
        env.collection_variation = int(case['source']['variation'])
        env.reset_development(variation=env.collection_variation)
        if case['template_asset']:
            asset = json.loads(Path(case['template_asset']).read_text())
        elif case['split'] == 'train':
            asset = export_train_geometry(env, dict(case['source'], split='train'))
        else:
            raise CollectionError('VAL cannot export/calibrate geometry')
        asset_path = out/'TRAIN_geometry.json'
        write_json(asset_path, asset)
        bindings = declared_instance_bindings(env, asset, shared_permission=True)
        finite_symmetries = load_finite_rotation_symmetries(ROOT, plan['protocol'], case['task'], asset)
        cup_models = load_cup_feature_models(ROOT,plan['protocol'],case['task'],asset)
        if case['split'] == 'train':
            color_observation, _ = env.observe()
            color_path = out/'TRAIN_color_annotations.json'
            write_json(color_path, train_color_annotations(env, color_observation, bindings, case['parent_source_group']))
            files[str(color_path.relative_to(ROOT))] = sha(color_path)
        templates = body_templates_from_asset(asset, bindings,
            orientation_observability=plan['protocol']['body_orientation_observability'][case['task']],
            sphere_radii=plan['protocol'].get('body_sphere_radii', {}).get(case['task']),
            symmetry_axes=plan['protocol'].get('body_symmetry_axes',{}).get(case['task']),
            axis_polarity=plan['protocol'].get('body_axis_polarity',{}).get(case['task']),
            finite_rotation_symmetries=finite_symmetries,cup_feature_models=cup_models)
        constraints = load_rigid_constraints(rigid_layout, case['task']) if rigid_layout else ()
        tracker = MultiBodySE3Tracker(templates, task_spec=env.spec, rigid_children=constraints,
            config=RegistrationConfig(**plan['protocol'].get('registration_config', {})))
        collision_geometry_receipt=register_collision_asset(env.map_service, asset,
            orientation_observability=plan['protocol']['body_orientation_observability'][case['task']],
            axis_swept_assets=plan['protocol'].get('axis_swept_assets', {}).get(case['task']),
            finite_rotation_symmetries=finite_symmetries)
        mode_observer = AttachmentModeObserver(tuple(tracker.templates), env.spec.graspable_roles,
                                               plan['protocol']['attachment_association_distance_m'])
        teacher, policy, reference_executor = None, None, None
        if uses_teacher:
            teacher = native_teacher(env)
            if behavioral:
                from our_mani.multitask.train_teacher import (CommittedWaypointFollower,FollowerConfig,
                    native_path_planner,execute_train_reference)
                follower=CommittedWaypointFollower(teacher,native_path_planner(env),
                    FollowerConfig(**plan['protocol']['gripper_timing_config'],
                        joint_limit_recovery_margin_rad=plan['protocol']['joint_limit_recovery_margin_rad'],
                        release_settle_ticks=plan['protocol']['TRAIN_teacher_release_settle_ticks']),
                    initial_opening=env._requested_opening)
                reference_executor=lambda reference,opening,phase:execute_train_reference(env,reference,opening,phase=phase)
                teacher=None
        elif plan['validation_mode'] == 'frozen_policy':
            frozen = plan['frozen_policy']
            module = importlib.import_module(frozen['module'])
            if Path(module.__file__).resolve() != Path(frozen['source_path']).resolve():
                raise CollectionError('frozen causal policy module resolved to a different source')
            policy = getattr(module, frozen['factory'])(artifact_path=frozen['artifact_path'])
        phase_completion_events=[]
        required_phases=[phase.name for phase in env.spec.phase_skills if not phase.optional_after_native_completion]
        arrays, diagnostics = collect_episode(env, tracker, split=case['split'],
            maximum_ticks=case.get('maximum_collection_ticks',200),
            mode_observer=mode_observer, targets=lambda ids: read_native_targets(env, ids), teacher=teacher, policy=policy,
            opening_adapter=opening_adapter,behavior=case.get('behavior'),follower=follower,
            reference_executor=reference_executor,gripper_measurement=env.gripper.get_open_amount,
            phase_order=tuple(phase.name for phase in env.spec.phase_skills),
            final_required_phase=required_phases[-1],phase_completion_events=phase_completion_events)
        motion_metadata=None
        if plan['protocol'].get('dynamics_target_label_schema') is not None:
            from our_mani.multitask.motion_labels import backward_pose_supervision
            raw_arrays=arrays
            arrays,motion_metadata=backward_pose_supervision(arrays,expected_dt=plan['protocol']['dt'])
            label_path=out/'motion_label_diagnostics.json';write_json(label_path,motion_metadata)
            files[str(label_path.relative_to(ROOT))]=sha(label_path)
            if motion_metadata['native_pose_marked_valid_but_nonfinite_or_non_SO3']:
                np.savez_compressed(out/'invalid_GT_raw_body_tracks.npz',**raw_arrays)
                raise CollectionError('invalid declared native GT pose retained as infrastructure evidence; cannot fit')
        if follower is not None:
            command_path=out/'TRAIN_teacher_commands.json'
            write_json(command_path,dict(version='r10_TRAIN_committed_waypoint_commands_v1',
                commands=follower.commands,phase_completion_events=phase_completion_events,
                raw_decisions=len(follower.commands),
                held_frames_are_independent_teacher_decisions=False,
                command_completion_is_native_phase_success=False,
                privileged_paths_used_only_for_TRAIN_supervision=True))
            files[str(command_path.relative_to(ROOT))]=sha(command_path)
        track_path = out/'body_tracks.npz'
        if track_path.exists():
            raise CollectionError('partial track cache exists; retry must use a new attempt directory')
        with track_path.with_suffix('.npz.tmp').open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        track_path.with_suffix('.npz.tmp').replace(track_path)
        with np.load(track_path, allow_pickle=False) as loaded:
            if any(not np.array_equal(loaded[key], value) for key, value in arrays.items()):
                raise CollectionError('body-track NPZ roundtrip failed')
        diagnostics_path = out/'tracker_diagnostics.json'
        write_json(diagnostics_path, dict(version='r10_current_observation_registration_diagnostics_v1', frames=diagnostics))
        if case['split'] == 'train':
            layout_path = out/'TRAIN_rigid_layout.json'
            write_json(layout_path, fit_rigid_layout(arrays, task=case['task'], source_group=case['parent_source_group']))
            files[str(layout_path.relative_to(ROOT))] = sha(layout_path)
        for path in (asset_path, track_path, diagnostics_path):
            files[str(path.relative_to(ROOT))] = sha(path)
        receipt = env.alignment_receipt()
        if (receipt['clock']['ticks'] > 200 or receipt['actual_physics_seconds'] > 10.0001
                or receipt['actual_task_seconds'] > 10.0001):
            raise CollectionError('native collection exceeded the common counted budget')
        eligibility = context_window_eligibility(arrays)
        entry = dict(split=case['split'], task=case['task'], raw_episode_id=case['raw_episode_id'],
            source_id=case['task']+':'+case['raw_episode_id'], augmentation_id=case['raw_episode_id'],
            parent_raw_episode_id=case['parent_raw_episode_id'], parent_source_group=case['parent_source_group'],
            augmentation_replica=case['replica'], cache=str(track_path.relative_to(ROOT)), cache_sha256=sha(track_path),
            observation_source='causal_RGBD_and_proprioception', target_source='TRAIN_VAL_native_supervision',
            source_kind=('TRAIN_native_generated_augmentation' if uses_teacher else 'TRAIN_native_generated_hold')
                if case['split']=='train' else 'VAL_native_generated_non_teacher',
            original_HDF5_motion_replayed=False, source_demonstration=case['offline_demonstration'],
            minimum_history_frames_available=len(arrays['times']) >= 5, frames=len(arrays['times']),
            context_training_eligible=eligibility['usable_context_windows']>0 and case.get('context_primary_view',True),
            diagnostic_context_windows_available=eligibility['usable_context_windows']>0,
            context_window_eligibility=eligibility,
            teacher_opening_schema='teacher_desired_opening is raw author command; teacher_applied_opening is issued native command; teacher_opening aliases applied',
            interaction_opening_array='teacher_applied_opening')
        if behavioral:
            entry.update(behavior=case['behavior'],behavior_replica=case['behavior_replica'],
                maximum_collection_ticks=case['maximum_collection_ticks'],teacher_used=uses_teacher,
                context_primary_view=case['context_primary_view'],interaction_eligible=case['interaction_eligible'],
                interaction_tcp_array='measured_tcp_xyzw' if uses_teacher else None,
                phase_completion_events_receipt=str(command_path.relative_to(ROOT)) if follower is not None else None,
                teacher_command_semantics='one decision per committed waypoint; repeated raw target rows are held commands',
                teacher_command_receipt=str(command_path.relative_to(ROOT)) if follower is not None else None)
        if 'TRAIN_reference_kind_overrides' in plan['protocol']:
            from our_mani.multitask.reference_overrides import SCHEMA as REFERENCE_SCHEMA
            entry.update(TRAIN_reference_kind_override_schema=REFERENCE_SCHEMA,
                TRAIN_reference_kind_overrides_for_task=plan['protocol']['TRAIN_reference_kind_overrides'].get(case['task'],{}),
                TRAIN_reference_kind_override_evidence=plan['protocol']['TRAIN_reference_kind_override_evidence'])
        if motion_metadata is not None:
            entry.update(dynamics_target_label_schema=motion_metadata['schema'],
                motion_label_diagnostics=str(label_path.relative_to(ROOT)),
                target_velocity_source=motion_metadata['source'],
                raw_native_velocity_arrays=motion_metadata['raw_native_velocity_arrays'],
                target_twist_valid_array='target_twist_valid',
                previous_pose_valid_array='target_pose_valid_before_twist_mask',
                position_boundary_samples_conservatively_masked=True)
        record = dict(version='r10_single_case_collection_receipt_v1', status='completed', case_id=case_id,
            plan_sha256=plan['plan_sha256'], split=case['split'], task=case['task'],
            raw_episode_id=case['raw_episode_id'], parent_source_group=case['parent_source_group'],
            process_id=os.getpid(), started_utc=started, finished_utc=utc(),
            case_isolation='fresh_simulator_process_per_episode', gate=gate, entry=entry, files_sha256=files,
            alignment=receipt, native_reset_receipt=env.reset_receipt,
            generated_motion_is_original_source_replay=False, teacher_actions_used=uses_teacher,
            teacher_paths_requested=follower is not None, policy_history_contains_native_GT=False,
            native_GT_namespace='target_* arrays, raw native_reported_* velocities, metric-only receipt and offline TRAIN teacher bindings',
            native_pose_valid_vs_target_valid='native_pose_valid marks measured GT; target_valid additionally masks ambiguous controller attachment identity and public plate press segments',
            public_press_segment_policy='mask plate history/target validity from issued press command until episode end; no native joint/phase/force reads; clearance remains unvalidated',
            semantic_role_mask_permission_shared=True, geometry_prior_shared=True,
            current_instance_bindings=bindings, current_instance_bindings_have_native_pose=False,
            TRAIN_collision_geometry_receipt=collision_geometry_receipt,
            valid_observation_rows=int(arrays['valid'].sum()), teacher_action_rows=int(arrays['teacher_action_valid'].sum()),
            teacher_opening_timing=dict(enabled=uses_teacher,
                configuration=plan['protocol'].get('gripper_timing_config') if uses_teacher else None,
                close_delay_rows=int(arrays['teacher_opening_gate_delayed'].sum()),
                desired_opening_array='teacher_desired_opening',applied_opening_array='teacher_applied_opening',
                compatibility_teacher_opening_equals_applied=bool(np.array_equal(arrays['teacher_opening'],arrays['teacher_applied_opening'])),
                interaction_fit_opening='teacher_applied_opening',close_only_in_acquire=not behavioral,
                release_policy=('apply held raw opening after path completion and measured SE3 arrival, then actual gripper done and declared counted settle'
                    if behavioral else 'honor author current-stage open immediately; no second gate against predicted/retreat waypoint'),
                shared_native_once_close_grasp_semantics_changed=False),
            diagnostic_interpretation='data acquisition; native task success is a metric and does not determine whether this episode enters dynamics fitting')
        # Cleanup is part of infrastructure acceptance and precedes completed.
        env.close(); env = None
        write_json(out/'case.json', record)
        return record
    except Exception:
        if follower is not None:
            try:follower.close()
            except Exception:traceback.print_exc()
        if env is not None:
            try:
                env.close()
            except Exception:
                traceback.print_exc()
        raise


def merge_completed(plan_path, output):
    plan = verify_plan(plan_path)
    entries, cases, missing = [], [], []
    for case in plan['cases']:
        path = case_directory(output, case)/'case.json'
        if not path.is_file():
            missing.append(case['case_id']); continue
        record = validate_case_result(path, plan, case)
        cases.append(dict(case_id=case['case_id'], path=str(path.relative_to(ROOT)), sha256=sha(path)))
        entries.append(record['entry'])
    keys = [(row['split'], row['task'], row['raw_episode_id']) for row in entries]
    if len(keys) != len(set(keys)):
        raise CollectionError('duplicate augmentation identity')
    manifest = dict(version=TRACK_VERSION, created_utc=utc(), plan_sha256=plan['plan_sha256'], entries=entries,
        expected=len(plan['cases']), completed=len(entries), complete=not missing, missing_case_ids=missing,
        cache_episode_count=len(entries), source_episode_count=len({(row['task'],row['parent_raw_episode_id']) for row in entries}),
        case_receipts=cases, original_demonstration_budget=plan['original_demonstration_budget'],
        augmentation_budget=dict(plan['augmentation_budget'], completed_native_cases=len(entries), native_started=bool(entries)),
        source_grouping='parent_source_group is the resampling/split unit across replicas',
        context_eligible_cache_episodes=sum(row.get('context_training_eligible',False) for row in entries),
        context_ineligible_case_ids=[record['case_id'] for record, entry in zip(cases, entries)
                                     if not entry.get('context_training_eligible',False)],
        acquisition_complete_is_not_training_coverage=True,
        short_terminal_cases_are_not_infrastructure_retries=True,
        native_failures_are_observed_data=True, infrastructure_failures_are_not_policy_outcomes=True,
        TEST_used=False, validation_teacher_actions_used=False,
        learning_or_model_fitting_performed=False)
    if plan['version']==BEHAVIOR_PLAN_VERSION:
        manifest.update(context_primary_behavior=plan['context_primary_behavior'],
            completed_by_behavior=dict(Counter(entry['behavior'] for entry in entries)),
            context_primary_cache_episodes=sum(entry['context_primary_view'] for entry in entries),
            teacher_cache_episodes_kept_out_of_primary_context=sum(entry['teacher_used'] for entry in entries),
            no_teacher_fallback_for_missing_hold_parent=True,
            source_kind_budgets_are_separate=True)
    if plan['protocol'].get('dynamics_target_label_schema') is not None:
        manifest['dynamics_target_label_schema']=plan['protocol']['dynamics_target_label_schema']
    write_json(Path(output)/'manifest.json', manifest, overwrite=True)
    return manifest


def interaction_source_from_case(path):
    """Adapt explicit teacher/GT namespaces to the TRAIN-only fitter.

    This does not fit a library, run a simulator, or substitute the original
    HDF5 debug XYZ for a body trajectory. VAL conversion is rejected.
    """
    from our_mani.multitask.interaction import (TRAINBodyTrack, TRAINInteractionSource, TRAINReferenceBinding)
    record = json.loads(Path(path).read_text())
    entry = record['entry']
    if entry.get('behavior') is not None:
        raise CollectionError('v5 completed-skill events require receipt-validated artifact_fitting adapter; legacy raw-waypoint adapter is forbidden')
    if entry['split'] != 'train' or entry.get('source_kind') != 'TRAIN_native_generated_augmentation':
        raise CollectionError('only independently generated TRAIN teacher tracks can use this adapter')
    cache = ROOT/entry['cache']
    if sha(cache) != entry['cache_sha256']:
        raise CollectionError('interaction track cache changed')
    with np.load(cache, allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    if 'teacher_manipulated_role' not in arrays:
        raise CollectionError('TRAIN interaction labels require explicit semantic manipulated roles')
    selected = arrays['teacher_action_valid']
    if selected.sum() < 2:
        raise CollectionError('not enough actual teacher actions for interaction fitting')
    tracks = {}
    for i, body in enumerate(map(str, arrays['body_ids'])):
        if not arrays['target_valid'][:, i].all():
            raise CollectionError('interaction adapter requires complete supervised body-pose labels')
        xyzw = np.column_stack((arrays['target_positions'][:, i],
                                Rotation.from_matrix(arrays['target_rotations'][:, i]).as_quat()))
        tracks[body] = TRAINBodyTrack(body, arrays['times'], xyzw,
            tuple(('external', 'attached', 'released')[int(mode)] for mode in arrays['modes'][:, i]),
            arrays['target_rotation_observable'][:, i])
    refs = tuple(TRAINReferenceBinding(reference_role=str(role), reference_id=str(body) or None,
        reference_kind=str(kind), anchor_id=str(anchor), manipulated_role=str(manipulated))
        for role, body, kind, anchor, manipulated in zip(arrays['teacher_reference_role'][selected],
            arrays['teacher_reference_body'][selected], arrays['teacher_reference_kind'][selected],
            arrays['teacher_anchor_id'][selected], arrays['teacher_manipulated_role'][selected]))
    return TRAINInteractionSource(task_name=entry['task'], source_id=entry['source_id'], split='train',
        source_kind=entry['source_kind'], times=arrays['times'][selected],
        expert_tcp_xyzw=arrays['teacher_tcp_xyzw'][selected], expert_opening=arrays['teacher_opening'][selected],
        measured_tcp_xyzw=arrays['measured_tcp_xyzw'][selected], stages=tuple(map(str, arrays['teacher_stage'][selected])),
        references=refs, body_tracks=tracks, parent_source_group=entry['parent_source_group'],
        augmentation_id=entry['augmentation_id'], raw_episode_id=entry['raw_episode_id'],
        parent_raw_episode_id=entry['parent_raw_episode_id'])


def run_queue(plan_path, output, guard_report, *, workers=8, launcher=None, case_timeout=900, case_ids=None):
    """Explicit opt-in parent queue; subprocess per case and no hidden retry.

    Each unfinished/failed attempt gets a separate directory. Infrastructure
    exceptions never become negative policy labels. Completed cases are SHA
    verified on resume. The queue does not modify a model or select parameters.
    """
    plan = verify_plan(plan_path)
    selected_cases = plan['cases']
    if case_ids is not None:
        requested = tuple(case_ids)
        selected = set(requested)
        known = {case['case_id'] for case in plan['cases']}
        if not requested or len(selected) != len(requested) or selected-known:
            raise CollectionError('queue subset must name unique cases from the unchanged full plan')
        selected_cases = [case for case in plan['cases'] if case['case_id'] in selected]
    if plan['protocol'].get('draft') is True or plan['protocol'].get('native_execution_authorized') is False:
        raise CollectionError('draft configuration cannot launch native workers')
    gate = require_r09_release(guard_report)
    if not 1 <= workers <= 8:
        raise CollectionError('CPU worker limit is 1..8')
    launcher = ROOT/'scripts/run_local_sim.sh' if launcher is None else Path(launcher).resolve()
    if launcher.resolve() != (ROOT/'scripts/run_local_sim.sh').resolve():
        raise CollectionError('use the CPU native launcher with CUDA_VISIBLE_DEVICES empty')
    output = destination(output)
    receipts = []

    def one(case):
        out = case_directory(output, case)
        if (out/'case.json').exists():
            return dict(case_id=case['case_id'], status='resumed_verified',
                        process_id=validate_case_result(out/'case.json', plan, case)['process_id'])
        if COLLECTION_STOP.exists():
            return dict(case_id=case['case_id'], status='cancelled_stop_marker',
                        detail='no new native case launched after the explicit project stop marker',
                        policy_failure_label_assigned=False)
        # Partial failed attempts are preserved. A fresh root prevents stale
        # assets from being accepted as outputs from the next native process.
        attempt = 0
        attempts = output/'infrastructure_attempts'/case['case_id']
        while (attempts/str(attempt)).exists():
            attempt += 1
        attempt_path = destination(attempts/str(attempt))
        attempt_path.mkdir(parents=True)
        attempt_output = attempt_path/'output'
        command = ['bash', str(launcher), str(Path(__file__).resolve()), '--collect-case',
            '--plan', str(Path(plan_path).resolve()), '--case-id', case['case_id'],
            '--output', str(attempt_output), '--guard-report', str(Path(guard_report).resolve())]
        started, status, detail = utc(), 'infrastructure_error', ''
        with (attempt_path/'native.log').open('wb') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                         MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'), start_new_session=True)
            try:
                code = process.wait(timeout=case_timeout)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
                code, detail = None, 'native case wall timeout; no policy outcome assigned'
        if code == 0:
            child = case_directory(attempt_output, case)
            if (child/'case.json').exists():
                validate_case_result(child/'case.json', plan, case)
                # Keep child raw paths/hash receipts. Copying NPZ would change
                # provenance; the outer case receipt references child outputs.
                record = json.loads((child/'case.json').read_text())
                record['parent_process_receipt'] = dict(launcher=str(launcher.relative_to(ROOT)),
                    launcher_sha256=sha(launcher), process_id=process.pid, command=command,
                    child_case_path=str((child/'case.json').relative_to(ROOT)), child_case_sha256=sha(child/'case.json'))
                write_json(out/'case.json', record)
                status = 'completed'
            else:
                detail = 'child exited without a valid completed case receipt'
        receipt = dict(case_id=case['case_id'], status=status, detail=detail, returncode=code,
            process_id=process.pid, started_utc=started, finished_utc=utc(), command=command,
            attempt=attempt, policy_failure_label_assigned=False, gate=gate)
        write_json(attempt_path/'parent_receipt.json', receipt)
        return receipt

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, case) for case in selected_cases]):
            try:
                receipts.append(future.result())
            except Exception as error:
                receipts.append(dict(status='infrastructure_error', detail=str(error), policy_failure_label_assigned=False))
    merged = merge_completed(plan_path, output)
    selected_complete = all(row['status'] in ('completed', 'resumed_verified') for row in receipts)
    write_json(output/'queue_report.json', dict(created_utc=utc(), receipts=receipts,
        workers=workers, complete=bool(merged['complete'] and selected_complete),
        expected=len(plan['cases']), selected_expected=len(selected_cases), selected_complete=selected_complete,
        unchanged_full_plan_subset=case_ids is not None,
        teacher_training_only=True, GPU_used=False, automatic_retries=False), overwrite=True)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare-plan', action='store_true')
    action.add_argument('--prepare-offline-priors', action='store_true')
    action.add_argument('--collect-case', action='store_true')
    action.add_argument('--merge', action='store_true')
    action.add_argument('--run-queue', action='store_true')
    parser.add_argument('--demonstrations', type=Path, default=BASE/'demonstrations/manifest.json')
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output', type=Path, default=BASE/'native_augmentation')
    parser.add_argument('--split', choices=('train', 'validation'), default='train')
    parser.add_argument('--replicas', type=int, default=1)
    parser.add_argument('--sources-per-task', type=int, help='explicit subset budget, 1 required for pre-calibration TRAIN smoke')
    parser.add_argument('--task', action='append', default=[])
    parser.add_argument('--case-id')
    parser.add_argument('--queue-case-id', action='append', help='run a fixed subset of the unchanged full plan; remaining cases stay pending')
    parser.add_argument('--template-assets', type=Path, help='JSON task -> frozen TRAIN asset path')
    parser.add_argument('--rigid-layouts', type=Path, help='JSON task -> frozen TRAIN rigid-layout path')
    parser.add_argument('--offline-priors', type=Path, help='SHA-bound 1400-source offline TRAIN action/phase prior artifact')
    parser.add_argument('--frozen-policy', type=Path, help='JSON factory/code/artifact SHA receipt')
    parser.add_argument('--validation-mode', choices=('hold', 'frozen_policy'), default='hold')
    parser.add_argument('--guard-report', type=Path, default=ROOT/'artifacts/framework/aligned_r09/final_report.json')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--case-timeout', type=int, default=900)
    args = parser.parse_args()
    if args.prepare_offline_priors:
        value = prepare_offline_priors(args.demonstrations, args.output)
        print(json.dumps(dict(sources_read=value['sources_read'], native_started=False)))
        return
    if args.plan is None:
        parser.error('this action requires --plan')
    if args.prepare_plan:
        if args.protocol is None:
            parser.error('--prepare-plan requires --protocol')
        value = build_plan(args.demonstrations, args.protocol, args.plan, split=args.split, replicas=args.replicas,
            tasks=args.task, validation_mode=args.validation_mode,
            template_assets=json.loads(args.template_assets.read_text()) if args.template_assets else None,
            frozen_policy=json.loads(args.frozen_policy.read_text()) if args.frozen_policy else None,
            rigid_layouts=json.loads(args.rigid_layouts.read_text()) if args.rigid_layouts else None,
            offline_priors_path=args.offline_priors, sources_per_task=args.sources_per_task)
        print(json.dumps(dict(plan=str(args.plan), cases=len(value['cases']), native_started=False)))
    elif args.collect_case:
        if not args.case_id:
            parser.error('--collect-case requires --case-id')
        value = collect_native_case(args.plan, args.case_id, args.output, args.guard_report)
        print(json.dumps(dict(case_id=value['case_id'], status=value['status'])))
    elif args.run_queue:
        value = run_queue(args.plan, args.output, args.guard_report, workers=args.workers, case_timeout=args.case_timeout,
                          case_ids=args.queue_case_id)
        print(json.dumps(dict(completed=value['completed'], expected=value['expected'])))
    else:
        value = merge_completed(args.plan, args.output)
        print(json.dumps(dict(completed=value['completed'], expected=value['expected'], native_started=False)))


if __name__ == '__main__':
    main()
