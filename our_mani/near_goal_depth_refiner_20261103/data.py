"""Current public crops and separate counterfactual labels, never transitions."""
from pathlib import Path
import json

import numpy as np
from torch.utils.data import Dataset

from ..near_goal_segmentation_20261103.data import (
    CAMERAS, CROP, FEATURE_NAMES, PUBLIC_KEYS, PARTITION_IDS, PILOT_IDS, SHOT_IDS,
    ROLE_MAP, ROLE_COUNTS, sha256, calibration_digest, load_npz,
    validate_public, make_near_goal_crops, current_cube_measurement,
)

SCHEMA = 'near_goal_depth_influence_capture_TRAIN_v1'
CLASS_NAMES = ('known_no_Cube_depth_influence', 'Cube_current_depth_influence')
LABEL_PROTOCOL = 'raw_Cube_depth_influence_innerAA_positive_outer1px_unknown_v1'
IGNORE = 255
PROTOCOL_FILES = {
    'artifacts/framework/near_goal_segmentation_20261103/binary_depth_influence_protocol_v1/protocol.json':
        '34f8ef0256d6320396d6a465133ad3252ff6d2d1a248c257c86485c8a8069081',
    'artifacts/framework/near_goal_segmentation_20261103/binary_depth_influence_protocol_v1/boundary_addendum.json':
        'edf45a08ad37a5311253e328384dc93ce61f90fefd48c699727feb24ff741ecc',
}


def erode_one_pixel(mask):
    mask = np.asarray(mask)
    if mask.ndim != 3 or mask.dtype != np.bool_:
        raise ValueError('bool[camera,height,width] mask required')
    padded = np.pad(mask, ((0, 0), (1, 1), (1, 1)), constant_values=False)
    result = np.ones_like(mask)
    for dy in range(3):
        for dx in range(3):
            result &= padded[:, dy:dy+mask.shape[1], dx:dx+mask.shape[2]]
    return result


def _boundary(mask):
    padded = np.pad(mask, ((0, 0), (1, 1), (1, 1)), constant_values=False)
    dilated = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            dilated |= padded[:, dy:dy+mask.shape[1], dx:dx+mask.shape[2]]
    return dilated & ~erode_one_pixel(mask)


def binary_depth_labels(inputs, labels):
    """Raw positives incl. innerAA; zero-delta outer ring stays ignore255.

    No Cube pose/presence, robot, history, semantic-ID map or GT is read here.
    All classes refer to the current normal-renderer depth measurement only.
    """
    validate_public(inputs)
    if not isinstance(labels, dict):
        raise ValueError('physically separate numeric counterfactual labels required')
    base = inputs['depth_m']
    derived = {}
    for role in ('cube', 'goal'):
        without = labels.get('without_'+role+'_depth_m')
        if (not isinstance(without, np.ndarray) or without.shape != base.shape
                or without.dtype != np.float32 or not np.isfinite(without).all()
                or np.any(without < 0)):
            raise ValueError('actual finite float32 counterfactual metric-depth label required')
        delta = without.astype(float) - base.astype(float)
        valid = np.isfinite(base) & (base > 0) & (without > 0)
        positive, negative = valid & (delta > 0), valid & (delta < 0)
        derived[role] = dict(delta=delta, valid=valid, positive=positive,
                             negative=negative, invalid=~valid, boundary=_boundary(positive))
        for suffix, expected in (('raw_positive', positive), ('negative_delta', negative),
                                 ('invalid_depth', ~valid), ('one_pixel_boundary', _boundary(positive))):
            stored = labels.get(role+'_'+suffix)
            if (not isinstance(stored, np.ndarray) or stored.dtype != np.bool_
                    or stored.shape != base.shape or not np.array_equal(stored, expected)):
                raise ValueError('counterfactual label plane differs from actual depth: '+role+'_'+suffix)
        if not np.array_equal(labels.get(role+'_signed_depth_delta_m'), delta):
            raise ValueError('saved signed-depth delta differs from actual base/counterfactual')
    conflict = derived['cube']['positive'] & derived['goal']['positive']
    if not np.array_equal(labels.get('conflicting_depth_owner'), conflict):
        raise ValueError('raw positive owner conflict must remain explicit')
    goal_rgb = labels.get('without_goal_rgb')
    if not isinstance(goal_rgb, np.ndarray) or goal_rgb.shape != inputs['rgb'].shape or goal_rgb.dtype != np.uint8:
        raise ValueError('actual label-only withoutGoalRGB required for unresolved RGB response')
    goal_rgb_only = (np.any(goal_rgb != inputs['rgb'], axis=-1)
                     & ~derived['goal']['positive'] & ~derived['cube']['positive'])
    if not np.array_equal(labels.get('goal_rgb_only_unknown'), goal_rgb_only):
        raise ValueError('Goal RGB-only response cannot become a measured-depth role')
    ambiguity = conflict | goal_rgb_only
    for role in derived.values():
        ambiguity |= role['negative'] | role['invalid']
    positive = derived['cube']['positive'] & ~ambiguity
    negative = (derived['cube']['valid'] & (derived['cube']['delta'] == 0)
                & ~ambiguity & ~derived['cube']['boundary'])
    target = np.full(base.shape, IGNORE, np.uint8)
    target[negative], target[positive] = 0, 1
    return target


def crop_labels(labels, crops):
    if (not isinstance(labels, np.ndarray) or labels.dtype != np.uint8
            or labels.shape != (4, 256, 256) or not set(np.unique(labels)) <= {0, 1, IGNORE}):
        raise ValueError('binary current-depth labels0/1/255 required')
    result = np.full((4, CROP, CROP), IGNORE, np.uint8)
    yy, xx = np.indices((CROP, CROP))
    for camera in range(4):
        valid = crops['valid_crop'][camera]
        x, y = xx+crops['origins_xy'][camera, 0], yy+crops['origins_xy'][camera, 1]
        result[camera, valid] = labels[camera, y[valid], x[valid]]
    return result


def make_measurement(inputs, crops, raw_mask):
    """One erosion then exact old quantile algebra; inspect crop edge BEFORE it."""
    raw_mask = np.asarray(raw_mask)
    if raw_mask.shape != (4, CROP, CROP) or raw_mask.dtype != np.bool_:
        raise ValueError('current binary bool4x64x64 raw mask required')
    eroded = erode_one_pixel(raw_mask)
    result = current_cube_measurement(inputs, crops, eroded)
    result.pop('mask_is_current_visible_pixels_only', None)
    if (raw_mask[:, 0, :].any() or raw_mask[:, -1, :].any()
            or raw_mask[:, :, 0].any() or raw_mask[:, :, -1].any()):
        result['reasons'].append('raw_Cube_mask_touches_crop_boundary_before_erosion')
    if np.any(raw_mask & ~crops['valid_crop']):
        result['reasons'].append('raw_Cube_prediction_outside_observed_crop')
    result['refiner_available'] = result['refiner_available'] and not result['reasons']
    if not result['refiner_available']:
        result['centre'] = None
    result.update(raw_positive_pixels=int(raw_mask.sum()), eroded_pixels=int(eroded.sum()),
        erosion_pixels_removed=int(raw_mask.sum()-eroded.sum()), geometry_erosions=1,
        label_semantics=LABEL_PROTOCOL, complete_visible_instance_claimed=False)
    return result


def checked_pointer(base, pointer):
    if not isinstance(pointer, dict) or set(pointer) != {'path', 'sha256'}:
        raise ValueError('exact immutable path/SHA pointer required')
    path = (Path(base)/pointer['path']).resolve(strict=True)
    if any('test' in part.lower() for part in path.parts) or sha256(path) != pointer['sha256']:
        raise ValueError('TRAIN-only bound pointer changed')
    return path


class DepthInfluenceDataset(Dataset):
    """Complete fixed TRAIN parents only; labels are never forward arguments."""
    def __init__(self, manifest_path, role=None):
        self.path = Path(manifest_path).resolve(strict=True)
        if any('test' in part.lower() for part in self.path.parts):
            raise ValueError('Only explicit TRAIN manifests, never TEST')
        self.manifest = json.loads(self.path.read_text())
        manifest = self.manifest
        if (manifest.get('schema') != SCHEMA or manifest.get('dataset_split') != 'TRAIN'
                or manifest.get('label_protocol') != LABEL_PROTOCOL
                or any(manifest.get(key) is not True for key in (
                    'complete', 'all_native_closed', 'all_process_exits_observed', 'source_unchanged'))):
            raise ValueError('Completed closed TRAIN binary-depth-influence manifest required')
        self.scope = manifest.get('scope')
        if self.scope not in ROLE_COUNTS:
            raise ValueError('Only fixed pilot_smoke/full32 scope')
        self.bound_pointers = {}
        def pointer(ref, *, base=None):
            path = checked_pointer(self.path.parent if base is None else base, ref)
            self.bound_pointers[str(path)] = ref['sha256']
            return path
        self.partition_path = pointer(manifest['partition'])
        self.partition_sha256 = sha256(self.partition_path)
        self.capture_plan_path = pointer(manifest['capture_plan'])
        partition = json.loads(self.partition_path.read_text())
        declared = partition.get('parents', [])
        if len(declared) != 32 or {p.get('parent_id') for p in declared} != set(PARTITION_IDS):
            raise ValueError('Entire exact predeclared32-parent partition required')
        partition_rows = {p['parent_id']: p for p in declared}
        for i, parent_id in enumerate(PARTITION_IDS):
            parent = partition_rows[parent_id]
            expected = 'fit' if i < 24 else ('selection' if i < 28 else 'calib')
            if (parent.get('seed') != 261103600+i or ROLE_MAP.get(parent.get('role')) != expected
                    or tuple(parent.get('shot_ids', ())) != SHOT_IDS
                    or parent.get('pilot') is not (parent_id in PILOT_IDS)):
                raise ValueError('Preassigned seed/fold/pilot/all8shot identity changed')
        self.capture_plan = json.loads(self.capture_plan_path.read_text())
        if (self.capture_plan.get('split') != 'train'
                or self.capture_plan.get('label_method') != 'counterfactual_current_depth_owner'
                or self.capture_plan.get('native_instance_id_masks') is not False
                or self.capture_plan.get('material_or_alpha_edits') is not False
                or type(self.capture_plan.get('render_warmup_rounds')) is not int
                or self.capture_plan['render_warmup_rounds'] != 12):
            raise ValueError('Pinned TRAIN depth-owner capture plan required')
        plan_rows = {p['parent_id']: p for p in self.capture_plan.get('parents', [])}
        parents = manifest.get('parents', [])
        expected_ids = PILOT_IDS if self.scope == 'pilot_smoke' else PARTITION_IDS
        if (len(parents) != len(expected_ids)
                or {p.get('parent_id') for p in parents} != set(expected_ids)):
            raise ValueError('Exact whole-parent cohort, no missing or extra source')
        self.all_records, scenes, data_paths = [], set(), set()
        self.q_source_groups = {}
        self.capture_source_identity = None
        repository = Path(__file__).resolve().parents[2]
        for name, expected_sha in PROTOCOL_FILES.items():
            path = repository/name
            if sha256(path) != expected_sha:
                raise ValueError('Predeclared binary protocol changed')
            self.bound_pointers[str(path)] = expected_sha
        for parent in parents:
            parent_id = parent['parent_id']
            declared = partition_rows[parent_id]
            planned = plan_rows.get(parent_id, {})
            if (parent.get('seed') != declared['seed'] or parent.get('role') != declared['role']
                    or planned.get('seed') != declared['seed']
                    or ROLE_MAP.get(planned.get('fold', planned.get('role'))) != ROLE_MAP[declared['role']]):
                raise ValueError('Capture/manifest parent differs from fixed seed and fold')
            planned_q = planned.get('TRAIN_q_source')
            if (not isinstance(planned_q, dict) or not isinstance(planned_q.get('dataset'), str)
                    or not planned_q['dataset'] or type(planned_q.get('index')) is not int
                    or planned_q['index'] < 0):
                raise ValueError('Explicit parent-planned actual TRAIN q source required')
            q_path = pointer(planned_q['file'])
            q_group = (str(q_path), planned_q['file']['sha256'], planned_q['dataset'], planned_q['index'])
            fold = ROLE_MAP[declared['role']]
            if q_group in self.q_source_groups and self.q_source_groups[q_group] != fold:
                raise ValueError('One exact TRAIN q source group cannot cross fit/selection/calibration folds')
            self.q_source_groups[q_group] = fold
            capture = parent.get('capture', {})
            if set(capture) != {'result', 'owner_actual_exit', 'run_inputs', 'source_manifest'}:
                raise ValueError('All four actual capture evidence pointers required')
            evidence = {key: pointer(value) for key, value in capture.items()}
            result, owner, run, sources = (json.loads(evidence[key].read_text()) for key in (
                'result', 'owner_actual_exit', 'run_inputs', 'source_manifest'))
            case = evidence['result'].parent
            if (evidence['run_inputs'].parent != case or evidence['source_manifest'].parent != case/'source_snapshot'
                    or owner.get('actual_process_returncode') != 0 or type(owner.get('actual_process_returncode')) is not int
                    or any(owner.get(key) is not True for key in ('actual_wait_completed', 'source_unchanged'))
                    or owner.get('stop_reason') is not None
                    or any(result.get(key) is not True for key in ('complete', 'native_closed', 'source_unchanged'))
                    or result.get('controlled_physics_ticks') != 0
                    or result.get('parent_id') != parent_id or owner.get('parent_id') != parent_id
                    or run.get('parent_id') != parent_id or run.get('seed') != parent['seed']
                    or ROLE_MAP.get(run.get('fold')) != ROLE_MAP[declared['role']]
                    or run.get('TRAIN_q_source') != planned_q
                    or run.get('plan_sha256') != sha256(self.capture_plan_path)
                    or run.get('label_method') != 'counterfactual_current_depth_owner'):
                raise ValueError('Actual closed0exit same-source TRAIN capture evidence required')
            archived_files = sources.get('files', [])
            if not archived_files:
                raise ValueError('Complete byte archived capture source graph required')
            archived = {}
            for row in archived_files:
                path = (evidence['source_manifest'].parent/row['archive']).resolve(strict=True)
                if evidence['source_manifest'].parent not in path.parents or sha256(path) != row['sha256']:
                    raise ValueError('Archived source missing or changed')
                self.bound_pointers[str(path)] = row['sha256']
                if row['original'] in archived:
                    raise ValueError('Duplicated source archive identity')
                archived[row['original']] = row['sha256']
            expected_sources = {str((repository/name).resolve()): digest for name, digest in run['source_sha256'].items()}
            expected_sources.update(run['external_source_sha256'])
            if (not run['source_sha256'] or not run['external_source_sha256']
                    or run['source_sha256'] != self.capture_plan.get('source_sha256')
                    or run['external_source_sha256'] != self.capture_plan.get('external_source_sha256')
                    or run.get('native_config') != self.capture_plan.get('native_config')
                    or any(archived.get(name) != digest for name, digest in expected_sources.items())):
                raise ValueError('Capture run source graph differs from exact byte archive')
            if sha256(case/'source_snapshot/plan.json') != run['plan_sha256']:
                raise ValueError('Archived capture plan changed')
            native_before = run.get('Native_sources_sha256')
            if not native_before or result.get('Native_sources_sha256_after') != native_before:
                raise ValueError('Actual original Native source fingerprint changed')
            source_identity = dict(repo=run['source_sha256'], external=run['external_source_sha256'], native=native_before)
            if self.capture_source_identity is None:
                self.capture_source_identity = source_identity
            elif self.capture_source_identity != source_identity:
                raise ValueError('Parent capture source maps differ')
            initial_path, fit_path = pointer(parent['initial_public_rgbd']), pointer(parent['public_sphere_fit'])
            if initial_path != case/'initialization/public_rgbd.npz' or fit_path != case/'public_initial_sphere.json':
                raise ValueError('Actual immutable parent initial public sphere sources required')
            initial, fit = load_npz(initial_path), json.loads(fit_path.read_text())
            validate_public(initial)
            if (fit.get('Native_pose_used') is not False
                    or fit.get('metadata', {}).get('simulator_pose_used') is not False
                    or not np.array_equal(initial['public_sphere_center'], fit['sphere']['center'])
                    or float(initial['timestamp']) != 0):
                raise ValueError('Public first-frame sphere only; no refit or GT fallback')
            result_frames = {f['shot_id']: f for f in result.get('shots', [])}
            shots = parent.get('shots', [])
            if (len(result_frames) != 8 or tuple(s.get('shot_id') for s in shots) != SHOT_IDS
                    or set(result_frames) != set(SHOT_IDS)
                    or tuple(s.get('shot_id') for s in planned.get('shots', [])) != SHOT_IDS):
                raise ValueError('Every exact preassigned8shot source retained; two-shot smoke is not training data')
            planned_shots = {s['shot_id']: s for s in planned['shots']}
            for shot in shots:
                frame = result_frames[shot['shot_id']]
                if (shot.get('scene_id') != parent_id+'/'+shot['shot_id'] or shot['scene_id'] in scenes
                        or frame.get('scene_id') != shot['scene_id'] or shot.get('independent_scene') is not True
                        or type(shot.get('physics_ticks')) is not int or shot['physics_ticks'] != 0
                        or shot.get('condition') not in ('sphere_only', 'cube_far', 'near_goal')
                        or planned_shots[shot['shot_id']].get('condition') != shot['condition']
                        or not isinstance(shot.get('condition_detail'), str) or not shot['condition_detail']
                        or planned_shots[shot['shot_id']].get('condition_detail') != shot['condition_detail']):
                    raise ValueError('Fixed independent shot/condition/source identity required')
                camera = frame.get('camera', {})
                if (camera.get('label_method') != 'counterfactual_current_depth_owner'
                        or any(camera.get(key) is not True for key in (
                            'original_clone_equal', 'all_counterfactual_flags_restored', 'all_restore_RGBD_equal',
                            'clock_unchanged', 'handling_restored', 'global_vision_sensors_enabled', 'created_nonpassive'))
                        or camera.get('warmup_rounds_fixed') != 12
                        or frame.get('augmentation', {}).get('restored') is not True
                        or frame.get('timestamp') != 0):
                    raise ValueError('Every actual zero-tick shot requires exact renderer/wholeflags/restore evidence')
                scenes.add(shot['scene_id'])
                record = dict(shot, parent_id=parent_id, role=ROLE_MAP[declared['role']],
                              declared_role=declared['role'], initial_public=initial)
                for key, folder, frame_key in (('inputs', 'inputs', 'input'), ('labels', 'labels', 'labels')):
                    path = pointer(shot[key])
                    if (path.parent != case/folder or path in data_paths
                            or path != (case/frame[frame_key]['path']).resolve()
                            or shot[key]['sha256'] != frame[frame_key]['sha256']):
                        raise ValueError('Original sibling input/label pointer or SHA differs')
                    data_paths.add(path)
                    record[key+'_path'] = path
                metadata_path = pointer(shot['annotation_metadata'])
                if metadata_path.parent != case/'labels':
                    raise ValueError('Presence/construction proof metadata belongs in labels sibling')
                metadata = json.loads(metadata_path.read_text())
                if (metadata.get('scene_id') != shot['scene_id'] or metadata.get('shot_id') != shot['shot_id']
                        or metadata.get('condition') != shot['condition']
                        or metadata.get('condition_detail') != shot['condition_detail']
                        or metadata.get('cube_scene_present') is not True
                        or type(metadata.get('absence_from_all_camera_frusta_by_construction')) is not bool):
                    raise ValueError('Explicit label-only scene presence/construction metadata required')
                absent = metadata['absence_from_all_camera_frusta_by_construction']
                if shot['condition'] == 'sphere_only':
                    if not absent:
                        raise ValueError('Sphere-only is preassigned proven outside cameras, never inferred from zero masks')
                    proof_path = pointer(metadata['absence_proof'], base=case)
                    if proof_path.parent != case/'labels':
                        raise ValueError('Offline frusta proof must stay in the original case labels sibling')
                    proof = json.loads(proof_path.read_text())
                    if (proof.get('verified') is not True or proof.get('all_four_camera_frusta_outside') is not True
                            or proof.get('source') != 'predeclared_analytic_four_camera_frusta'):
                        raise ValueError('Pinned analytic all-camera absence proof required')
                elif absent:
                    raise ValueError('Cube-far/near source cannot silently become Sphere-only')
                record['annotation_metadata'] = metadata
                self.all_records.append(record)
        self.bound_pointers[str(self.path)] = sha256(self.path)
        normalized = ROLE_MAP.get(role, role)
        if role is not None and normalized not in ROLE_COUNTS[self.scope]:
            raise ValueError('Unknown TRAIN-only role')
        self.records = [r for r in self.all_records if role is None or r['role'] == normalized]

    def __len__(self):
        return len(self.records)

    def revalidate(self):
        for name, digest in self.bound_pointers.items():
            if sha256(name) != digest:
                raise ValueError('Bound data/source/manifest changed after readiness')

    def __getitem__(self, index):
        record = self.records[index]
        for key in ('inputs', 'labels'):
            if sha256(record[key+'_path']) != record[key]['sha256']:
                raise ValueError('Current capture changed after dataset load')
        inputs, labels = load_npz(record['inputs_path']), load_npz(record['labels_path'])
        validate_public(inputs)
        initial = record['initial_public']
        if (not np.array_equal(inputs['public_sphere_center'], initial['public_sphere_center'])
                or calibration_digest(inputs) != calibration_digest(initial)
                or float(inputs['timestamp']) != 0 or float(labels.get('timestamp', np.nan)) != 0
                or int(labels.get('counted_tick', -1)) != 0
                or str(labels.get('scene_id')) != record['scene_id']):
            raise ValueError('Same-frame actual zero clock/initial calibration/sphere identity required')
        target = binary_depth_labels(inputs, labels)
        crops = make_near_goal_crops(inputs)
        semantic = crop_labels(target, crops)
        unknown = target == IGNORE
        reason_full = {key: labels[key] & unknown for key in (
            'cube_negative_delta', 'goal_negative_delta', 'cube_invalid_depth',
            'goal_invalid_depth', 'conflicting_depth_owner', 'goal_rgb_only_unknown')}
        reason_full['zero_delta_outer_Cube_boundary'] = labels['cube_one_pixel_boundary'] & ~labels['cube_raw_positive'] & unknown
        covered = np.zeros_like(unknown)
        for mask in reason_full.values():
            covered |= mask
        reason_full['other_unresolved'] = unknown & ~covered
        unknown_reasons = {key: crop_labels(np.where(mask, 1, 0).astype(np.uint8), crops) == 1
                           for key, mask in reason_full.items()}
        pose = np.asarray(labels.get('cube_pose_world'), float)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError('Actual labels-only Cube pose required for offline scoring')
        metadata = record['annotation_metadata']
        return dict(features=crops['features'], semantic=semantic, inputs=inputs, crops=crops,
            cube_center_world=pose[:3].copy(), cube_scene_present=metadata['cube_scene_present'],
            absence_from_all_camera_frusta_by_construction=metadata['absence_from_all_camera_frusta_by_construction'],
            parent_id=record['parent_id'], scene_id=record['scene_id'], role=record['role'],
            condition=record['condition'], condition_detail=record.get('condition_detail'),
            unknown_reason_masks=unknown_reasons,
            frame_weight=1/8., label_semantics=LABEL_PROTOCOL)
