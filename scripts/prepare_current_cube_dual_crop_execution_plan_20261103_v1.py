"""Materialize or check only the fixed40/520 capture metadata; never Native.

The original preparation manifest and shared source-parent split stay immutable.
This derived plan binds Root's uniform .16-palm asset reuse before images exist.
Training/evaluation code is deliberately outside the capture dependency graph.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.local_mpc_20261103 import current_cube_dual_crop_layout_cap16_v2 as layout

PREPARATION = ROOT/'artifacts/framework/local_mpc_20261102/current_RGBD_centre_TRAIN_data_audit_v1/dual_crop_capture_preparation_manifest_v2.json'
SHARED = ROOT/'artifacts/framework/local_mpc_20261102/current_Cube_shared_TRAIN_source_parent_registry_v1/registry.json'
Q_REGISTRY = ROOT/'artifacts/framework/local_mpc_20261102/P3_dual_crop_CF_pilot40_TRAIN_q_registry_v1/q_sources.json'
PROFILE = ROOT/'artifacts/framework/local_mpc_20261102/current_cube_dual_crop_CF_hand_layout_profile_v2/protocol.json'
BASE = ROOT/'config/framework/P3_near_goal_depth_owner_position_calibration8_v10_20261103_v1.json'
DEFAULT_DIR = ROOT/'config/framework/current_cube_dual_crop_CF_pilot40_20261103_v1'
OUTPUT_ROOT = 'runtime/framework/near_goal_perception_20261103/current_cube_dual_crop_CF_pilot40_v2'
CORE_FILES = (
    'our_mani/local_mpc_20261103/current_cube_dual_crop_capture_v1.py',
    'our_mani/local_mpc_20261103/current_cube_dual_crop_capture_v2.py',
    'our_mani/local_mpc_20261103/current_cube_dual_crop_layout_v1.py',
    'our_mani/local_mpc_20261103/current_cube_dual_crop_layout_cap16_v2.py',
    'scripts/collect_current_cube_dual_crop_20261103_v1.py',
    'scripts/launch_current_cube_dual_crop_20261103_v1.py',
    'scripts/run_current_cube_dual_crop_20261103_v1.sh',
    'scripts/collect_current_cube_dual_crop_20261103_v2.py',
    'scripts/launch_current_cube_dual_crop_20261103_v2.py',
    'scripts/run_current_cube_dual_crop_20261103_v2.sh',
    'scripts/run_current_cube_dual_crop_capture_queue_20261103_v1.py',
    'scripts/prepare_current_cube_dual_crop_execution_plan_20261103_v1.py',
    'our_mani/current_cube_rgbd_centre_20261103_v1/features.py',
    'our_mani/current_cube_rgbd_centre_20261103_v1/model.py',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def pointer(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def metadata():
    expected = {PREPARATION: 'befefa20fd2c0d26635729f38138ba4ba8b459c6c31ab3083f753f01cbf5e32c',
        SHARED: '76e836b6a872dce703598b18a694953d4ae48be69458bfde7bbde0c214c3cb9c',
        Q_REGISTRY: 'c482d57a9145924471ddf4bb0b7f4887805d09e89a42d2574a85c8d22684c272',
        PROFILE: '9b5d2bdf8414ad20c0e7fc752ad30149b988628aa7c1622fb9ccd28988224c28'}
    if any(sha(path) != digest for path, digest in expected.items()):
        raise ValueError('Immutable metadata/assignment/profile changed')
    preparation, q = read(PREPARATION), read(Q_REGISTRY)
    jobs = {p['parent_id']: p for p in preparation['jobs']}
    ids = preparation['minimal_pilot_capture']['parent_ids']
    if [row['parent_id'] for row in q['rows']] != ids:
        raise ValueError('Fixed metadata-first parent order required')
    rows = []
    for ident, qrow in zip(ids, q['rows']):
        old = jobs[ident]
        row = dict(parent_id=ident, seed=old['seed'], role=old['role'], fold=old['role'],
            pilot=True, source_task=old['source_task'], TRAIN_q_source=deepcopy(qrow),
            near_direction_ordinal=old['direction_ordinal'],
            near_direction_unit_world=old['direction_unit_world'],
            cube_rotation_degrees=old['cube_rotation_degrees'], cube_rotation_axis=old['cube_rotation_axis'])
        row['shots'] = layout.shot_specs_for_parent(row)
        if [s['condition'] for s in row['shots']] != [s['condition'] for s in old['recipes']]:
            raise ValueError('All original13 condition recipes required')
        rows.append(row)
    if dict(Counter(p['role'] for p in rows)) != dict(fit=16, selection=16, reserved_fresh_centre_VAL=8):
        raise ValueError('Fixed16/16/8 split required')
    return rows


def source_closure(partition):
    original = read(BASE)
    sources = deepcopy(original['source_sha256'])
    for name, expected in sources.items():
        if sha(ROOT/name) != expected:
            raise ValueError('Frozen original dependency changed: '+name)
    for name, expected in original['external_source_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Frozen external dependency changed: '+name)
    more = list(CORE_FILES) + [str(p.relative_to(ROOT)) for p in
        (PREPARATION, SHARED, Q_REGISTRY, PROFILE, partition, Path(read(Q_REGISTRY)['assignment']['path']))]
    for name in more:
        sources[name] = sha(ROOT/name)
    return sources, deepcopy(original['external_source_sha256'])


def materialize(directory):
    directory = Path(directory).resolve()
    if ROOT/'config/framework' not in directory.parents or directory.exists():
        raise ValueError('Fresh versioned config/framework directory required')
    parents = metadata()
    base = read(BASE)
    directory.mkdir(parents=True, exist_ok=False)
    partition = directory/'partition.json'
    write(partition, dict(schema='CURRENT_CUBE_dual_crop_CF_pilot40_partition_20261103_v1',
        scope='current_cube_dual_crop_CF_pilot40', pilot=True, split='TRAIN',
        source_parent_registry=pointer(SHARED), metadata_preparation_manifest=pointer(PREPARATION),
        TRAIN_q_registry=pointer(Q_REGISTRY),
        parents=[dict(parent_id=p['parent_id'], seed=p['seed'], role=p['role'], fold=p['fold'],
            pilot=True, shot_ids=list(layout.SHOT_IDS)) for p in parents]))
    sources, external = source_closure(partition)
    plan = {key: deepcopy(base[key]) for key in ('native_config', 'goal_calibration',
        'frozen_environment_sha256', 'baseline_finger_normalization', 'label_method',
        'render_warmup_rounds', 'joint_limits', 'kinematics', 'marker_offset_train_m',
        'DLS', 'world_outside_position_m', 'tool_quaternion_xyzw', 'fixed_seed_deltas_rad')}
    plan.update(schema='CURRENT_CUBE_dual_crop_CF_pilot40_plan_20261103_v1',
        scope='current_cube_dual_crop_CF_pilot40', split='train', perception_only=True,
        preparation_only=True, execution_authorized=False, capture_ready=False,
        Native_started=False, fit_started=False, TEST_read=False, benchmark_episode=False,
        execution_transition=False, native_instance_id_masks=False, material_or_alpha_edits=False,
        source_parent_registry=pointer(SHARED), metadata_preparation_manifest=pointer(PREPARATION),
        TRAIN_q_registry=pointer(Q_REGISTRY), partition=pointer(partition),
        layout_profile=pointer(PROFILE), hand_pad_radius_cap_m=.16, parents=parents,
        source_sha256=sources, external_source_sha256=external,
        collector_script='scripts/collect_current_cube_dual_crop_20261103_v2.py',
        owner_launcher='scripts/launch_current_cube_dual_crop_20261103_v2.py',
        owner_shell='scripts/run_current_cube_dual_crop_20261103_v2.sh',
        capture_queue='scripts/run_current_cube_dual_crop_capture_queue_20261103_v1.py',
        capture_infrastructure_revision='independent_v2_fullasset_reuse_and_unknown_absence',
        target_label_protocol='current_cube_crop_depth_influence_availability_20261103_v1',
        actual_Native_capture_task='pick_moving_target_on_the_table',
        source_task_is_TRAIN_q0_provenance_not_task_motion_replay=True,
        source_q_and_Cube_pose_never_NN_inputs=True,
        static_visual_placements_are_not_grasp_or_attachment=True,
        public_Sphere_anchor='one actualRGBD outsideCube+TRAINq0 bootstrap, cachedsame for13shots; neverNativeSpherepose',
        unavailable_outside_frusta_proof='UNKNOWN keeprequestedrow, nevernegative orreplaced',
        all_13_shots_preserved_regardless_visibility_or_geometry=True,
        parents_count=40, per_parent_shots=13, total_shots=520,
        role_counts=dict(fit=16, selection=16, reserved_fresh_centre_VAL=8),
        output_root=OUTPUT_ROOT, first_fit_preview_parent_id=parents[0]['parent_id'],
        execution_stages=['fixed_first_fit_13shot_QA', 'samefrozenplan_prefixaudit_then_remaining39'],
        automatic_continuation_after_preview=False,
        downstream_training_evaluation_sources_not_capture_dependencies=True,
        required_pending=['Root finalsource/plan execution admission', 'firstfit13shotQA before remaining39'])
    path = directory/'plan.json'
    write(path, plan)
    return path


def refresh_draft(path):
    path = Path(path).resolve(strict=True)
    plan = read(path)
    if plan.get('preparation_only') is not True or plan.get('execution_authorized') is not False:
        raise ValueError('Only unadmitted draft source pins may be refreshed')
    partition = ROOT/plan['partition']['path']
    if sha(partition) != plan['partition']['sha256']:
        raise ValueError('Fixed partition changed')
    snapshot = path.with_name('plan.before_source_refresh.'+sha(path)+'.json')
    with snapshot.open('xb') as stream:
        stream.write(path.read_bytes())
    plan['source_sha256'], plan['external_source_sha256'] = source_closure(partition)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_DIR)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--materialize', action='store_true')
    actions.add_argument('--refresh-draft-source-pins', action='store_true')
    args = parser.parse_args(argv)
    path = args.output_dir.resolve()/'plan.json'
    if args.materialize:
        path = materialize(args.output_dir)
    elif args.refresh_draft_source_pins:
        refresh_draft(path)
    if not path.exists():
        parents = metadata()
        print(json.dumps(dict(preparation_only=True, execution_authorized=False,
            capture_ready=False, parents=len(parents), shots=520, Native_started=False,
            required_action='--materialize creates metadata only')))
        return 0
    from our_mani.local_mpc_20261103.current_cube_dual_crop_capture_v2 import check_capture_plan
    result = check_capture_plan(ROOT, path)
    result.update(plan=str(path), plan_sha256=sha(path), output_root=OUTPUT_ROOT)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
