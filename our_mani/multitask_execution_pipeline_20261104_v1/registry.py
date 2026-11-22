"""Real task/source inventory; code and manifest metadata only, no episode reads."""
import ast
import hashlib
import json
from pathlib import Path

BASE_TASKS = (
    'beat_the_rotating_buzz', 'insert_onto_rotating_peg',
    'moving_basketball_in_hoop', 'pick_moving_target_on_the_table',
    'place_cups_on_rotating_frame', 'push_moving_button',
    'put_rubbish_in_moving_bin', 'remove_cups_from_rotating_frame',
)
TASKS = tuple(t + suffix for t in BASE_TASKS for suffix in ('', '_high_speed'))
METHOD = 'current_network'
NATIVE_REPO = Path('/home/ubuntu/桌面/phsimani/PhysMani')

# Descriptions are audit metadata, never a replacement implementation of success.
CONTRACTS = {
    'beat_the_rotating_buzz': (
        'wand detected at right_sensor AND nothing grasped; middle_sensor is native failure',
        ('wand', 'rotating_frame', 'right_end'),
        ('approach', 'acquire', 'gap_follow', 'release'),
        'rotating author reset; preserve first-terminal failure'),
    'insert_onto_rotating_peg': (
        'square_ring detected simultaneously by all four success_detector0..3; no explicit release requirement',
        ('ring', 'selected_coloured_peg', 'rotating_frame'),
        ('approach', 'acquire', 'lift', 'align', 'insert'),
        'author peg reset with bool_random_place=False; do not invent insertion candidates'),
    'moving_basketball_in_hoop': (
        'ball detected in success sensor AND nothing grasped AND open-gripper condition AND has-been-picked latch',
        ('ball', 'moving_hoop'), ('approach', 'acquire', 'lift', 'deposit', 'release'),
        'two independently moving author drivers'),
    'pick_moving_target_on_the_table': (
        'GraspedCondition(target_block) AND DetectedCondition(target_block, pick_and_lift_success)',
        ('Cube', 'public_goal_marker'), ('approach', 'acquire', 'lift', 'transport'),
        'native detection volume is not a public 20mm centre-distance goal'),
    'place_cups_on_rotating_frame': (
        'nothing grasped AND each required cup detected by at least one peg sensor, per native variation',
        ('cups', 'selected_spokes', 'rotating_frame'),
        ('approach', 'acquire', 'lift', 'align', 'hang', 'release'),
        'variation selects number of required cups; retain native OR conditions'),
    'push_moving_button': (
        'native JointCondition(0.003m) AND GripperDescendingCondition simultaneously',
        ('button', 'top_plate'), ('approach', 'press', 'verify'),
        'native descending-history condition is part of task.success(), not a geometric surrogate'),
    'put_rubbish_in_moving_bin': (
        'rubbish detected by native success sensor; no additional release predicate invented',
        ('rubbish', 'moving_bin'), ('approach', 'acquire', 'lift', 'deposit'),
        'retain native rubbish identity and moving-bin driver'),
    'remove_cups_from_rotating_frame': (
        'nothing grasped AND native OR of cup/table-sensor detections',
        ('cups', 'rotating_frame', 'table_destination'),
        ('approach', 'acquire', 'unhang', 'place', 'release'),
        'preserve the actual native OR predicate; do not silently strengthen it to all cups'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_family(task):
    if task not in TASKS:
        raise ValueError('Only the explicit sixteen native task identifiers are supported')
    return task.removesuffix('_high_speed')


def inventory(root, native_repo=NATIVE_REPO):
    """Read source text and frozen roster JSON, never any HDF/pickle/TEST episode."""
    root, native_repo = Path(root), Path(native_repo)
    rosters = {}
    for split, name in (('train', 'train_manifest.json'),
                        ('validation', 'validation_manifest.json'),
                        ('test', 'test_manifest_all16.json')):
        path = root / 'artifacts/framework/protocol' / name
        value = json.loads(path.read_text())
        rows = value['episodes']
        rosters[split] = dict(path=str(path.resolve()), sha256=sha(path),
            counts={t: sum(r['task'] == t for r in rows) for t in TASKS})
    common = {str(native_repo / name): sha(native_repo / name) for name in (
        'third_party/RLBench/rlbench/backend/task.py',
        'third_party/RLBench/rlbench/backend/conditions.py')}
    rows = []
    for task in TASKS:
        family = task_family(task)
        path = native_repo / 'third_party/RLBench/rlbench/tasks' / (task + '.py')
        source = path.read_text()
        tree = ast.parse(source)
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
                   and any(isinstance(b, ast.Name) and b.id == 'Task' for b in n.bases)]
        if len(classes) != 1:
            raise ValueError('One real native Task class required: ' + task)
        registrations = []
        for node in ast.walk(classes[0]):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ('register_success_conditions', 'register_fail_conditions')):
                registrations.append(dict(line=node.lineno, expression=ast.get_source_segment(source, node)))
        description, roles, phases, note = CONTRACTS[family]
        rows.append(dict(task=task, family=family,
            operation_id=BASE_TASKS.index(family), normal_high_share_operation=True,
            speed='high_speed' if task.endswith('_high_speed') else 'normal',
            native_class=classes[0].name, native_module='rlbench.tasks.' + task,
            native_source=dict(path=str(path), sha256=sha(path)),
            native_condition_sources=common,
            native_success_authority='task.success()', native_condition_summary=description,
            native_registration_source=registrations, public_roles_required=list(roles),
            phase_adapter_requirements=list(phases), reset_note=note,
            metadata_episode_counts={s: rosters[s]['counts'][task] for s in rosters},
            current_network_execution_ready=False,
            blockers=['current-network task observation/phase/reset adapter must be explicitly bound',
                      'real current-network training and Native evaluation hooks not bound'],
            old_R10_controller_is_current_network_support=False))
    return dict(schema='current_network16_task_inventory_20261104_v1', method=METHOD,
        tasks=rows, roster_metadata=rosters,
        public_observation_boundary=dict(
            reusable_code='our_mani/multitask/simulation.py:MultitaskSimulation._observe_current',
            reusable_sensor_code='our_mani/sensor_alignment/author_rgbd.py',
            legacy_caveat='R10 also returns native instance masks and native grasped-object contact; these are not automatically admitted current-network inputs',
            required='current/past RGBD + public K/T + robot proprioception + explicit public task cues',
            Native_poses_masks_contact_success_or_expert_as_policy_inputs_allowed=False),
        episode_payloads_opened=0, raw_TEST_files_opened=0, Native_started=False,
        model_forward_calls=0, training_started=False)
