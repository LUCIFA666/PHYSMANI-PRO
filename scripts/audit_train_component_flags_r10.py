"""Seven isolated TRAIN reset/current-read component audits, zero actions.

Never change native properties or a frozen collector. Preparation and tests
are CPU-only; --run is a separately authorized native diagnostic. Component
selection is reported, not installed into perception, maps, or an experiment.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
BASE = ROOT / 'artifacts/framework/multitask_r10'
VERSION = 'r10_TRAIN_native_component_flag_audit_v2'
from our_mani.multitask.component_flags import (
    SCHEMA, partition_components, read_native_component_flags,
)
from our_mani.multitask.specs import TASK_SPECS


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def process_ancestry():
    """xvfb-run is an ancestor, not necessarily the actual Python PID."""
    result = []; current = os.getpid()
    while current > 0 and current not in result:
        result.append(current)
        try:
            fields = Path('/proc', str(current), 'stat').read_text().rsplit(')', 1)[1].split()
            current = int(fields[1])
        except (OSError, ValueError, IndexError): break
    return result


def require(value, message):
    if not value: raise ValueError(message)


def write(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE.resolve()) and not path.exists(), 'new diagnostic output only')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.' + str(os.getpid()) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def ordinary_tasks():
    tasks = tuple(name for name, spec in TASK_SPECS.items() if spec.speed == 'normal')
    require(len(tasks) == 7 and len({TASK_SPECS[name].family for name in tasks}) == 7,
            'exact seven ordinary family representatives required')
    return tasks


def audit_sources(collection):
    repo = Path(collection['protocol']['baseline_repo'])
    paths = [Path(__file__), ROOT / 'our_mani/multitask/component_flags.py',
             ROOT / 'our_mani/multitask/safe_native_mesh.py', ROOT / 'scripts/audit_train_resets_r10.py',
             ROOT / 'scripts/run_local_sim.sh',
             *[repo / 'third_party/PyRep/pyrep' / path for path in
               ('objects/object.py', 'objects/shape.py', 'backend/sim.py', 'backend/simConst.py', 'const.py')]]
    return {str(path.resolve()): sha(path) for path in paths}


def build_plan(collection_plan, output):
    from collect_multitask_training_r10 import verify_plan as verify_collection
    from audit_train_resets_r10 import native_inputs
    pointer = Path(collection_plan).resolve(); collection = verify_collection(pointer)
    cases = []
    for task in ordinary_tasks():
        row = next(row for row in collection['cases']
                   if row['task'] == task and row['split'] == 'train' and row['replica'] == 0)
        source = Path(row['source']['path']).resolve()
        require(source.name == 'data.h5' and 'train_package_compressed' in source.parts,
                'native component audit cannot read a VAL/TEST source')
        cases.append(dict(task=task, family=TASK_SPECS[task].family, source=row['source'],
                          replica=0, fixed_collection_case_id=row['case_id']))
    inputs = dict(collection['inputs']); inputs.update(native_inputs(collection['protocol']))
    inputs.update(audit_sources(collection)); inputs[str(pointer)] = sha(pointer)
    for case in cases:
        p = str(Path(case['source']['path']).resolve()); inputs[p] = sha(p)
    plan = dict(version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
        split='train', cases=cases, protocol=collection['protocol'],
        collection_plan_path=str(pointer), collection_plan_sha256=sha(pointer), inputs_sha256=inputs,
        component_role_schema=SCHEMA, maximum_workers=1, counted_actions=0,
        teacher_used=False, GPU_used=False, TEST_used=False,
        source_selection='first fixed TRAIN source and replica0 per normal-speed family',
        native_property_setters_used=False, geometry_filter_installed=False,
        flags='raw native renderable, collidable, respondable; physical := collidable OR respondable',
        evidence_scope='normal scene representatives only; high-speed transfer requires separate topology/flag evidence',
        execution_gate='R09 accepted formal400+ten-demo index; unchanged frozen v5 sources; root authorizes diagnostic only')
    plan['plan_sha256'] = digest(plan); write(output, plan); return plan


def verify_plan(pointer):
    from collect_multitask_training_r10 import verify_plan as verify_collection
    from audit_train_resets_r10 import native_inputs
    plan = json.loads(Path(pointer).read_text())
    require(plan.get('version') == VERSION and plan.get('split') == 'train'
            and plan.get('plan_sha256') == digest({k: v for k, v in plan.items() if k != 'plan_sha256'}),
            'component audit plan changed')
    require(plan.get('maximum_workers') == 1 and plan.get('counted_actions') == 0
            and plan.get('teacher_used') is False and plan.get('GPU_used') is False
            and plan.get('TEST_used') is False and plan.get('component_role_schema') == SCHEMA,
            'component audit scope changed')
    collection = verify_collection(plan['collection_plan_path'])
    require(sha(plan['collection_plan_path']) == plan['collection_plan_sha256']
            and collection['protocol'] == plan['protocol'], 'collection source/protocol changed')
    require([row['task'] for row in plan['cases']] == list(ordinary_tasks()), 'seven fixed family cases required')
    for row in plan['cases']:
        first = next(c for c in collection['cases']
                     if c['task'] == row['task'] and c['split'] == 'train' and c['replica'] == 0)
        require(row == dict(task=first['task'], family=TASK_SPECS[first['task']].family,
            source=first['source'], replica=0, fixed_collection_case_id=first['case_id']), 'TRAIN audit source changed')
    required = {**collection['inputs'], **native_inputs(plan['protocol']), **audit_sources(collection),
                str(Path(plan['collection_plan_path']).resolve()): plan['collection_plan_sha256']}
    require(all(plan['inputs_sha256'].get(k) == v for k, v in required.items()), 'audit omitted a frozen/native source')
    for path, expected in plan['inputs_sha256'].items(): require(sha(path) == expected, 'audit source changed: ' + path)
    return plan


def mesh_statistics(vertices, triangles):
    vertices, triangles = np.asarray(vertices, float).reshape(-1, 3), np.asarray(triangles).reshape(-1, 3)
    require(len(vertices) > 0 and len(triangles) > 0 and np.isfinite(vertices).all()
            and np.issubdtype(triangles.dtype, np.integer) and triangles.min() >= 0
            and triangles.max() < len(vertices), 'invalid native mesh in flag audit')
    faces = vertices[triangles]
    area = np.linalg.norm(np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1) / 2
    return dict(vertex_count=len(vertices), triangle_count=len(triangles),
        native_mesh_local_bounds=[vertices.min(0).tolist(), vertices.max(0).tolist()],
        native_mesh_local_extent=np.ptp(vertices, axis=0).tolist(),
        surface_area_m2=float(area.sum()), degenerate_triangle_count=int((area <= 1e-14).sum()),
        original_pointer_safe_reader=True, body_or_world_pose_included=False)


def run_case(pointer, task, output):
    # All permissions/locks are checked before any simulator import.
    from audit_train_resets_r10 import require_completed_delivery
    plan = verify_plan(pointer); gate = require_completed_delivery()
    row = next(row for row in plan['cases'] if row['task'] == task)
    output = Path(output).resolve()
    require(output.is_relative_to(BASE.resolve()) and not output.exists(), 'fresh audit child output required')
    output.mkdir(parents=True); env = None
    try:
        from our_mani.multitask.simulation import AlignedDevelopmentSimulation
        from our_mani.multitask.training_geometry import _owned_shapes
        from our_mani.multitask.safe_native_mesh import read_native_mesh
        env = AlignedDevelopmentSimulation(task, row['source']['path'], source=row['source'], split='train',
            protocol=plan['protocol'], seed=plan['protocol'].get('seed', 0), replica=0,
            servo_lead=plan['protocol'].get('local_servo_lead', 1.4))
        env.reset_development(variation=row['source']['variation'])
        before = (env.physics_time(), float(env.task.t), env.ledger.ticks)
        observation, _ = env.observe()
        roles, owned = _owned_shapes(env, None)
        mask_ids = {}
        for view, values in observation['observation/instance_masks'].items():
            channels = np.rint(np.asarray(values) * 255).astype(np.int64)
            mask_ids[view] = channels[..., 0] + 256 * channels[..., 1] + 65536 * channels[..., 2]
        bodies = {}
        journal = output / 'components.jsonl'
        for role, shapes in sorted(owned.items()):
            records = []
            for shape in shapes:
                flags = read_native_component_flags(shape)
                vertices, triangles, _ = read_native_mesh(shape)
                handle = int(shape.get_handle())
                record = dict(name=shape.get_name(), owned_root_role=role,
                    owned_root_name=roles[role].get_name(), native_handle_metric_only=handle,
                    flags=flags['flags'], native_getter_values=flags['native_getter_values'],
                    native_true_values=flags['native_true_values'], category=flags['category'],
                    mesh=mesh_statistics(vertices, triangles),
                    current_mask_pixels={view: int((ids == handle).sum()) for view, ids in mask_ids.items()})
                records.append(record)
                with journal.open('a') as stream:
                    stream.write(json.dumps(record, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
                print(role, record['name'], record['category'], flush=True)
            bodies[role] = partition_components(records)
        require(before == (env.physics_time(), float(env.task.t), env.ledger.ticks)
                and env.ledger.ticks == 0 and env.time == 0, 'flag/mesh/current-frame audit advanced counted time')
        alignment = env.alignment_receipt(); reset = env.reset_receipt
        env.close(); env = None
        verify_plan(pointer); require(require_completed_delivery() == gate, 'R09 delivery changed during diagnostic')
        report = dict(version=VERSION, complete=True, TRAIN_only=True, teacher_used=False, GPU_used=False,
            formal_metrics_included=False, task=task, source=row['source'], evaluation_process_pid=os.getpid(),
            evaluation_process_ancestry_pids=process_ancestry(),
            plan_sha256=plan['plan_sha256'], release_gate=gate, counted_ticks=0,
            physics_task_clock_unchanged_during_observation_flags_and_mesh_read=True,
            native_property_setters_used=False, root_ownership_implementation='frozen training_geometry._owned_shapes',
            bodies=bodies, metric_only_reset_receipt=reset, alignment_receipt=alignment,
            geometry_filter_installed=False, annotation_changes_written=False,
            flags_are_TRAIN_CAD_annotations_not_policy_current_visibility=True,
            limitation='raw shape flags do not prove current pixel visibility or effective model-level collision; '
                       'single normal TRAIN reset cannot validate all sources or high-speed variants')
        write(output / 'report.json', report); return report
    finally:
        if env is not None: env.close()


def run_queue(pointer, output):
    from audit_train_resets_r10 import require_completed_delivery
    plan = verify_plan(pointer); gate = require_completed_delivery(); output = Path(output).resolve()
    require(output.is_relative_to(BASE.resolve()) and not output.exists(), 'fresh audit queue output required')
    output.mkdir(parents=True); entries = []
    for row in plan['cases']:
        verify_plan(pointer); require(require_completed_delivery() == gate, 'R09 delivery changed')
        task = row['task']; child_out = output / task; log = output / (task + '.native.log')
        command = ['bash', str(ROOT / 'scripts/run_local_sim.sh'), str(Path(__file__).resolve()),
                   '--plan', str(Path(pointer).resolve()), '--child-task', task, '--out', str(child_out)]
        with log.open('wb') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONUNBUFFERED': '1'})
            timed_out = False
            try: code = child.wait(timeout=240)
            except subprocess.TimeoutExpired:
                timed_out = True; os.killpg(child.pid, signal.SIGTERM)
                try: code = child.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); code = child.wait()
        report = child_out / 'report.json'
        data = json.loads(report.read_text()) if report.exists() else None
        okay = code == 0 and data is not None and data.get('complete') is True
        if okay:
            require(child.pid in data['evaluation_process_ancestry_pids'] and data['task'] == task
                    and data['plan_sha256'] == plan['plan_sha256'] and data['counted_ticks'] == 0,
                    'native child receipt identity differs')
        entries.append(dict(task=task, child_process_pid=child.pid,
            native_python_process_pid=data['evaluation_process_pid'] if data else None,
            launch_command=command, returncode=code,
            timed_out=timed_out, complete=okay, report_path=str(report),
            report_sha256=sha(report) if report.exists() else None, log_path=str(log), log_sha256=sha(log),
            counted_as_policy_failure=False))
        write(output / (task + '.parent_receipt.json'), entries[-1])
        print(json.dumps(entries[-1]), flush=True)
    verify_plan(pointer); require(require_completed_delivery() == gate, 'R09 delivery changed')
    require(len({row['child_process_pid'] for row in entries}) == 7, 'expected seven fresh native processes')
    native_pids = [row['native_python_process_pid'] for row in entries if row['complete']]
    require(len(native_pids) == len(set(native_pids)), 'native Python processes must be fresh per TRAIN case')
    result = dict(version=VERSION, complete=all(row['complete'] for row in entries),
        expected=7, completed=sum(row['complete'] for row in entries),
        infrastructure_failures=sum(not row['complete'] for row in entries), entries=entries,
        plan_path=str(Path(pointer).resolve()), plan_sha256=plan['plan_sha256'], release_gate=gate,
        workers=1, GPU_used=False, teacher_used=False, counted_actions=0, formal_metrics_included=False,
        geometry_filter_installed=False, classification_schema=SCHEMA,
        empty_visual_roles=[], empty_collision_roles=[], category_totals={kind: 0 for kind in
            ('visual_only', 'collision_only', 'both', 'neither')})
    for row in entries:
        if not row['complete']: continue
        report = json.loads(Path(row['report_path']).read_text())
        for role, body in report['bodies'].items():
            if body['empty_visual_role']: result['empty_visual_roles'].append([row['task'], role])
            if body['empty_collision_role']: result['empty_collision_roles'].append([row['task'], role])
            for kind, names in body['groups'].items(): result['category_totals'][kind] += len(names)
    write(output / 'report.json', result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--run', action='store_true')
    parser.add_argument('--collection-plan'); parser.add_argument('--plan'); parser.add_argument('--child-task')
    parser.add_argument('--out', required=True); args = parser.parse_args()
    require(sum((args.prepare, args.run, args.child_task is not None)) == 1, 'choose prepare, run, or one child')
    if args.prepare:
        require(args.collection_plan, 'preparation requires frozen collection plan'); value = build_plan(args.collection_plan, args.out)
    else:
        require(args.plan, 'native diagnostic requires its immutable plan')
        value = run_case(args.plan, args.child_task, args.out) if args.child_task else run_queue(args.plan, args.out)
    print(json.dumps(dict(complete=value.get('complete'), plan_sha256=value.get('plan_sha256')), ensure_ascii=False))


if __name__ == '__main__': main()
