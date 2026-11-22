"""Compile only the complete fixed CF40 cohort, never Native or neural inference.

The whole40 closed/source/archive gate precedes any label metadata or NPZ getter.
Only the416 fit/selection samples are decoded for structural supervision QA;
the104 reserved samples remain metadata/SHA-only until post-selection evaluation.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.current_cube_rgbd_centre_20261103_v1 import data

QUEUE = ROOT/'scripts/run_current_cube_dual_crop_capture_queue_20261104_v1.py'
FAMILY = ROOT/'runtime/framework/near_goal_perception_20261103'


def read(path):
    return json.loads(Path(path).read_text())


def pointer(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=data.sha256(path))


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def plan_metadata(path):
    path = Path(path).resolve(strict=True)
    if ROOT/'config/framework' not in path.parents:
        raise ValueError('Versioned ROOT capture plan required')
    plan = read(path)
    expected = data.pilot_metadata()
    if (plan.get('schema') != data.PLAN_SCHEMA or plan.get('scope') != data.SCOPE
            or plan.get('split') != 'train'
            or tuple(p['parent_id'] for p in plan.get('parents', [])) != data.PARTITION_IDS
            or any(p['role'] != old['role'] or p['seed'] != old['seed']
                or tuple(s['shot_id'] for s in p['shots']) != data.SHOT_IDS
                for p, old in zip(plan['parents'], expected))):
        raise ValueError('Exact fixed40/520 source-role/recipe identity required')
    data.checked_pointer(ROOT, plan['metadata_preparation_manifest'])
    data.checked_pointer(ROOT, plan['source_parent_registry'])
    data.checked_pointer(ROOT, plan['TRAIN_q_registry'])
    if (plan['metadata_preparation_manifest']['sha256'] != data.PREPARATION_SHA
            or plan['source_parent_registry']['sha256'] != data.SHARED_REGISTRY_SHA
            or plan['TRAIN_q_registry']['sha256'] != data.REGISTRY_SHA):
        raise ValueError('Immutable source-parent partition pointers required')
    return path, plan


def audit_all_closed(capture_plan, capture_root):
    path, plan = plan_metadata(capture_plan)
    root = Path(capture_root).resolve(strict=True)
    if FAMILY.resolve() not in root.parents:
        raise ValueError('Actual perception runtime cohort root required')
    if plan.get('preparation_only') is not False:
        raise ValueError('An unadmitted preparation plan cannot describe captured data')
    # Only names/status/source/archives are read in this phase, not annotations.
    if {p.name for p in root.iterdir() if p.is_dir()} != set(data.PARTITION_IDS):
        raise ValueError('Whole exact40 directories required; no prefix, extra parent or subset')
    source_key = str(QUEUE.relative_to(ROOT))
    if plan['source_sha256'].get(source_key) != data.sha256(QUEUE):
        raise ValueError('Original frozen queue closure auditor required')
    spec = importlib.util.spec_from_file_location('frozen_dual_CF_queue', QUEUE)
    queue = importlib.util.module_from_spec(spec); spec.loader.exec_module(queue)
    queue.verify_plan(path, data.sha256(path))
    rows = [queue.audit_closed(root, parent, plan, data.sha256(path)) for parent in plan['parents']]
    if len(rows) != 40:
        raise ValueError('All40 actualclosed0 required before any label access')
    return path, plan, root, rows


def candidate_manifest(path, plan, root):
    expected = {r['parent_id']: r for r in data.pilot_metadata()}
    parents = []
    for parent in plan['parents']:
        pid = parent['parent_id']; case = root/pid; result = read(case/'result.json')
        shots = []
        for frame in result['shots']:
            shots.append(dict(scene_id=frame['scene_id'], shot_id=frame['shot_id'],
                condition=frame['condition'], condition_detail=frame['condition_detail'],
                position_band=frame['position_band'],
                inputs=pointer(case/frame['input']['path']),
                labels=pointer(case/frame['labels']['path']),
                annotation_metadata=pointer(case/frame['annotation_metadata']['path'])))
        parents.append(dict(parent_id=pid, seed=parent['seed'], role=parent['role'],
            source_parent_group=expected[pid]['source_parent_group'],
            capture=dict(result=pointer(case/'result.json'),
                owner_actual_exit=pointer(root/(pid+'.owner.actual_exit.json')),
                run_inputs=pointer(case/'run_inputs.json'),
                source_manifest=pointer(case/'source_snapshot/manifest.json')),
            initial_public_rgbd=pointer(case/'initialization/public_rgbd.npz'),
            public_sphere_fit=pointer(case/'public_initial_sphere.json'), shots=shots))
    return dict(schema=data.SCHEMA, scope=data.SCOPE, dataset_split='TRAIN',
        label_protocol=data.LABEL_PROTOCOL, complete=True, all_native_closed=True,
        all_process_exits_observed=True, source_unchanged=True,
        preparation_manifest=pointer(ROOT/plan['metadata_preparation_manifest']['path']),
        shared_registry=pointer(ROOT/plan['source_parent_registry']['path']),
        capture_plan=pointer(path), parents=parents,
        model_or_measurement_qualified=False, formal_task_or_grasp_qualification=False,
        reserved104_labels_not_decoded_by_compiler=True,
        compiler=dict(path=str(Path(__file__).resolve()), sha256=data.sha256(__file__)))


def build_manifest(capture_plan, capture_root, output):
    output = Path(output).resolve()
    if (output.exists() or ROOT/'artifacts' not in output.parents
            or Path(capture_root).resolve() in output.parents):
        raise ValueError('Fresh ROOT/artifacts output outside capture cohort required')
    # Do not create an apparently usable aggregate until the whole closure passes.
    path, plan, root, audit = audit_all_closed(capture_plan, capture_root)
    output.mkdir(parents=True, exist_ok=False)
    published = False
    try:
        manifest = candidate_manifest(path, plan, root)
        draft = output/'candidate_manifest.json'
        write(draft, manifest)
        stats = {}; total = 0
        for role in ('fit', 'selection'):
            dataset = data.DualCropCubeDataset(draft, role=role)
            targets, reasons = Counter(), Counter()
            for index in range(len(dataset)):
                sample = dataset[index]; target = sample['targets']
                targets[str(target['presence_target'])] += 1
                reasons[target['reason']] += 1
                if sample['role'] != role or bool(target['centre_valid']) != (target['presence_target'] == 1):
                    raise ValueError('Role or unknown/centre supervision contract failed')
            verification = dataset.revalidate()
            if len(dataset) != 208 or any(r['role'] == 'reserved_fresh_centre_VAL' for r in dataset.records):
                raise ValueError('Only exact208 samples per training role may be decoded')
            stats[role] = dict(samples=len(dataset), availability_targets=dict(targets),
                supervision_reasons=dict(reasons), source_verification=verification)
            total += len(dataset)
        if total != 416:
            raise ValueError('Whole416 fit/selection structural QA required')
        if data.sha256(path) != manifest['capture_plan']['sha256']:
            raise ValueError('Capture plan changed during structural QA')
        # Existing datasets must never be overwritten; same-directory hardlink is atomic.
        final = output/'manifest.json'
        os.link(draft, final)
        published = True
        receipt = dict(schema='dual_crop_CF_pilot40_wholecohort_manifest_compile_receipt_v2',
            complete=True, manifest=pointer(final), capture_plan=pointer(path),
            capture_root=str(root), all40_actualclosed_before_any_label_getter=True,
            audited_parents=audit, structural_QA=stats,
            decoded_fit_samples=208, decoded_selection_samples=208,
            decoded_reserved_samples=0, reserved_metadata_and_hash_only=104,
            current_visibility_not_from_GT_projection=True,
            unknown_not_absent=True, no_qualification_or_accuracy_gate_claim=True,
            created_utc=datetime.now(timezone.utc).isoformat(),
            Native_started=False, NN_calls=0, training_started=False, TEST_read=False)
        write(output/'receipt.json', receipt)
        return receipt
    except BaseException as error:
        if published and not (output/'receipt.json').exists():
            (output/'manifest.json').rename(output/'manifest.invalidated.json')
        write(output/'failure.json', dict(complete=False, error_type=type(error).__name__,
            error=str(error), final_manifest_published=False, original_capture_unchanged=True))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-plan', type=Path, required=True)
    parser.add_argument('--capture-root', type=Path)
    parser.add_argument('--output', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--audit-closed', action='store_true')
    mode.add_argument('--compile', action='store_true')
    args = parser.parse_args(argv)
    if args.compile:
        if args.capture_root is None or args.output is None:
            parser.error('--compile requires --capture-root and fresh --output')
        result = build_manifest(args.capture_plan, args.capture_root, args.output)
    elif args.audit_closed:
        if args.capture_root is None: parser.error('--audit-closed requires --capture-root')
        path, _, root, rows = audit_all_closed(args.capture_plan, args.capture_root)
        result = dict(complete=True, all40_actualclosed=True, parents=rows,
            capture_plan=pointer(path), capture_root=str(root), NPZ_getters=0,
            Native_started=False, NN_calls=0, training_started=False)
    else:
        path, _ = plan_metadata(args.capture_plan)
        result = dict(complete=True, plan=pointer(path), expected_parents=40,
            expected_shots=520, capture_closed_gate_checked=False,
            active_capture_or_labels_or_NPZ_not_read=True, Native_started=False,
            NN_calls=0, training_started=False, execution_ready=False)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
