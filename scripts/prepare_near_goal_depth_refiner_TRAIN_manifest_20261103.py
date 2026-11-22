"""Aggregate only a complete, actually closed fixed TRAIN pilot64 cohort.

This tool never launches Native, constructs a model, or fits anything. A fresh
output directory receives a manifest only after the frozen loader checks every
shot. Optional oracle scoring is separate from acquisition completeness.
"""
from datetime import datetime
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from our_mani.near_goal_depth_refiner_20261103.data import (
    DepthInfluenceDataset, PILOT_IDS, SHOT_IDS, SCHEMA, LABEL_PROTOCOL,
    checked_pointer, sha256,
)

VERSION = 'near_goal_depth_influence_closed_TRAIN_manifest_builder_20261103_v1'


class ReadinessError(ValueError):
    pass


def _json(path):
    return json.loads(Path(path).read_text())


def _pointer(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha256(path))


def _write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _plan(capture_plan):
    path = Path(capture_plan).resolve(strict=True)
    if any('test' in part.lower() for part in path.parts):
        raise ReadinessError('Only a frozen TRAIN capture plan is accepted')
    plan = _json(path)
    if (plan.get('split') != 'train' or plan.get('scope') != 'pilot64'
            or plan.get('perception_only') is not True
            or plan.get('label_method') != 'counterfactual_current_depth_owner'
            or any(plan.get(key) is not False for key in (
                'Native_started', 'fit_started', 'TEST_read', 'benchmark_episode',
                'execution_transition', 'native_instance_id_masks', 'material_or_alpha_edits'))
            or type(plan.get('render_warmup_rounds')) is not int
            or plan['render_warmup_rounds'] != 12):
        raise ReadinessError('Explicit immutable perception-only TRAIN pilot64 plan required')
    parents = plan.get('parents', [])
    if tuple(p.get('parent_id') for p in parents) != PILOT_IDS:
        raise ReadinessError('Exact eight preassigned parents in fixed order required; no replacements')
    for parent in parents:
        if tuple(s.get('shot_id') for s in parent.get('shots', [])) != SHOT_IDS:
            raise ReadinessError('Every parent requires the preassigned ordered shot00..07')
    partition = checked_pointer(ROOT, plan['partition'])
    return path, plan, partition


def _closed_cohort(capture_root, parents):
    root = Path(capture_root).resolve(strict=True)
    if any('test' in part.lower() for part in root.parts):
        raise ReadinessError('TRAIN capture root required')
    expected = {p['parent_id'] for p in parents}
    directories = {p.name for p in root.iterdir() if p.is_dir()}
    if directories != expected or any((root/name).is_symlink() for name in expected):
        raise ReadinessError('Capture root must contain exactly the fixed eight original case directories')
    exits = {p.name for p in root.glob('*.owner.actual_exit.json')}
    if exits != {name+'.owner.actual_exit.json' for name in expected}:
        raise ReadinessError('All eight exact sibling owner actual-exit files required')
    failures, closed = [], []
    # No NPZ/GT is opened while any member still lacks an actual terminal record.
    for parent in parents:
        parent_id = parent['parent_id']
        case = root/parent_id
        owner_path = root/(parent_id+'.owner.actual_exit.json')
        result_path = case/'result.json'
        try:
            owner, result = _json(owner_path), _json(result_path)
            if (type(owner.get('actual_process_returncode')) is not int
                    or owner['actual_process_returncode'] != 0
                    or owner.get('actual_wait_completed') is not True
                    or owner.get('source_unchanged') is not True or owner.get('stop_reason') is not None
                    or owner.get('parent_id') != parent_id
                    or Path(owner.get('output', '')).resolve() != case
                    or any(result.get(key) is not True for key in ('complete', 'native_closed', 'source_unchanged'))
                    or result.get('parent_id') != parent_id
                    or type(result.get('controlled_physics_ticks')) is not int
                    or result['controlled_physics_ticks'] != 0):
                raise ReadinessError('Actual rc0/wait/nativeclose/samecase/source/zeroTicks gate failed')
            started, finished = owner.get('started_utc'), owner.get('finished_utc')
            if (not isinstance(started, str) or not isinstance(finished, str)
                    or datetime.fromisoformat(finished) < datetime.fromisoformat(started)):
                raise ReadinessError('Actual owner launch/wait interval required')
            closed.append((parent, case, owner_path, owner, result_path, result))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(dict(parent_id=parent_id, error=type(exc).__name__+': '+str(exc)))
    if failures:
        raise ReadinessError(json.dumps(dict(reason='incomplete_or_failed_fixed_cohort', failures=failures)))
    return root, closed


def _owner_launch(capture_root, parent, case, owner, plan_path, plan):
    launch_path = capture_root/(parent['parent_id']+'.owner.json')
    log_path = capture_root/(parent['parent_id']+'.owner.log')
    launch = _json(launch_path)
    argv = launch.get('argv', [])
    if (launch.get('parent_id') != parent['parent_id'] or launch.get('started_utc') != owner['started_utc']
            or type(launch.get('child_pid')) is not int or launch['child_pid'] <= 0
            or launch.get('process_group') != launch['child_pid']
            or launch.get('perception_only') is not True or launch.get('benchmark_episode') is not False
            or launch.get('old_runs_resumed') is not False
            or not isinstance(argv, list) or len(argv) != 8 or argv[0] != 'bash'
            or argv[2::2] != ['--plan', '--parent-id', '--output']
            or Path(argv[3]).resolve() != plan_path or argv[5] != parent['parent_id']
            or Path(argv[7]).resolve() != case):
        raise ReadinessError('Same-case actual owner launch/plan/worker command required')
    shell = Path(argv[1]).resolve()
    if ROOT not in shell.parents:
        raise ReadinessError('Pinned repository capture shell required')
    expected = {
        'scripts/launch_near_goal_perception_capture_20261103.py',
        str(shell.relative_to(ROOT)), launch.get('collector_script'),
    }
    if None in expected:
        raise ReadinessError('Explicit collector source identity required')
    expected_sources = {name: plan['source_sha256'].get(name) for name in expected}
    if any(not isinstance(digest, str) for digest in expected_sources.values()):
        raise ReadinessError('Owner executable sources must be frozen by capture plan')
    launch_sources = launch.get('source_sha256', {})
    plan_keys = [name for name in launch_sources if (ROOT/name).resolve() == plan_path]
    if len(plan_keys) != 1:
        raise ReadinessError('Owner must pin the actual capture plan once')
    expected_sources[plan_keys[0]] = sha256(plan_path)
    if launch_sources != expected_sources:
        raise ReadinessError('Actual owner launch source hashes differ from captured frozen plan')
    if plan.get('collector_script') is not None and launch['collector_script'] != plan['collector_script']:
        raise ReadinessError('Actual selected collector differs from plan')
    return dict(owner_launch=_pointer(launch_path), owner_log=_pointer(log_path))


def _candidate(capture_root, closed, plan_path, plan, partition):
    parents, extra_evidence = [], []
    for parent, case, owner_path, owner, result_path, result in closed:
        extra_evidence.append(_owner_launch(capture_root, parent, case, owner, plan_path, plan))
        frames = result.get('shots', [])
        if (len(frames) != 8 or tuple(f.get('shot_id') for f in frames) != SHOT_IDS
                or len({f.get('shot_id') for f in frames}) != 8):
            raise ReadinessError('Actual result must retain ordered eight unique shots; never dict-deduplicate failures')
        shots = []
        for planned, frame in zip(parent['shots'], frames):
            if (frame.get('scene_id') != parent['parent_id']+'/'+planned['shot_id']
                    or frame.get('condition') != planned.get('condition')
                    or frame.get('condition_detail') != planned.get('condition_detail')
                    or frame.get('independent_scene') is not True
                    or type(frame.get('physics_ticks')) is not int or frame['physics_ticks'] != 0):
                raise ReadinessError('Actual shot scene/condition/independent zero-tick assignment differs from plan')
            pointers = {}
            for manifest_key, frame_key, folder in (('inputs', 'input', 'inputs'),
                                                    ('labels', 'labels', 'labels'),
                                                    ('annotation_metadata', 'annotation_metadata', 'labels')):
                path = checked_pointer(case, frame[frame_key])
                if path.parent != case/folder:
                    raise ReadinessError('Actual inputs/labels must remain original physical siblings')
                pointers[manifest_key] = _pointer(path)
            shots.append(dict(shot_id=planned['shot_id'], scene_id=frame['scene_id'],
                independent_scene=True, physics_ticks=0, condition=planned['condition'],
                condition_detail=planned['condition_detail'], **pointers))
        parents.append(dict(parent_id=parent['parent_id'], seed=parent['seed'], role=parent['role'],
            capture=dict(result=_pointer(result_path), owner_actual_exit=_pointer(owner_path),
                run_inputs=_pointer(case/'run_inputs.json'), source_manifest=_pointer(case/'source_snapshot/manifest.json')),
            initial_public_rgbd=_pointer(case/'initialization/public_rgbd.npz'),
            public_sphere_fit=_pointer(case/'public_initial_sphere.json'), shots=shots))
    return dict(schema=SCHEMA, dataset_split='TRAIN', label_protocol=LABEL_PROTOCOL,
        scope='pilot_smoke', complete=True, all_native_closed=True,
        all_process_exits_observed=True, source_unchanged=True,
        partition=_pointer(partition), capture_plan=_pointer(plan_path), parents=parents), extra_evidence


def build_manifest(capture_plan, capture_root, output, *, scope='pilot_smoke', oracle_check=False):
    """Return paths/receipt. Output is a fresh directory, never a case or file."""
    if scope != 'pilot_smoke' or type(oracle_check) is not bool:
        raise ReadinessError('This builder is fixed to the preassigned pilot_smoke; explicit oracle bool required')
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError('Fresh output directory required; no overwrite/resume')
    plan_path, plan, partition = _plan(capture_plan)
    plan_sha, script_sha = sha256(plan_path), sha256(__file__)
    capture_root, closed = _closed_cohort(capture_root, plan['parents'])
    if capture_root == out or capture_root in out.parents:
        raise ReadinessError('Aggregate output must stay outside the original capture root')
    candidate, extra_evidence = _candidate(capture_root, closed, plan_path, plan, partition)
    out.mkdir(parents=True, exist_ok=False)
    draft, final = out/'.manifest.candidate.json', out/'manifest.json'
    published = False
    try:
        _write_new(draft, candidate)
        dataset = DepthInfluenceDataset(draft)
        if len(dataset) != 64:
            raise ReadinessError('Whole64 original shots required')
        label_counts = {'0': 0, '1': 0, '255': 0}
        for sample in dataset:
            if sample['features'].shape != (4, 9, 64, 64):
                raise ReadinessError('Frozen public crop schema changed')
            for key in label_counts:
                label_counts[key] += int((sample['semantic'] == int(key)).sum())
        oracle = None
        if oracle_check:
            from our_mani.near_goal_depth_refiner_20261103.evaluation import oracle_diagnostic
            oracle = oracle_diagnostic(dataset)
            _write_new(out/'oracle.json', oracle)
        dataset.revalidate()
        for evidence in extra_evidence:
            for pointer in evidence.values():
                if sha256(pointer['path']) != pointer['sha256']:
                    raise ReadinessError('Actual owner launch/log evidence changed during validation')
        if sha256(plan_path) != plan_sha or sha256(__file__) != script_sha:
            raise ReadinessError('Builder or capture plan changed during validation')
        draft_sha = sha256(draft)
        # Atomic exclusive publication: link fails if any final file already exists.
        os.link(draft, final)
        published = True
        draft.unlink()
        if sha256(final) != draft_sha:
            raise ReadinessError('Published manifest differs from fully checked candidate')
        receipt = dict(version=VERSION, complete=True, structural_check_complete=True,
            capture_plan=_pointer(plan_path), capture_root=str(capture_root),
            manifest=_pointer(final), scope=scope, parent_order=list(PILOT_IDS),
            parent_count=8, shot_count=64, role_shots=dict(fit=48, selection=8, calibration=8),
            builder=dict(path=str(Path(__file__).resolve()), sha256=script_sha),
            frozen_loader=dict(path=str(Path(sys.modules[DepthInfluenceDataset.__module__].__file__).resolve()),
                sha256=sha256(sys.modules[DepthInfluenceDataset.__module__].__file__)),
            original_owner_evidence=extra_evidence, crop_label_counts=label_counts,
            oracle_requested=oracle_check,
            oracle_ready_for_fit=None if oracle is None else oracle['ready_for_segmentation_fit'],
            pilot_never_runtime_qualified=True, model_or_threshold_qualified=False,
            Native_started=False, fit_started=False, TEST_read=False)
        _write_new(out/'receipt.json', receipt)
        return receipt
    except BaseException as exc:
        if published and final.is_file():
            final.rename(out/'manifest.invalidated.json')
            published = False
        _write_new(out/'failure.json', dict(version=VERSION, complete=False, manifest_published=published,
            error_type=type(exc).__name__, error=str(exc), Native_started=False, fit_started=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-plan', type=Path, required=True)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh aggregate output directory')
    parser.add_argument('--scope', choices=('pilot_smoke',), default='pilot_smoke')
    parser.add_argument('--oracle-check', action='store_true')
    args = parser.parse_args(argv)
    try:
        receipt = build_manifest(args.capture_plan, args.capture_root, args.output,
            scope=args.scope, oracle_check=args.oracle_check)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps(dict(complete=False, error_type=type(exc).__name__, error=str(exc),
            Native_started=False, fit_started=False)), file=sys.stderr, flush=True)
        return 2
    print(json.dumps(receipt, indent=2, allow_nan=False))
    return 3 if receipt['oracle_ready_for_fit'] is False else 0


if __name__ == '__main__':
    raise SystemExit(main())
