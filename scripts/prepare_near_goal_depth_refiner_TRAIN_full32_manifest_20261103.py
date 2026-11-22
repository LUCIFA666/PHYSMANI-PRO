"""Aggregate one new full32 TRAIN capture plan: exact32 parents/256 shots.

Reuse frozen pilot CODE for generic owner/archive checks, never pilot DATA.
All32 cases, including the8 reused q-source identities, must have been captured
again under this one new full32 plan. This tool never captures or fits anything.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.near_goal_depth_refiner_20261103.data import (
    DepthInfluenceDataset, PARTITION_IDS, SHOT_IDS, checked_pointer, sha256,
)

VERSION = 'near_goal_depth_influence_new_full32_TRAIN_manifest_builder_20261103_v1'
PLAN_SCHEMA = 'P3_near_goal_depth_owner_full32_plan_20261103_v1'
FULL_REGISTRY_SHA256 = '57106527829fa9c23d796a0e6e5ceb265ada804fcc3dd48cdc6f460af669c317'
PILOT_CODE = ROOT/'scripts/prepare_near_goal_depth_refiner_TRAIN_manifest_20261103.py'
PILOT_CODE_SHA256 = '231782b9b1493c39acaa8aab77c8e038e471cef759a4133dab5d4970f95ae98e'
OCTANTS = ((1, 1, 1), (-1, -1, -1), (1, 1, -1), (-1, -1, 1),
           (1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1))


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


def _frozen_code():
    if sha256(PILOT_CODE) != PILOT_CODE_SHA256:
        raise ReadinessError('Frozen generic owner/archive helper source changed')
    spec = importlib.util.spec_from_file_location('frozen_TRAIN_manifest_helpers_231782b9', PILOT_CODE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan(capture_plan):
    path = Path(capture_plan).resolve(strict=True)
    if any('test' in part.lower() for part in path.parts):
        raise ReadinessError('New explicit TRAIN full32 plan required')
    plan = _json(path)
    if (plan.get('schema') != PLAN_SCHEMA or plan.get('scope') != 'full32'
            or plan.get('split') != 'train' or plan.get('perception_only') is not True
            or plan.get('label_method') != 'counterfactual_current_depth_owner'
            or any(plan.get(key) is not False for key in (
                'Native_started', 'fit_started', 'TEST_read', 'benchmark_episode',
                'execution_transition', 'native_instance_id_masks', 'material_or_alpha_edits'))
            or type(plan.get('render_warmup_rounds')) is not int or plan['render_warmup_rounds'] != 12):
        raise ReadinessError('One new frozen full32/256 TRAIN protocol only; pilot plan is never accepted')
    parents = plan.get('parents', [])
    if tuple(p.get('parent_id') for p in parents) != PARTITION_IDS:
        raise ReadinessError('Exact ordered32 preassigned parents required, without substitution or partial cohort')
    partition_path = checked_pointer(ROOT, plan['partition'])
    partition = _json(partition_path)
    declared = partition.get('parents', [])
    if tuple(p.get('parent_id') for p in declared) != PARTITION_IDS:
        raise ReadinessError('Whole ordered32-parent preassigned partition required')
    registry_path = checked_pointer(ROOT, plan['TRAIN_q_registry'])
    if sha256(registry_path) != FULL_REGISTRY_SHA256:
        raise ReadinessError('The preassigned metadata-first full32 q registry SHA is fixed')
    registry = _json(registry_path)
    rows = registry.get('rows', [])
    if len(rows) != 32 or registry.get('pilot_rows_exactly_preserved') is not True:
        raise ReadinessError('All32 fixed q sources with original8 bindings preserved required')
    groups = set()
    for i, (parent, original, source) in enumerate(zip(parents, declared, rows)):
        seed = 261103600+i
        role = 'fit' if i < 24 else ('selection' if i < 28 else 'calibration')
        if (parent.get('seed') != seed or parent.get('role') != role
                or parent.get('fold') != role or original.get('seed') != seed
                or original.get('role') != role or parent.get('TRAIN_q_source') != source
                or source.get('parent_seed') != seed or source.get('parent_role') != role
                or tuple(s.get('shot_id') for s in parent.get('shots', [])) != SHOT_IDS):
            raise ReadinessError('Preassigned32 seed/24-4-4/wholeqsource/ordered8shot identities must match')
        ordinal = parent.get('near_direction_ordinal')
        direction = np.asarray(parent.get('near_direction_unit_world'), float)
        if (type(ordinal) is not int or ordinal != i % 8 or direction.shape != (3,)
                or not np.isfinite(direction).all()
                or not np.allclose(direction, np.asarray(OCTANTS[ordinal])/np.sqrt(3.), rtol=0, atol=1e-12)):
            raise ReadinessError('Predeclared balanced octant direction must follow parent ordinal%8')
        for shot in parent['shots']:
            if (shot.get('near_direction_ordinal') != ordinal
                    or shot.get('near_direction_unit_world') != parent['near_direction_unit_world']):
                raise ReadinessError('Every planned shot must retain exact parent direction metadata')
        group = (str(Path(source['file']['path']).resolve()), source['file']['sha256'],
                 source['dataset'], source['index'])
        if group in groups:
            raise ReadinessError('Full32 registry requires32 independent source groups, never cross-fold or within-fold reuse')
        groups.add(group)
    return path, plan, partition_path, registry_path


def _direction_evidence(closed, candidate, extra_evidence):
    """Direction is audited plan/label metadata only, never a public feature."""
    balance = {str(i): 0 for i in range(8)}
    for i, ((parent, case, _, _, _, result), assembled) in enumerate(zip(closed, candidate['parents'])):
        expected = {key: parent[key] for key in ('near_direction_ordinal', 'near_direction_unit_world')}
        materialized_path = case/'labels/materialized_shot_plan.json'
        if sha256(materialized_path) != result.get('materialized_shot_plan_sha256'):
            raise ReadinessError('Original pre-image materialized shot plan SHA required')
        materialized = _json(materialized_path)
        shots = materialized.get('shots', [])
        if tuple(shot.get('shot_id') for shot in shots) != SHOT_IDS:
            raise ReadinessError('Materialized direction evidence must retain all8 ordered shots')
        for frame, planned, numerical, output in zip(result['shots'], parent['shots'], shots, assembled['shots']):
            metadata_path = checked_pointer(case, frame['annotation_metadata'])
            metadata = _json(metadata_path)
            for source in (planned, numerical, frame, metadata):
                if any(source.get(key) != value for key, value in expected.items()):
                    raise ReadinessError('Actual frame/metadata/materialization direction differs from frozen parent plan')
            output.update(expected)
        assembled.update(expected)
        extra_evidence[i]['materialized_shot_plan'] = _pointer(materialized_path)
        balance[str(parent['near_direction_ordinal'])] += 1
    if any(count != 4 for count in balance.values()):
        raise ReadinessError('All8 fixed directions require exactly4 parents each')
    return balance


def build_manifest(capture_plan, capture_root, output, *, scope='full32', oracle_check=False):
    """Fresh output directory. No old pilot data is searched or accepted."""
    if scope != 'full32' or type(oracle_check) is not bool:
        raise ReadinessError('Explicit full32 scope/oracle bool required; no pilot alias')
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError('Fresh output only, no overwrite/resume')
    plan_path, plan, partition, registry = _plan(capture_plan)
    script_sha, plan_sha = sha256(__file__), sha256(plan_path)
    helpers = _frozen_code()
    try:
        capture_root, closed = helpers._closed_cohort(capture_root, plan['parents'])
    except (ValueError, OSError) as exc:
        raise ReadinessError('Full32 requires all32 actually closed under one new plan: '+str(exc)) from exc
    if capture_root == out or capture_root in out.parents:
        raise ReadinessError('Aggregate output must stay outside the immutable new capture root')
    candidate, extra_evidence = helpers._candidate(capture_root, closed, plan_path, plan, partition)
    if len(candidate['parents']) != 32 or sum(len(p['shots']) for p in candidate['parents']) != 256:
        raise ReadinessError('Complete newly captured32x8 cohort required')
    direction_balance = _direction_evidence(closed, candidate, extra_evidence)
    # Generic frozen helper has a pilot scope literal; change only this explicit
    # scope after the new full32 plan/cohort checks. All pointer checks are shared.
    candidate['scope'] = 'full32'
    out.mkdir(parents=True, exist_ok=False)
    draft, final = out/'.manifest.candidate.json', out/'manifest.json'
    published = False
    try:
        _write_new(draft, candidate)
        dataset = DepthInfluenceDataset(draft)
        if dataset.scope != 'full32' or len(dataset) != 256:
            raise ReadinessError('Frozen loader must validate all256 samples as full32')
        labels = {'0': 0, '1': 0, '255': 0}
        role_shots = dict(fit=0, selection=0, calib=0)
        for sample in dataset:
            if sample['features'].shape != (4, 9, 64, 64):
                raise ReadinessError('Frozen current-public crop schema changed')
            role_shots[sample['role']] += 1
            for key in labels:
                labels[key] += int((sample['semantic'] == int(key)).sum())
        if role_shots != dict(fit=192, selection=32, calib=32):
            raise ReadinessError('Exact24/4/4 whole-parent role allocation required')
        oracle = None
        if oracle_check:
            from our_mani.near_goal_depth_refiner_20261103.evaluation import oracle_diagnostic
            oracle = oracle_diagnostic(dataset)
            _write_new(out/'oracle.json', oracle)
        dataset.revalidate()
        for evidence in extra_evidence:
            for pointer in evidence.values():
                if sha256(pointer['path']) != pointer['sha256']:
                    raise ReadinessError('Actual owner launch/log evidence changed')
        if (sha256(plan_path) != plan_sha or sha256(__file__) != script_sha
                or sha256(PILOT_CODE) != PILOT_CODE_SHA256 or sha256(registry) != FULL_REGISTRY_SHA256):
            raise ReadinessError('New plan/builder/frozen helper/registry changed during validation')
        expected_sha = sha256(draft)
        os.link(draft, final)
        published = True
        draft.unlink()
        if sha256(final) != expected_sha:
            raise ReadinessError('Published manifest differs from validated256-shot candidate')
        receipt = dict(version=VERSION, complete=True, structural_check_complete=True,
            capture_plan=_pointer(plan_path), capture_root=str(capture_root),
            manifest=_pointer(final), full32_q_registry=_pointer(registry), partition=_pointer(partition),
            scope='full32', parent_order=list(PARTITION_IDS), parent_count=32, shot_count=256,
            parent_roles=dict(fit=24, selection=4, calibration=4), role_shots=role_shots,
            predeclared_direction_parent_counts=direction_balance,
            direction_metadata_is_not_model_input=True,
            builder=dict(path=str(Path(__file__).resolve()), sha256=script_sha),
            reused_frozen_CODE_only=_pointer(PILOT_CODE), old_pilot_DATA_reused=False,
            all32_cases_bound_to_one_NEW_plan_sha256=plan_sha,
            original_owner_evidence=extra_evidence, crop_label_counts=labels,
            oracle_requested=oracle_check,
            oracle_ready_for_fit=None if oracle is None else oracle['ready_for_segmentation_fit'],
            acquisition_complete_is_not_model_qualification=True, model_or_threshold_qualified=False,
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
    parser.add_argument('--output', type=Path, required=True, help='Fresh directory outside capture root')
    parser.add_argument('--scope', choices=('full32',), default='full32')
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
