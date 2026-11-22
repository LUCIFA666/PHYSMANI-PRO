"""Freeze new recovery sources and declare the exact transition before native.

This prepares metadata only. Existing plans, cases, raw arrays and source files
are never edited. The independent recovery runner performs its own preflight.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import collect_multitask_training_r10 as collection
from our_mani.multitask import acquisition_lineage as lineage


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pointer(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def freeze(original_registration, plan_path, new_collection, output):
    original = json.loads(Path(original_registration).read_text())
    previous = lineage.load_archived_plan(ROOT, original, {})
    plan_path = Path(plan_path).resolve()
    plan = collection.verify_plan(plan_path)
    output = collection.destination(output)
    new_collection = collection.destination(new_collection)
    if output.exists():
        raise ValueError('freeze metadata needs a fresh output namespace')
    output.mkdir(parents=True)
    entries = []
    code = {str(Path(name).resolve()): value for name, value in plan['inputs'].items()
            if Path(name).suffix in ('.py', '.sh')}
    for original_path, expected in sorted(code.items()):
        source = Path(original_path)
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('live executable source changed before freeze')
        if source.is_relative_to(ROOT):
            destination = output/'sources'/source.relative_to(ROOT)
        else:
            destination = output/'sources/external'/expected/source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(raw)
        entries.append(dict(original_path=original_path, **pointer(destination)))
    collection.verify_plan(plan_path)
    for item in entries:
        if sha(ROOT/item['path']) != item['sha256']:
            raise ValueError('frozen executable source differs from the plan')
    snapshot = dict(version=lineage.SOURCE_SCHEMA,
        created_utc=datetime.now(timezone.utc).isoformat(),
        plan=dict(**pointer(plan_path), internal_sha256=plan['plan_sha256']),
        all_plan_inputs_verified_before_and_after=True, source_count=len(entries), entries=entries,
        source_files_modified=False, case_records_or_labels_modified=False,
        frozen_before_native_execution=True, freezer_source=pointer(__file__))
    snapshot_path = output/'source_snapshot.json'
    write(snapshot_path, snapshot)
    recovery_input = json.loads((ROOT/plan['protocol']['recovery_input']['path']).read_text())
    before = {name: row['sha256'] for name, row in previous['sources'].items()}
    changed = [dict(original_path=name, before_sha256=before.get(name), after_sha256=code.get(name),
        reason='Explicit owned recovery modules and newly bound native reset dependencies; existing v11 sources remain unchanged.')
        for name in sorted(set(before)|set(code)) if before.get(name) != code.get(name)]
    if any(row['before_sha256'] is not None for row in changed):
        raise ValueError('this recovery setup must preserve every original v11 executable')
    transitions = []
    for field in ('development_reset_profile', 'RML_factory_hook', 'recovery_input'):
        key = '/protocol/'+field
        transitions.append(dict(json_pointer=key,
            old=lineage.protocol_value(previous['plan'], key), new=lineage.protocol_value(plan, key),
            reason=('Explicit guarded development physical initialization, separate expert reachability; no scene resampling.'
                    if field == 'development_reset_profile' else
                    'Explicit owned previous-q reset in virtual RML probes; native real path execution remains unchanged.'
                    if field == 'RML_factory_hook' else
                    'Same-case infrastructure-only authorization and preserved raw predecessor receipts.'),
            evidence=plan['protocol'][field]))
    declaration = dict(schema='r10_declared_infrastructure_only_acquisition_transition_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), from_plan_sha256=previous['plan']['plan_sha256'],
        to_plan_sha256=plan['plan_sha256'], preserves_case_seed_variation_geometry_labels_budget=True,
        allowed_recovery_case_ids=recovery_input['case_ids'], source_changes=changed,
        protocol_changes=transitions,
        reason='Recover only fixed prior infrastructure failures; preserve original valid cases and all task outcomes. Development reset protocol change is explicit; no physical-equivalence or policy-benefit claim.')
    declaration_path = output/'transition_declaration.json'
    write(declaration_path, declaration)
    current = dict(plan=pointer(plan_path), source_snapshot=pointer(snapshot_path),
        collection=str(new_collection.relative_to(ROOT)),
        predecessor_plan_sha256=previous['plan']['plan_sha256'],
        infrastructure_change_declaration=pointer(declaration_path))
    registry = dict(schema='r10_explicit_recovery_execution_registration_v1', original=original, recovery=current)
    registry_path = output/'execution_registration.json'
    write(registry_path, registry)
    actual = lineage.load_archived_plan(ROOT, current, {})
    lineage.verify_recovery_transition(ROOT, previous, actual, {})
    return dict(execution_registration=pointer(registry_path), source_count=len(entries),
        declared_source_additions=len(changed), fixed_case_ids=recovery_input['case_ids'], native_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-registration', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.original_registration, args.plan, args.collection, args.out), ensure_ascii=False))


if __name__ == '__main__':
    main()
