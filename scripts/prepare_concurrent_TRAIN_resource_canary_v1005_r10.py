"""Prepare one fresh12-TRAIN resource canary only after956 naturally drains.

No action occurs without --prepare. This creates only separately frozen small
controls, never starts a child, reads raw/model bytes, or changes formal policy.
The original745 first four TRAIN routes are reused verbatim except fresh --out.
"""
import argparse
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_concurrent_TRAIN_resource_canary_v1005_r10 as runner

SOURCE = 'scripts/prepare_concurrent_TRAIN_resource_canary_v1005_r10.py'
BASE_CONTROL_SHA = '51836554f983428646f8a10914239853bbc9417036c7e73f5dda5b72ba7ca25d'


def prepare(authorization, authorization_sha256):
    require = runner.require
    event = threading.Event()
    require(not runner.stopped(event), 'STOP before preparing resource qualification')
    require(all(not (ROOT / p).exists() for p in (runner.SETUP, runner.RUNTIME, runner.OUTPUT,
        'runtime/framework/concurrent_TRAIN_resource_audit_v1005',
        'runtime/framework/concurrent_TRAIN_resource_qualification_v1005.json')),
        'all1005 namespaces must be fresh; never overwrite an attempt')
    authorization_pointer = dict(path=authorization, sha256=authorization_sha256)
    require(runner.valid_pointer(authorization_pointer), 'external authorization SHA/path required')
    base_pointer = dict(path=runner.BASE_CONTROL, sha256=BASE_CONTROL_SHA)
    base = runner.read_pointer(base_pointer)
    standing = runner.read_pointer(runner.point(runner.STANDING))
    auth = runner.read_pointer(authorization_pointer)
    require(standing.get('continue_until_user_stop') is True and standing.get('user_stop_received') is False
        and auth.get('user_authorized_concurrency_change') is True,
        'real continuing authorization must not be replaced by a synthetic user event')
    implementation = runner.implementation_from_sources(
        {name: runner.point(name) for name in base['execution_implementation']['sources']})
    require(implementation == base['execution_implementation'], 'original901/900/751/742/743/744 changed')
    inventory_pointer = runner.point(runner.INVENTORY)
    require(inventory_pointer['sha256'] == runner.INVENTORY_SHA, 'exact previously archived790 TRAIN inventory required')
    inventory = runner.read_pointer(inventory_pointer)
    train_plan_pointer, routes_pointer = runner.point(runner.TRAIN_PLAN), runner.point(runner.TRAIN_ROUTES)
    original_plan, routes = runner.read_pointer(train_plan_pointer), runner.read_pointer(routes_pointer)
    require(inventory.get('plan') == train_plan_pointer and inventory.get('routes') == routes_pointer,
            'original790 archived plan/routes pointers must still match')
    selected = [r for r in inventory.get('cases', []) if r.get('task') == runner.TASK
                and r.get('raw_episode_id') in runner.RAWS and r.get('method') in runner.METHODS]
    require(len(selected) == 12, 'fixed original745 twelve TRAIN records required')
    cpu_pointers = [r['provenance']['CPU_receipt'] for r in selected]
    cpu_receipts = {p['path']: runner.read_pointer(p) for p in cpu_pointers}
    rml = {r.get('common_RML_profile_sha256') for r in routes.get('routes', []) if r.get('task') == runner.TASK}
    require(len(rml) == 1 and isinstance(next(iter(rml)), str), 'one unchanged common TRAIN RML profile required')
    bootstrap_control = runner.SETUP + '/training_bootstrap_control.json'
    bootstrap_lock = runner.SETUP + '/training_bootstrap_inputs.json'
    # Future pointers are filled from the exact serialized bytes, without writing
    # a half-setup before the exclusive resource and namespace gates have passed.
    def encode(value):
        return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    source_files = {str(ROOT / name): runner.point(name)['sha256'] for name in runner.SOURCES}
    train_control = dict(schema='r10_scoped_frozen_lookup_bootstrap_v901', execution_enabled=True,
        entry='train', scientific_policy_changed=False, models_or_training_changed=False,
        TRAIN_cases=[[runner.TASK, raw, method] for raw in runner.RAWS for method in runner.METHODS],
        argv_bindings={'--split': 'train', '--manifest': runner.MANIFEST, '--frozen-inputs': runner.BASE_LOCK})
    train_control_bytes = encode(train_control)
    bootstrap = dict(control=dict(path=bootstrap_control, sha256=runner.sha(train_control_bytes)))
    batch_files = {str(ROOT / name): p['sha256'] for name, p in implementation['sources'].items()}
    batch_files[str(ROOT / bootstrap_control)] = bootstrap['control']['sha256']
    train_lock = dict(schema='r10_scoped_frozen_lookup_bootstrap_inputs_v901', execution_enabled=True, files=batch_files)
    train_lock_bytes = encode(train_lock)
    bootstrap['lock'] = dict(path=bootstrap_lock, sha256=runner.sha(train_lock_bytes))
    control = dict(schema=runner.CONTROL_SCHEMA, execution_enabled=True, created_utc=runner.utc(), split='train',
        task=runner.TASK, raw_episode_ids=list(runner.RAWS), resource_profile=runner.PROFILE,
        training_weights_updated=False, formal_score_contribution=False, TEST_read=False,
        common_RML_profile_sha256=next(iter(rml)), resource_qualifier=runner.point(runner.AUDITOR),
        inventory=inventory_pointer, original_TRAIN_plan=train_plan_pointer, original_TRAIN_routes=routes_pointer,
        base_formal_control=base_pointer, standing_continuation_record=runner.point(runner.STANDING),
        concurrency_authorization_record=authorization_pointer, training_bootstrap=bootstrap,
        execution_implementation=implementation, previous_CPU_receipts=cpu_pointers,
        policy_sha256=base['policy_sha256'], author_outcome_protocol=base['author_outcome_protocol'],
        **{key: base[key] for key in ('environment_manifest', 'environment_wrapper', 'environment_namespace_proof')},
        **{key: runner.point(path) for key, path in runner.PREVIOUS_PATHS.items()})
    runner.selected_rows(inventory, cpu_receipts, control, original_plan)
    # This gate checks exact956/958 PID/start exit, three shared nonblocking
    # locks, complete host-visible native inventory, current820 namespace+STOP.
    with runner.resource_guard(control, event):
        control_bytes = encode(control)
        files = dict(source_files)
        pointers = [inventory_pointer, train_plan_pointer, routes_pointer, base_pointer,
                    control['standing_continuation_record'], authorization_pointer, *cpu_pointers,
                    *(control[key] for key in runner.PREVIOUS_PATHS),
                    *(control[key] for key in ('environment_manifest', 'environment_wrapper', 'environment_namespace_proof'))]
        for p in pointers:
            actual = runner.point(p['path'])
            require(actual == p, 'small evidence changed while preparing1005')
            key = str(ROOT / p['path'])
            require(key not in files or files[key] == p['sha256'], 'conflicting small source/evidence SHA')
            files[key] = p['sha256']
        files[str(ROOT / runner.CONTROL)] = runner.sha(control_bytes)
        files[str(ROOT / bootstrap_control)] = bootstrap['control']['sha256']
        files[str(ROOT / bootstrap_lock)] = bootstrap['lock']['sha256']
        # Pin every just-read source again before committing any setup.
        require(all(runner.point(name)['sha256'] == source_files[str(ROOT/name)] for name in runner.SOURCES),
                'source changed during preparation')
        require(not runner.stopped(event), 'STOP before freezing the fresh setup')
        setup = ROOT / runner.SETUP
        setup.mkdir(parents=True, exist_ok=False)
        for path, payload in ((bootstrap_control, train_control_bytes), (bootstrap_lock, train_lock_bytes),
                              (runner.CONTROL, control_bytes)):
            with (ROOT/path).open('xb') as stream:
                stream.write(payload)
        runner.atomic_new(ROOT / runner.INPUT_LOCK, dict(schema=runner.LOCK_SCHEMA, execution_enabled=True, files=files))
        _, binding = runner.load_control(runner.CONTROL, runner.INPUT_LOCK, runner.point(runner.INPUT_LOCK)['sha256'])
    return dict(complete=True, control=binding['control'], lock=binding['lock'], case_count=12,
                native_started=False, GPU_started=False, resource_profile=runner.PROFILE)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--authorization')
    parser.add_argument('--authorization-sha256')
    args = parser.parse_args(argv)
    if not args.prepare:
        return dict(enabled=False, native_started=False, GPU_started=False)
    result = prepare(args.authorization, args.authorization_sha256)
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    main()
