"""Check/prepare the single current-network16 pipeline; default starts nothing.

--prepare writes a fresh inventory/plan, with absent execution hooks explicitly
blocked. --execute-stage requires a later Root-finalized plan and external SHA.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask_execution_pipeline_20261104_v1.registry import inventory, TASKS, METHOD
from our_mani.multitask_execution_pipeline_20261104_v1.runner import (
    PLAN_SCHEMA, STAGES, dry_run, run_stage, pointer, write_new, load, verify_sources)

NATIVE_PYTHON = '/home/ubuntu/physmani-conda/envs/physmani5090/bin/python'


def bind_offline_hooks(plan, source_map_path, *, fit=False, fit_entrypoint=None):
    """Bind the real new adapter from an explicit final map, without importing it."""
    value = load(source_map_path)
    sources = value.get('source_sha256', value)
    verify_sources(ROOT, sources)
    if fit and fit_entrypoint is None:
        raise ValueError('Root-selected final fit entrypoint is required; never infer a small/large model')
    entry = (ROOT / fit_entrypoint).resolve() if fit else ROOT / 'scripts/prepare_current_network_multitask_data_20261104_v1.py'
    declared = sources.get(str(entry), sources.get(str(entry.relative_to(ROOT))))
    if declared != pointer(entry)['sha256']:
        raise ValueError('Real data/fit entrypoint missing from its final source map')
    plan['source_sha256'].update(sources)
    evidence = pointer(source_map_path)
    plan['source_sha256'][evidence['path']] = evidence['sha256']
    for stage in (('fit',) if fit else ('data_check', 'preprocess')):
        if fit:
            argv = value.get('fit_argv')
            if not isinstance(argv, list) or not argv or any(not isinstance(v, str) for v in argv):
                raise ValueError('Final full fit_argv, including required Root GPU-preflight arguments, must be supplied explicitly')
        else:
            argv = [NATIVE_PYTHON, '-B', str(entry),
                    '--check' if stage == 'data_check' else '--preprocess',
                    '--manifest', '{split_manifest}', '--output', '{output}']
        plan['hooks'][stage] = dict(method=METHOD, stage=stage, argv=argv,
            entrypoint=pointer(entry), source_sha256=sources,
            resource_lock_owner='pipeline', resource='training' if fit else 'cpu',
            timeout_seconds=86400, final_source_map=evidence,
            current_network_end_to_end_ready=False,
            fitted_component='semantic_task_e8_only' if fit else None)
        if fit:
            plan['hooks'][stage].update(
                fit_filter=dict(split='train', role='fit', offline_eligible=True),
                selection_uses_optimizer=False, TEST_payload_decode_allowed=False)
    if not fit:
        plan['real_data_adapter']['binding_status'] = 'BOUND_REAL_HDF_ADAPTER'
        plan['real_data_adapter']['final_source_map'] = evidence


def prepare(output, split_manifest=None, data_source_map=None, fit_source_map=None, fit_entrypoint=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Fresh preparation directory required')
    inspected = inventory(ROOT)
    sources = {}
    for relative in (
        'our_mani/multitask_execution_pipeline_20261104_v1/__init__.py',
        'our_mani/multitask_execution_pipeline_20261104_v1/registry.py',
        'our_mani/multitask_execution_pipeline_20261104_v1/native_adapter.py',
        'our_mani/multitask_execution_pipeline_20261104_v1/runner.py',
        'scripts/run_multitask_execution_pipeline_20261104_v1.py',
    ):
        record = pointer(ROOT / relative)
        sources[record['path']] = record['sha256']
    for task in inspected['tasks']:
        record = task['native_source']
        sources[record['path']] = record['sha256']
        sources.update(task['native_condition_sources'])
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'task_inventory.json', inspected)
    plan = dict(schema=PLAN_SCHEMA, method=METHOD, tasks=list(TASKS),
        preparation_only=True, execution_authorized=False,
        inventory=pointer(output / 'task_inventory.json'),
        split_manifest=None if split_manifest is None else pointer(split_manifest),
        source_sha256=sources, native_success_authority='task.success()',
        hooks={s: None for s in STAGES if s != 'freeze'},
        task_adapter_readiness={t: dict(public_observation=False, task_phase_controller=False,
            native_reset=False, current_network_not_historical_R10=True) for t in TASKS},
        stage_contract=dict(order=list(STAGES), fit_uses_only_eligible_TRAIN=True,
            fixed_VAL_all_cases_before_freeze=True, immutable_same_model_source_into_TEST=True,
            TEST_metadata_false_eligibility_means_no_offline_decode_not_case_deletion=True,
            task_False_is_an_outcome_infrastructure_failure_stops=True,
            one_attempt_TEST_claim_before_process=True, no_silent_retry=True),
        real_data_adapter=dict(
            package='our_mani.multitask_data_preparation_20261104_v1',
            entrypoint='scripts/prepare_current_network_multitask_data_20261104_v1.py',
            data_check_argv=['--check', '--manifest', '{split_manifest}', '--output', '{output}'],
            preprocess_argv=['--preprocess', '--manifest', '{split_manifest}', '--output', '{output}'],
            binding_status='PENDING_FINAL_SOURCE_AND_RECEIPT_API'),
        legacy_reference_code=dict(
            task_spec='our_mani/multitask/specs.py (14 non-Cube task contracts)',
            task_native_metrics='our_mani/multitask/native_tasks.py:native_result',
            insertion_TRAIN_diagnostic='scripts/run_task_executor_smoke_high_peg_insert_v395_r10.py',
            old_five_method_formal_entry='scripts/evaluate_multitask_case_r10.py',
            directly_executable_as_current_network=False),
        no_current_task_policy_or_training_adapter_fabricated=True)
    if data_source_map is not None:
        if split_manifest is None:
            raise ValueError('Real data hooks require the exact frozen split manifest')
        bind_offline_hooks(plan, data_source_map)
    if fit_source_map is not None:
        if data_source_map is None:
            raise ValueError('Fit hook must follow the real data adapter')
        bind_offline_hooks(plan, fit_source_map, fit=True, fit_entrypoint=fit_entrypoint)
    write_new(output / 'plan.json', plan)
    check = dry_run(ROOT, output / 'plan.json')
    write_new(output / 'dry_run.json', check)
    return dict(plan=pointer(output / 'plan.json'), inventory=pointer(output / 'task_inventory.json'), **check)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--check', action='store_true', help='Explicit default dry-run; never launch')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--split-manifest', type=Path)
    parser.add_argument('--data-source-map', type=Path, help='Final explicit source_sha256 JSON for the real HDF adapter')
    parser.add_argument('--fit-source-map', type=Path, help='Final explicit source_sha256 JSON for semantic-task-e8 component pretraining')
    parser.add_argument('--fit-entrypoint', type=Path, help='Exact Root-selected final fit CLI; no architecture is inferred')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--state-root', type=Path)
    parser.add_argument('--execute-stage', choices=STAGES)
    parser.add_argument('--expected-plan-sha256')
    args = parser.parse_args(argv)
    if args.check and (args.prepare or args.execute_stage):
        parser.error('--check is not an execution/preparation mode')
    if args.prepare:
        if args.output is None or args.plan is not None or args.execute_stage:
            parser.error('--prepare needs a fresh --output only')
        result = prepare(args.output, args.split_manifest, args.data_source_map, args.fit_source_map, args.fit_entrypoint)
    elif args.plan is None:
        result = inventory(ROOT)
    elif args.execute_stage:
        if args.state_root is None or not args.expected_plan_sha256:
            parser.error('Execution needs --state-root and external --expected-plan-sha256')
        result = run_stage(ROOT, args.plan, args.execute_stage, args.state_root, args.expected_plan_sha256)
    else:
        result = dry_run(ROOT, args.plan, args.state_root)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
