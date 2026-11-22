"""Complete five-method VAL with a verified independent baseline component.

Four ours methods use the final model config. The baseline's 275 raw cases
keep their actual earlier config, lock and queue. Its complete dependency
projection is revalidated against the final lock; a component alone cannot
release TEST. Original single-queue acceptance remains unchanged.
"""
from collections import defaultdict
from pathlib import Path

from .baseline_component import COMPONENT, validate_baseline_component
from .evaluation import (METHODS, atomic_json, comparison_protocol_sha256, digest,
                         file_sha, validate_completed_case)
from .release_preflight import normalized_lock, require_frozen_coverage_certificate
from .reset_parity import compare_five_way, profile_problems
from .specs import TASK_SPECS
from .trace_evidence import audit_selection_usage
from .validation_release import Proof, _manifest, _plan, _cases, _performance, proof_directory, require

SCHEMA = 'r10_complete_five_method_component_validation_release_v1'
COMPOSITION = 'r10_explicit_four_ours_plus_baseline_VAL_composition_v1'
OURS = METHODS[:-1]


def _composition(proof, value, canonical):
    data = proof.read_pointer(value)
    require(set(data) == {'schema', 'sha256', 'created_utc', 'ours_queue_plan',
        'baseline_early_queue_plan', 'baseline_execution_contract', 'canonical_validation_manifest',
        'expected_source_cases', 'expected_method_cases', 'raw_identities_rewritten'},
        'exact explicit VAL component composition required')
    require(data['schema'] == COMPOSITION and data['sha256'] == digest(
        {key: item for key, item in data.items() if key != 'sha256'})
        and data['expected_source_cases'] == 275 and data['expected_method_cases'] == 1375
        and data['raw_identities_rewritten'] is False, 'VAL composition identity or fixed coverage changed')
    require(_manifest(proof, proof.pointer(data['canonical_validation_manifest'])) == canonical,
        'component composition changed original canonical VAL inventory')
    for name in ('ours_queue_plan', 'baseline_early_queue_plan', 'baseline_execution_contract'):
        proof.pointer(data[name])
    return data


def _baseline_groups(proof, component, canonical):
    """Consume only records freshly verified by the full component validator."""
    require(component.get('schema') == COMPONENT and component.get('complete') is True
        and component.get('baseline_component_reusable') is True and component.get('sources') == 275
        and component.get('raw_identity_rewritten') is False
        and component.get('raw_receipts_rewritten') is False
        and component.get('single_full_config_identity_claimed') is False,
        'complete actual 275-case baseline component required')
    groups, receipts = {}, []
    for item in component.get('completed_cases', []):
        key = (item['task'], item['raw_episode_id'])
        require(key in canonical and key not in groups, 'baseline source duplicated or outside canonical VAL')
        record = proof.read_pointer(item['canonical_receipt'])
        validate_completed_case(record, item['actual_identity'])
        child = proof.read_pointer(item['raw_child_receipt'])
        require(child == {key: value for key, value in record.items() if key != 'parent_case_process_receipt'}
            and record['method'] == 'physmani' and record['split'] == 'validation'
            and item['actual_identity']['source_row_sha256'] == digest(canonical[key]),
            'component cannot rewrite raw baseline identity/source')
        parent = record['parent_case_process_receipt']
        proof.file(parent['log_path'], parent['log_sha256'])
        groups[key] = {'physmani': record}
        receipts.append(dict(task=key[0], parent_raw_episode_id=key[1], method='physmani',
            **item['canonical_receipt'], raw_child_path=item['raw_child_receipt']['path'],
            raw_child_sha256=item['raw_child_receipt']['sha256'],
            identity_sha256=item['actual_identity']['sha256'],
            actual_config=component['actual_early_config'], actual_lock=component['actual_early_lock'],
            actual_queue_plan=component['actual_early_queue_plan']))
    require(len(receipts) == 275 and set(groups) == set(canonical), 'complete baseline275 coverage missing')
    return groups, receipts


def finalize_composite_validation_release(root, composition, coverage_certificate, out, *,
        canonical_validation_manifest='artifacts/framework/protocol/validation_manifest.json',
        experiment_plan='config/framework/multitask_r10_experiment_plan.json'):
    proof = Proof(root)
    out = (proof.root / out).resolve()
    require(out.is_relative_to(proof.root / 'artifacts/framework/multitask_r10') and not out.exists(),
        'fresh desktop composite VAL release path required')
    stage = 'canonical_inventory'
    try:
        canonical = _manifest(proof, canonical_validation_manifest)
        spec = _composition(proof, composition, canonical)
        stage = 'four_ours_plan'
        plan_path = proof.pointer(spec['ours_queue_plan'])
        plan = _plan(proof, plan_path, canonical, methods=OURS)
        config = proof.read_pointer(plan['config']); plan['_config'] = config
        lock = proof.read_pointer(plan['frozen_inputs'])
        files = normalized_lock(proof.root, lock['files']); proof.include(files)
        for field in ('config', 'manifest', 'parent_source', 'child_source'):
            proof.locked(plan[field]['path'], files, plan[field]['sha256'])
        for pointer in [config['context_checkpoint']] + [pointer for task in TASK_SPECS
                for pointer in config['learned_artifacts'][task].values()
                if isinstance(pointer, dict) and 'path' in pointer]:
            proof.locked(pointer['path'], files, pointer['sha256'])
        stage = 'four_ours_actual_cases'
        groups, receipts, status = _cases(proof, plan, expected_cases=1100)
        require(set(groups) == set(canonical) and all(set(rows) == set(OURS) for rows in groups.values()),
            'four ours require 1100 complete original paired VAL executions')
        stage = 'fresh_baseline_component_validation'
        component = validate_baseline_component(proof.root, spec['baseline_early_queue_plan'],
            spec['baseline_execution_contract'], plan['config'], plan['frozen_inputs'],
            spec['canonical_validation_manifest'])
        early_plan = proof.read_pointer(component['actual_early_queue_plan'])
        proof.read_pointer(component['actual_early_config'])
        early_lock = proof.read_pointer(component['actual_early_lock'])
        proof.include(early_lock['files'])
        for field in ('config', 'manifest', 'frozen_inputs', 'parent_source', 'child_source'):
            proof.pointer(early_plan[field])
        baseline, baseline_receipts = _baseline_groups(proof, component, canonical)
        for key in canonical:
            require(not set(groups[key]) & set(baseline[key]), 'duplicate method in VAL component composition')
            groups[key].update(baseline[key])
        receipts += baseline_receipts
        require(len(receipts) == 1375 and all(set(rows) == set(METHODS) for rows in groups.values()),
            'complete five-method1375 paired evidence required')
        stage = 'final_model_coverage'
        coverage = proof.read_pointer(coverage_certificate); proof.include(coverage['input_files_sha256'])
        coverage_receipt = require_frozen_coverage_certificate(proof.root, coverage_certificate, config, proof.files)
        stage = 'TRAIN_frozen_tolerances'
        profile = proof.read_pointer(config['initial_condition_tolerance_profile'])
        proof.locked(config['initial_condition_tolerance_profile']['path'], files,
            config['initial_condition_tolerance_profile']['sha256'])
        require(not profile_problems(profile, evidence_root=proof.root), 'frozen TRAIN tolerance profile invalid')
        proof.pointer(profile['TRAIN_evidence'])
        stage = 'recorded_initial_conditions'
        comparisons = []
        for key, records in groups.items():
            comparison = compare_five_way(records, profile, evidence_root=proof.root, require_frozen=True)
            require(comparison.get('complete') is True and comparison.get('accepted') is True
                and len(comparison.get('checks', [])) == 10, 'recorded initial-condition parity failed: ' + repr(key))
            comparisons.append(dict(task=key[0], parent_raw_episode_id=key[1], **comparison))
        stage = 'selection_trace'; usage = audit_selection_usage(groups)
        stage = 'preregistered_stopping_rule'
        experiment = proof.read(experiment_plan); proof.locked(experiment_plan, files)
        stopping = experiment.get('stopping_rule', {})
        require(stopping.get('preregistered_before_remaining14_VAL_and_TEST') is True
            and stopping.get('deployment_or_success_module_benefit_assumed') is False
            and stopping.get('performance_screen_failure') and experiment.get('methods') == list(METHODS)
            and experiment.get('original_sources', {}).get('validation') == 275,
            'fixed original stopping rule missing')
        performance = _performance(groups)
        directory = proof_directory(out)
        artifacts = {
            'recorded_initial_conditions': dict(complete=True, accepted=True, split='validation',
                sources=275, pairs=2750, entries=comparisons, metric_only=True, policy_use_allowed=False,
                full_native_physics_state_identical_claimed=False),
            'canonical_cases': dict(complete=True, expected=1375, entries=receipts),
            'selection_usage': usage, 'baseline_component': component}
        pointers = {}
        for name, value in artifacts.items():
            path = directory / (name + '.json'); atomic_json(path, value); proof.file(path)
            pointers[name] = dict(path=proof.name(path), sha256=file_sha(path))
        for name in ('our_mani/multitask/composite_validation_release.py',
                     'our_mani/multitask/validation_release.py', 'our_mani/multitask/baseline_component.py',
                     'our_mani/multitask/trace_evidence.py',
                     'scripts/finalize_multitask_composite_validation_release_r10.py'):
            proof.file(name)
        proof.include(dict(proof.files))
        result = dict(schema=SCHEMA, complete=True, protocol_acceptance_passed=True, split='validation',
            TEST_used=False, formal_test_permitted=True, expected_source_cases=275,
            expected_method_cases=1375, completed_method_cases=1375,
            config_sha256=plan['config']['sha256'], validated_config=plan['config'],
            validated_input_lock=plan['frozen_inputs'], comparison_protocol_sha256=comparison_protocol_sha256(config),
            queue_plan=spec['ours_queue_plan'], explicit_VAL_composition=composition,
            actual_baseline_queue_plan=component['actual_early_queue_plan'],
            actual_baseline_config=component['actual_early_config'], actual_baseline_lock=component['actual_early_lock'],
            single_full_config_identity_claimed=False, raw_case_identities_rewritten=False,
            baseline_dependency_projection_revalidated=True,
            model_coverage_certificate=coverage_certificate, model_coverage_receipt=coverage_receipt,
            recorded_initial_conditions=dict(**pointers['recorded_initial_conditions'], sources=275, pairs=2750, accepted=True),
            canonical_cases=dict(**pointers['canonical_cases'], cases=1375),
            selection_usage=dict(**pointers['selection_usage'], diagnostics_valid=True),
            baseline_component=pointers['baseline_component'],
            initial_condition_tolerance_profile=config['initial_condition_tolerance_profile'],
            TRAIN_repeat_evidence=profile['TRAIN_evidence'],
            preregistered_experiment=dict(path=proof.name(experiment_plan), sha256=file_sha(proof.root/experiment_plan)),
            development_performance_screen=performance, development_performance_screen_passed=None,
            fixed_TEST_continuation_reason='Complete preregistered fixed comparison without a post-VAL numeric gate',
            success_rate_benefit_validated=False, input_files_sha256=proof.files,
            actual_R10_TEST_read=False, native_or_GPU_launched=False, model_fitting_performed=False,
            full_native_physics_state_identical_claimed=False,
            limitations=['Early baseline and final ours have distinct actual config/lock identities; dependency projection verified.',
                'Recorded initial conditions match; opaque solver/contact caches are unrecorded.',
                'VAL is development evidence; complete frozen formal TEST remains required.'])
        atomic_json(out, result)
        return result
    except Exception as error:
        pending = dict(schema=SCHEMA, complete=False, protocol_acceptance_passed=False,
            formal_test_permitted=False, split='validation', TEST_used=False, stage=stage, errors=[repr(error)],
            expected_source_cases=275, expected_method_cases=1375, no_release_written=True,
            input_files_sha256=proof.files, actual_R10_TEST_read=False, native_or_GPU_launched=False,
            model_fitting_performed=False, no_infrastructure_failure_scored_as_policy_negative=True)
        directory = out.parent / 'pending_validation_release'; directory.mkdir(parents=True, exist_ok=True)
        number = 0
        while (directory / (out.stem + f'__{number:04d}.json')).exists(): number += 1
        path = directory / (out.stem + f'__{number:04d}.json'); atomic_json(path, pending)
        pending['pending_report'] = dict(path=proof.name(path), sha256=file_sha(path))
        return pending
