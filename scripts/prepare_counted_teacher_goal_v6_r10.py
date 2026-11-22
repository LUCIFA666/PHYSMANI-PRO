"""Freeze explicit goal-v6 metadata; never launch fitting, native or GPU work.

Candidate mode archives the currently observed manifest and cannot be loaded
for production. Final mode requires actual complete100 and complete3100 audit
pointers; the subsequent source reconstruction still calls the real auditors.
"""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask import acquisition_lineage as a
from our_mani.multitask import artifact_fitting as adapter
from our_mani.multitask import command_goal_reference as goal
from our_mani.multitask import counted_teacher_bundle, interaction, requery_acquisition, semantic_acquire

BASE = ROOT / 'artifacts/framework/multitask_r10'
PREDECESSOR = BASE / 'command_goal_reference_rule_v5_translation_revision3_setup'
ACTUAL_MANIFEST = BASE / 'TRAIN_high_cup_requery_all100_candidate_v2/manifest.json'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def prepare(output, *, mode, complete100_audit=None, bundle=None, bundle_audit=None):
    a.require(mode in ('candidate', 'final'), 'explicit candidate/final mode required')
    a.require((mode == 'final') == all(x is not None for x in (complete100_audit, bundle, bundle_audit)),
              'final requires all three actual complete audit/bundle arguments')
    if mode == 'candidate':
        a.require(all(x is None for x in (complete100_audit, bundle, bundle_audit)),
                  'candidate cannot carry partial final audit arguments')
    out = a.local(ROOT, output)
    a.require(not out.exists(), 'fresh goal-v6 metadata directory required')
    manifest = a.read(ACTUAL_MANIFEST)
    registry = a.pointer(ROOT, ROOT / goal.OWNED_COUNTED_REGISTRY_PATH)
    a.require(registry['sha256'] == goal.OWNED_COUNTED_REGISTRY_SHA, 'approved counted registration changed')
    a.require(manifest['version'] == requery_acquisition.MANIFEST_VERSION
              and manifest['expected'] == 100 and manifest['execution_registration'] == registry,
              'manifest is not the owned v2 fixed100 collection')
    complete = manifest.get('complete') is True and manifest.get('completed') == 100
    if mode == 'final':
        a.require(complete and len(manifest['entries']) == len(manifest['case_receipts']) == 100
                  and not manifest['missing_case_ids'], 'actual terminal100 manifest required')
        audit100_path = a.local(ROOT, complete100_audit)
        audit100 = a.read(audit100_path)
        # The audit script may keep the strict auditor return under ``audit``.
        proof100 = audit100.get('audit', audit100)
        a.require(proof100.get('audit_passed') is True and proof100.get('complete') is True
                  and proof100['acquisition_projection'] == manifest,
                  'actual strict whole100 audit/projection required')
        bundle_path = a.local(ROOT, bundle)
        bundle_value = a.read(bundle_path)
        a.require(bundle_value['version'] == counted_teacher_bundle.BUNDLE_VERSION
                  and bundle_value['counted_high_cup_manifest'] == a.pointer(ROOT, ACTUAL_MANIFEST),
                  '3100 bundle must bind this actual complete100 manifest')
        audit3100_path = a.local(ROOT, bundle_audit)
        audit3100 = a.read(audit3100_path)
        proof3100 = audit3100.get('audit', audit3100)
        projection = proof3100['acquisition_projection']
        a.require(proof3100.get('audit_passed') is True and proof3100.get('complete') is True
                  and projection['version'] == counted_teacher_bundle.BUNDLE_VERSION
                  and projection['expected'] == projection['completed'] == len(projection['entries']) == 3100
                  and projection['teacher_rollouts'] == 1700 and projection['TRAIN_parent_sources'] == 1400
                  and projection['original_Context_K1_hold_rollouts'] == 1400
                  and projection['primary_Context_view_unchanged'] is True
                  and projection['counted_high_cup_manifest'] == bundle_value['counted_high_cup_manifest']
                  and projection['primary3000_bundle'] == bundle_value['primary3000_bundle'],
                  'actual complete3100 audited teacher/hold/parent/source projection required')
    out.mkdir(parents=True)
    observed = out / 'observed_manifest.json'
    shutil.copyfile(ACTUAL_MANIFEST, observed)
    a.require(a.read(observed) == manifest, 'manifest changed during metadata preparation')
    # A candidate intentionally binds an archived partial observation, outside
    # the actual collection; production is disabled and the loader rejects it.
    manifest_pointer = a.pointer(ROOT, ACTUAL_MANIFEST if mode == 'final' else observed)
    rule = deepcopy(a.read(PREDECESSOR / 'command_goal_rule_explicit_complete_aux200.json'))
    rule.update(schema=goal.FIT_RULE_SCHEMA_V6, production_enabled=mode == 'final',
                helper=a.pointer(ROOT, Path(goal.__file__)),
                original_adapter_source=a.pointer(ROOT, Path(adapter.__file__)),
                original_interaction_source=a.pointer(ROOT, Path(interaction.__file__)),
                counted_teacher_contexts=[dict(registration=registry, manifest=manifest_pointer,
                                               auditor=a.pointer(ROOT, Path(requery_acquisition.__file__)))])
    rule_path = out / 'command_goal_rule.json'
    write(rule_path, rule)
    sem = deepcopy(a.read(BASE / 'command_goal_reference_rule_v5_translation_revision2_setup/semantic_acquire_rule_v8_current_fitter_adapter.json'))
    sem['source_bindings'].update(helper=a.pointer(ROOT, Path(semantic_acquire.__file__)),
                                  fitter=a.pointer(ROOT, Path(interaction.__file__)),
                                  adapter=a.pointer(ROOT, Path(adapter.__file__)))
    semantic_path = out / 'semantic_acquire_rule_v9_current_goalv6_adapter.json'
    write(semantic_path, sem)
    source_names = ('command_goal_reference.py', 'translation_command_goal_reference.py',
                    'interaction.py', 'reference_time.py', 'artifact_fitting.py', 'release_preflight.py',
                    'semantic_acquire.py', 'bin_command_goal_reference.py', 'basketball_command_goal_reference.py',
                    'acquisition_lineage.py', 'training_view.py', 'acquisition_manifest.py',
                    'teacher_supplement.py', 'requery_acquisition.py', 'counted_teacher_bundle.py',
                    'specs.py', 'public_roles.py', 'evaluation.py')
    paths = [ROOT / 'our_mani/multitask' / name for name in source_names]
    paths += [ROOT / 'our_mani/contracts.py', Path(__file__).resolve(),
              ROOT / 'scripts/fit_multitask_interaction_artifacts_r10.py',
              ROOT / 'tests/test_multitask_goal_counted_teacher_context.py']
    archive = {}
    for source in paths:
        target = out / 'source_archive' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        archive[str(source.relative_to(ROOT))] = dict(live_source=a.pointer(ROOT, source),
                                                     archived_source=a.pointer(ROOT, target))
    snapshot_path = out / 'source_snapshot.json'
    write(snapshot_path, dict(schema='r10_goal_v6_explicit_counted_teacher_source_snapshot_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), sources=archive,
        predecessor=a.pointer(ROOT, PREDECESSOR / 'source_snapshot.json'),
        interaction_curve_math_or_caps_changed=False, public_runtime_roles_or_forecasts_changed=False,
        original_canonical_record_equality_strengthened=True, new_counted_execution_context_explicit=True,
        native_GPU_collection_or_fitting_started=False))
    result = dict(schema='r10_goal_v6_explicit_counted_teacher_preparation_v1', mode=mode,
        production_enabled=mode == 'final', actual100_complete_claimed=mode == 'final',
        observed_manifest=a.pointer(ROOT, observed), observed_completed=manifest['completed'],
        observed_complete=manifest['complete'], rule=a.pointer(ROOT, rule_path),
        semantic_rule=a.pointer(ROOT, semantic_path), source_snapshot=a.pointer(ROOT, snapshot_path),
        original_case_rosters_labels_failures_censoring_and_masks_preserved=True,
        thresholds=dict(minimum_parent_sources=8, maximum_position_tolerance_m=.03,
                        maximum_orientation_tolerance_rad=.15),
        native_GPU_collection_fitting_or_formal_policy_started=False)
    if mode == 'final':
        config = deepcopy(a.read(PREDECESSOR / 'artifact_fit_configuration_original300.json'))
        config.pop('translation_candidate_scope', None)
        config.update(version='r10_full3100_goalv6_semanticv9_TRAIN_configuration_v1',
                      expected_native_TRAIN_acquisitions=3100, expected_teacher_TRAIN_acquisitions=1700,
                      command_goal_reference_rule=a.pointer(ROOT, rule_path),
                      semantic_acquire_derivation_rule=a.pointer(ROOT, semantic_path),
                      fitting_core_source_snapshot=a.pointer(ROOT, snapshot_path),
                      fitting_producer_source_snapshot=a.pointer(ROOT, snapshot_path),
                      complete_current3100_bundle_audit=a.pointer(ROOT, audit3100_path),
                      complete_counted_high_cup100_audit=a.pointer(ROOT, audit100_path))
        config['full14_fit_scope'].update(current_total_acquisitions=3100, teacher_acquisitions=1700,
            auxiliary_teacher_acquisitions=300,
            counted_high_cup_teacher_acquisitions=100, primary_acquisitions=2800,
            basketball_auxiliary_teacher_acquisitions=200, distinct_original_TRAIN_parent_sources=1400,
            primary_Context_hold_acquisitions=1400)
        config['fixed_auxiliary_teacher_fit_scope'].pop('complete_current_bundle3000_verified', None)
        config['fixed_auxiliary_teacher_fit_scope']['complete_primary2800_plus_basketball200_subbundle_verified'] = True
        config['fixed_counted_high_cup_teacher_fit_scope'] = dict(explicit_registration=registry,
            actual_complete100_manifest=a.pointer(ROOT, ACTUAL_MANIFEST), teacher_acquisitions=100,
            replica=1, maximum_ticks_each=200, parent_groups=100,
            original_primary2800_and_basketball200_retained=True, is_infrastructure_recovery=False,
            is_basketball_teacher_supplement=False, all_failed_censored_sources_retained=True,
            original_Context_K1_hold1400_unchanged=True, public_observability_and_completion_not_qualified_by_fit=True)
        config_path = out / 'artifact_fit_configuration_full3100.json'
        write(config_path, config)
        argv = [str(ROOT / '.conda-env/bin/python'), str(ROOT / 'scripts/fit_multitask_interaction_artifacts_r10.py'),
                '--collector-manifest', str(bundle_path.relative_to(ROOT)),
                '--demonstrations', 'artifacts/framework/multitask_r10/demonstrations/manifest.json',
                '--original-prior', 'artifacts/framework/multitask_r10/original_TRAIN_action_phase_priors.json',
                '--fit-config', str(config_path.relative_to(ROOT)),
                '--out', 'artifacts/framework/multitask_r10/TRAIN_artifacts_full_goalv6_semanticv9_3100_v1']
        result.update(configuration=a.pointer(ROOT, config_path), complete100_audit=a.pointer(ROOT, audit100_path),
                      bundle=a.pointer(ROOT, bundle_path), bundle_audit=a.pointer(ROOT, audit3100_path),
                      proposed_CPU_fit_argv=argv, proposed_CPU_fit_not_started=True)
    write(out / 'preparation.json', result)
    return a.pointer(ROOT, out / 'preparation.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--mode', choices=('candidate', 'final'), required=True)
    parser.add_argument('--complete100-audit')
    parser.add_argument('--bundle')
    parser.add_argument('--bundle-audit')
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, mode=args.mode, complete100_audit=args.complete100_audit,
                             bundle=args.bundle, bundle_audit=args.bundle_audit)))
