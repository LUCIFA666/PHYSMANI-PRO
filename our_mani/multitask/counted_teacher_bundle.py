"""Admit complete primary3000 plus the separate counted high-cup100 TRAIN.

Both inputs must pass their own actual raw acquisition auditors. This does
not change the original Context view, parent folds or numerical fitting caps.
"""
from pathlib import Path

from . import acquisition_lineage as lineage

BUNDLE_VERSION='r10_TRAIN_primary3000_plus_counted_high_cup100_v1'
require=lineage.require


def audit_bundle(root,path,*,require_complete=True):
    root=Path(root).resolve();path=lineage.local(root,path);value=lineage.read(path);files={}
    require(set(value)=={'version','primary3000_bundle','counted_high_cup_manifest','primary_Context_view_unchanged'}
        and value['version']==BUNDLE_VERSION and value['primary_Context_view_unchanged'] is True,
        'strict primary3000 plus counted high-cup100 bundle required')
    from . import acquisition_manifest,teacher_supplement,requery_acquisition
    primary_path=lineage.checked_pointer(root,value['primary3000_bundle'],files)
    header=lineage.read(primary_path)
    require(header.get('version')==teacher_supplement.BUNDLE_VERSION
        or header.get('schema')==teacher_supplement.BUNDLE_VERSION,
        'counted supplement requires the complete audited original primary3000 bundle')
    primary,proof=acquisition_manifest.load_acquisition_manifest(root,primary_path)
    require(proof is not None and proof['complete'] is True and primary['complete'] is True
        and primary['expected']==primary['completed']==len(primary['entries'])==3000
        and primary['TRAIN_parent_sources']==1400 and primary['teacher_rollouts']==1600
        and primary['original_Context_K1_hold_rollouts']==1400
        and primary['primary_Context_view_unchanged'] is True,
        'counted supplement cannot replace missing original3000 TRAIN data')
    original=primary['entries'];parents={row['parent_source_group'] for row in original}
    require(len(parents)==1400 and sum(row['teacher_used'] is True for row in original)==1600
        and sum(row['context_primary_view'] is True for row in original)==1400,
        'original teacher/hold/parent budget differs')
    auxiliary_path=lineage.checked_pointer(root,value['counted_high_cup_manifest'],files)
    extra=requery_acquisition.audit_manifest(root,auxiliary_path,require_complete=True)
    auxiliary=extra['acquisition_projection']
    require(extra['audit_passed'] is True and extra['complete'] is True
        and auxiliary['version']==requery_acquisition.MANIFEST_VERSION
        and auxiliary['complete'] is True
        and auxiliary['expected']==auxiliary['completed']==len(auxiliary['entries'])==100
        and extra['base_plan_sha256']==primary['base_plan_sha256']
        and auxiliary['primary_K1_roster_unmodified'] is True
        and auxiliary['primary_Context_view_unmodified'] is True,
        'all100 actual counted teacher cases from the same original primary plan required')
    expected_parents={row['parent_source_group'] for row in original
        if row['task']==requery_acquisition.TASK and row['teacher_used'] is True
        and row['augmentation_replica']==0}
    require(len(expected_parents)==100
        and {row['parent_source_group'] for row in auxiliary['entries']}==expected_parents
        and all(row['task']==requery_acquisition.TASK and row['teacher_used'] is True
            and row['context_primary_view'] is False and row['context_training_eligible'] is False
            and row['interaction_eligible'] is True
            and type(row['augmentation_replica']) is int and row['augmentation_replica']==1
            and type(row['behavior_replica']) is int and row['behavior_replica']==1
            and row['maximum_collection_ticks']==200 for row in auxiliary['entries']),
        'new data must retain all100 original high-cup parents, teacher-only replica1 and grouped folds')
    mapping=dict(proof['per_case_actual_plan_sha256'])
    require(not set(mapping)&set(extra['per_case_actual_plan_sha256']),
        'counted supplement duplicates an existing acquisition case')
    mapping.update(extra['per_case_actual_plan_sha256'])
    entries=original+auxiliary['entries'];receipts=primary['case_receipts']+auxiliary['case_receipts']
    require(len(entries)==len(receipts)==len(mapping)==3100
        and len({row['case_id'] for row in receipts})==3100
        and {row['case_id'] for row in receipts}==set(mapping)
        and all(row['actual_plan_sha256']==mapping[row['case_id']] for row in receipts)
        and sum(row['teacher_used'] is True for row in entries)==1700
        and sum(row['context_primary_view'] is True for row in entries)==1400,
        '3100 cases must preserve exact raw receipts, actual plans and separate teacher/hold budgets')
    for checked in (proof,extra):
        for name,checksum in checked['verified_files_sha256'].items():
            require(name not in files or files[name]==checksum,'actual raw/source SHA conflict in counted bundle')
            files[name]=checksum
    files[str(path)]=lineage.file_sha(path)
    files[str(Path(__file__).resolve())]=lineage.file_sha(Path(__file__).resolve())
    require(all(lineage.file_sha(name)==checksum for name,checksum in files.items()),
        'counted bundle inputs changed during audit')
    projection=dict(version=BUNDLE_VERSION,split='train',TEST_used=False,complete=True,
        expected=3100,completed=3100,entries=entries,case_receipts=receipts,missing_case_ids=[],
        plan_sha256=None,single_plan_SHA_claimed=False,projection_rewrites_original_receipts=False,
        per_case_actual_plan_sha256=mapping,actual_plan_sha256_values=sorted(set(mapping.values())),
        base_plan_sha256=primary['base_plan_sha256'],primary3000_bundle=value['primary3000_bundle'],
        counted_high_cup_manifest=value['counted_high_cup_manifest'],primary_complete3000=True,
        counted_high_cup_complete100=True,TRAIN_parent_sources=1400,teacher_rollouts=1700,
        original_Context_K1_hold_rollouts=1400,primary_Context_view_unchanged=True,
        primary_augmentation_budget=dict(original=primary['original_primary_augmentation_budget'],
            basketball=primary['auxiliary_augmentation_budget']),
        counted_high_cup_augmentation_budget=requery_acquisition.BUDGET,
        new_teacher_policy_is_not_original_author_reproduction=True)
    return dict(schema=BUNDLE_VERSION,audit_passed=True,complete=True,
        execution_kind='primary3000_plus_separate_counted_high_cup_teacher100',
        acquisition_projection=projection,per_case_actual_plan_sha256=mapping,
        actual_plan_sha256_values=projection['actual_plan_sha256_values'],
        verified_files_sha256=files,primary3000_proof=proof,counted_high_cup_proof=extra,
        formal_model_or_policy_release=False)
