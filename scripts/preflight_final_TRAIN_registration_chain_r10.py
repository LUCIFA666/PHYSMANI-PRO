"""Read-only archival/transition preflight; never inventory live completions."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as audit

BASE=ROOT/'artifacts/framework/multitask_r10'
OUT=BASE/'TRAIN_composite_registration_preflight_v2'


def permission_maps(value,path='value'):
    """Inspect direct source SHA maps and code pointers in permission payloads."""
    result=[]
    if isinstance(value,dict):
        if {'path','sha256'}<=set(value) and Path(value['path']).suffix in ('.py','.sh'):
            result.append(dict(location=path,original_path=value['path'],sha256=value['sha256'],kind='explicit_source_pointer'))
        for key,item in value.items():
            if key=='inputs_sha256' and isinstance(item,dict):
                result.extend(dict(location=path+'.inputs_sha256',original_path=name,sha256=checksum,kind='inputs_sha256')
                    for name,checksum in item.items() if Path(name).suffix in ('.py','.sh'))
            else:result.extend(permission_maps(item,path+'.'+key))
    elif isinstance(value,list):
        for index,item in enumerate(value):result.extend(permission_maps(item,path+'['+str(index)+']'))
    return result


def main():
    require=audit.require;require(not OUT.exists(),'fresh metadata-only preflight directory required')
    files={};registrations=[];reports=[];original=None
    for version in (12,13,14,15,16):
        setup=BASE/('TRAIN_recovery_v'+str(version)+('_known_failures_setup' if version==16 else '_initial_pair_setup'))
        registry_path=setup/'frozen/execution_registration.json'
        registry=audit.read(audit.checked_pointer(ROOT,audit.pointer(ROOT,registry_path),files))
        if original is None:original=registry['original'];registrations.append(original)
        require(registry['original']==original,'historical predecessor registration changed across fixed recovery versions')
        previous=audit.load_archived_plan(ROOT,original,files)
        current=audit.load_archived_plan(ROOT,registry['recovery'],files)
        change=audit.verify_recovery_transition(ROOT,previous,current,files)
        registrations.append(registry['recovery']);updated_live=[];declared=[];payload_sources=[]
        for name,item in current['sources'].items():
            live=audit.file_sha(name) if Path(name).is_file() else None
            if live!=item['sha256']:updated_live.append(dict(original_path=name,actual_execution_sha256=item['sha256'],
                actual_archived_source=item['path'],current_live_sha256=live,archive_verified=True,current_bytes_not_substituted=True))
        for key in ('development_reset_profile','RML_factory_hook','recovery_input'):
            ptr=current['plan']['protocol'][key];path=audit.checked_pointer(ROOT,ptr,files);payload=audit.read(path)
            for item in permission_maps(payload,key):
                name=str((ROOT/item['original_path']).resolve());live=audit.file_sha(name) if Path(name).is_file() else None
                source=current['sources'].get(name)
                require(source is not None and source['sha256']==item['sha256'],'declared executable dependency missing from actual execution archive')
                declared.append(dict(item,original_path=name,actual_archived_source=source['path'],
                    actual_archived_sha256=source['sha256'],current_live_sha256=live,current_live_matches_historical_pointer=live==item['sha256']))
            # Recovery metadata references the original executed source manifest
            # by immutable JSON pointer, not direct stale live-code pointers.
            if key=='recovery_input':payload_sources.append(dict(source_snapshots=payload.get('source_snapshots'),
                forbidden_unregistered_physical_rule_fields=False))
        intersect=sorted({row['original_path'] for row in updated_live}&{row['original_path'] for row in declared})
        require(not intersect,'actual historical direct dependency unexpectedly overlaps updated live code')
        before=current['plan']['original_demonstration_budget'];budget=current['plan']['augmentation_budget']
        require(current['plan']['cases']==previous['plan']['cases'] and len(current['plan']['cases'])==2800
            and before['train_sources']==1400 and before['validation_sources']==275
            and all(value['replicas']==1 and value['replica_indices']==[0] for value in budget['behaviors'].values()),
            'original complete roster/source/K1 budget changed')
        reports.append(dict(version=version,registration=registry['recovery'],archive_load_passed=True,
            archived_source_count=len(current['sources']),plan_input_count=len(current['plan']['inputs']),
            frozen_v1_transition_preflight_passed=True,actual_plan_sha256=current['plan']['plan_sha256'],
            allowed_recovery_case_count=len(change['allowed_recovery_case_ids']),
            explicit_protocol_change_pointers=[row['json_pointer'] for row in change['protocol_changes']],
            historical_code_different_from_live_but_correctly_verified_by_archive=updated_live,
            direct_declared_executable_pointer_dependencies=declared,
            updated_live_source_and_direct_pointer_dependency_intersection=intersect,
            source_snapshot_reference_metadata=payload_sources,
            original_full2800_cases_and1400_TRAIN275_VAL_K1_budget_identical=True))
    train=audit.read(ROOT/'artifacts/framework/protocol/train_manifest.json')
    validation=audit.read(ROOT/'artifacts/framework/protocol/validation_manifest.json')
    fixed=audit.validate_fixed_roster(previous['plan'],train,validation)
    for path in (Path(audit.__file__),Path(__file__)):
        audit.checked_pointer(ROOT,audit.pointer(ROOT,path),files)
    require(audit.file_sha(audit.__file__)=='9351ec6cb0a3da907ddf51c2a84d7fdf979a45a2186b8427e9433f757af9e8d4','frozen historical audit source changed')
    for name,checksum in files.items():require(audit.file_sha(name)==checksum,'archival preflight input changed while reading')
    OUT.mkdir()
    registry_list=OUT/'registrations.json';registry_list.write_text(json.dumps(registrations,ensure_ascii=False,indent=2)+'\n')
    report=dict(schema='r10_final_TRAIN_historical_registration_metadata_preflight_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        complete=True,metadata_preflight_passed=True,actual_registration_count=len(registrations),original_registration=original,
        registration_list=audit.pointer(ROOT,registry_list),recovery_versions=reports,actual_errors=[],
        frozen_v1_lineage_SHA=audit.file_sha(audit.__file__),frozen_v1_archive_and_all5_transitions_pass=True,
        historical_executable_sources_are_verified_from_actual_archives=True,
        current_direct_permission_source_maps_reference_unchanged_RML_helpers=True,
        old_changed_physical_helpers_and_old_changed_runner_are_not_direct_permission_live_source_pointers=True,
        historical_live_pointer_failure_reproduced=False,new_v2_resolver_required_for_this_actual_chain=False,
        potential_future_resolver_need='If an actual historical permission directly references a later-modified live executable, frozen v1 will reject; resolve against that actual archived source map in a separate version rather than rewriting old SHA.',
        all_v12_v13_v14_failure_registrations_retained=True,all_v15_v16_completion_registrations_retained=True,
        original_coverage_contract=fixed,eight_completed_parent_phase_requirement_relaxed=False,
        canonical_collections_scanned=False,full_live_composite_built=False,acquisition_complete_claimed=False,
        training_or_VAL_or_phase_or_TEST_release_claimed=False,frozen_sources_modified=False,native_or_GPU_started=False,
        verified_files_sha256=files)
    path=OUT/'report.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (OUT/'report.txt').write_text('\n'.join([
        'PASS:metadata-only actual v11+v12+v13+v14+v15+v16 registration/archive/transition preflight.',
        '6 actual plans preserved. Frozen v1 source SHA9351 unchanged. All5 archive loads/transitions passed,0 errors.',
        'Old physical-helper/runner code differs from current live code but is checked through its correct archived bytes.',
        'Actual direct executable permission SHA maps currently reference unchanged RML helpers; none intersects changed live code.',
        'No newv2 resolver or consumer change is necessary for this actual chain. A future stale direct code pointer must use a separate archived resolver.',
        'Full2800 roster,1400TRAIN,275VAL,K1 and8 completed-parent phase requirements unchanged.',
        'No live canonical inventory/composite constructed. Acquisition/model/VAL/TEST completion is not claimed. No native/GPU started.'])+'\n')
    print(json.dumps(dict(metadata_preflight_passed=True,actual_plans=len(registrations),transition_errors=0,report=audit.pointer(ROOT,path))))


if __name__=='__main__':main()
