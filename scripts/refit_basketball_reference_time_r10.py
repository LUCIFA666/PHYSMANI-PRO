"""Actual original100-per-task TRAIN refit with explicit issued reference time.

No native/GPU or extra replicas. Historical code is checked in the preserved
CPU/source archive; original case/data receipts retain their exact old hashes.
"""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'scripts'):
    sys.path.insert(0,str(path))
from our_mani.multitask import acquisition_lineage as lineage
from our_mani.multitask.artifact_fitting import interaction_source
from our_mani.multitask.interaction import InteractionFitConfig,InteractionFittingPending,fit_interaction_library
from our_mani.multitask.reference_time import load_rule
from report_multitask_phase_support_r10 import verify_canonical_case

BASE=ROOT/'artifacts/framework/multitask_r10'
OUT=BASE/'basketball_command_issue_refit_v1'
PRIOR=BASE/'basketball_TRAIN_completed_skill_diagnostic_v1/report.json'
CONFIG=ROOT/'config/framework/multitask_r10_artifact_fit_reference_time_v2.json'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    lineage.require(not OUT.exists(),'fresh actual task100 issue refit directory required')
    began=time.perf_counter()
    prior=lineage.read(PRIOR);config=lineage.read(CONFIG)
    original_config=lineage.read(ROOT/config['original_numerical_configuration']['path'])
    lineage.require(config['interaction']==original_config['interaction'] and config['colour']==original_config['colour']
        and config['rigid_layout']==original_config['rigid_layout'],'original numerical fit settings changed')
    rule=load_rule(ROOT,config['reference_time_derivation_rule'])
    raw_rule=lineage.read(rule.path)
    snapshot=lineage.read(ROOT/raw_rule['source_snapshot']['path'])
    archived={str((ROOT/r['original_path']).resolve()):(ROOT/r['path']).resolve() for r in snapshot['entries']}
    verified={}
    for filename,checksum in prior['verified_files_sha256'].items():
        original=(ROOT/filename).resolve();actual=archived.get(str(original),original)
        lineage.require(lineage.file_sha(actual)==checksum,'prior target200 strict audit input changed: '+str(original))
        verified[str(actual)]=checksum
    lineage.require(lineage.file_sha(PRIOR)=='0b81e62c5ebd8c0e92037e66e857986659d5018c21a1346e8b3b10055606d7a6','prior strict audit changed')
    registrations=lineage.read(ROOT/prior['actual_registrations']['path'])
    base=lineage.read(ROOT/registrations[0]['plan']['path'])
    expected={case['case_id']:case for case in base['cases']}
    OUT.mkdir();results={}
    for task,previous in prior['tasks'].items():
        cases=[];sources=[];audits=[]
        for row in previous['case_receipts_and_adapter_diagnostics']:
            path=ROOT/row['canonical_receipt']['path']
            lineage.require(lineage.file_sha(path)==row['canonical_receipt']['sha256'],'original canonical changed')
            case,inputs=verify_canonical_case(ROOT,path,expected[row['case_id']],row['actual_plan_sha256'])
            verified.update(inputs)
            source,audit=interaction_source(case,reference_time_rule=rule)
            cases.append(case);sources.append(source)
            audits.append(dict(case_id=row['case_id'],canonical=row['canonical_receipt'],raw_child=row['raw_child_receipt'],
                raw_parent=row['parent_receipt'],actual_plan_sha256=row['actual_plan_sha256'],source_id=source.source_id,
                parent_source_group=source.grouping_source,derived_adapter_audit=audit))
        lineage.require(len(cases)==100 and len({source.grouping_source for source in sources})==100,'full original task100 source roster required')
        manifest=OUT/task/'TRAIN_derived_fit_manifest.json'
        write(manifest,dict(schema='r10_actual_task100_explicit_reference_time_fit_manifest_v1',TRAIN_only=True,
            VAL_used=False,TEST_used=False,entries=[case['entry'] for case in cases],original_strict_audit=lineage.pointer(ROOT,PRIOR),
            actual_acquisition_registrations=prior['actual_registrations'],fitting_configuration=lineage.pointer(ROOT,CONFIG),
            reference_time_rule=config['reference_time_derivation_rule'],source_derivation_provenance=audits,
            full1400_or_formal_release_claimed=False,raw_current_terminal_labels_and_all_masks_unchanged=True))
        started=time.perf_counter()
        try:
            library=fit_interaction_library(sources,InteractionFitConfig(**config['interaction']),TRAIN_source_manifest_path=manifest)
            result=dict(status='task100_fit_passed_diagnostic_only',diagnostics=library.metadata['completed_skill_fitting'],
                library_or_runtime_exported=False)
        except InteractionFittingPending as error:
            result=dict(status='task100_fit_pending',error=str(error),diagnostics=error.diagnostics,
                library_or_runtime_exported=False)
        result.update(CPU_seconds=time.perf_counter()-started,original_teacher_cases=100,
            distinct_original_parents=100,all_original_cases_retained=True,original_numeric_configuration_unchanged=True,
            derived_manifest=lineage.pointer(ROOT,manifest),raw_original_masks_terminal_labels_and_ledgers_unchanged=True)
        write(OUT/task/'fit_report.json',result);results[task]=result
        print(json.dumps(dict(task=task,status=result['status'],partitions={key:dict(parents=value['completed_parent_count'],
            passed=value.get('passed'),reference_time_basis=value['reference_time_basis'],position_tolerance_m=value.get('position_tolerance_m'),
            reason=value.get('reason')) for key,value in result['diagnostics']['phase_fits'].items()})),flush=True)
    verified.update(rule.verified_files_sha256)
    for path in (Path(__file__),CONFIG,PRIOR,ROOT/'our_mani/multitask/interaction.py',ROOT/'our_mani/multitask/artifact_fitting.py',
            ROOT/'our_mani/multitask/reference_time.py'):
        verified[str(path)]=lineage.file_sha(path)
    for filename,checksum in verified.items():
        lineage.require(lineage.file_sha(filename)==checksum,'source or actual training inputs changed during refit')
    write(OUT/'report.json',dict(schema='r10_actual_basketball_task100_command_issue_refit_v1',TRAIN_only=True,
        VAL_TEST_read=False,native_GPU_or_extra_replicas_started=False,original_K1_and_raw_receipts_unchanged=True,
        full1400_or_formal_release_claimed=False,fitting_configuration=lineage.pointer(ROOT,CONFIG),
        original_numerical_configuration=config['original_numerical_configuration'],reference_time_rule=config['reference_time_derivation_rule'],
        tasks=results,verified_files_sha256=verified,wall_seconds=time.perf_counter()-began))
    print(json.dumps(dict(report=lineage.pointer(ROOT,OUT/'report.json'))))


if __name__=='__main__':main()
