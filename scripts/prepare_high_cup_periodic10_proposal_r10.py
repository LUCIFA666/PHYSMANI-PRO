"""Freeze a separate replica2 periodic-feedback proposal; no native/approval."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a,periodic_acquisition as audit
from our_mani.multitask.train_teacher_periodic_replan import PeriodicReplanningConfig
from our_mani.multitask.train_teacher_requery import PlanningRequeryConfig
BASE=ROOT/'artifacts/framework/multitask_r10'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def prepare(output):
    out=a.local(ROOT,output);a.require(not out.exists(),'fresh periodic proposal namespace required')
    old_registration_path=BASE/'TRAIN_high_cup_requery_all100_candidate_v2_setup/execution_registration_approved.json'
    old_registration=a.read(old_registration_path);old_proposal=a.read(a.local(ROOT,old_registration['proposal']['path']))
    primary=a.read(a.local(ROOT,old_proposal['original_plan']['path']));cases=audit.expected_cases(primary)
    primary_rows={row['parent_source_group']:row for row in primary['cases'] if row['task']==audit.TASK and row['teacher_used'] is True}
    previous_rows={row['proposed_case']['parent_source_group']:row['proposed_case'] for row in old_proposal['proposed_cases']}
    proposed=[]
    for case in cases:
        reset=a.expected_reset(case,primary['protocol']);parent=case['parent_source_group']
        old_reset=a.expected_reset(previous_rows[parent],primary['protocol'])
        initial_reset=a.expected_reset(primary_rows[parent],primary['protocol'])
        a.require(reset['seed'] not in (old_reset['seed'],initial_reset['seed'])
            and reset['variation']==old_reset['variation']==initial_reset['variation'],
            'replica2 must retain variation and use its own source-derived new seed; never resample collisions')
        proposed.append(dict(original_K1_case_id=primary_rows[parent]['case_id'],
            previous_counted_r1_case_id=previous_rows[parent]['case_id'],proposed_case=case,proposed_reset=reset,
            original_K1_reset=initial_reset,previous_counted_r1_reset=old_reset,
            same_original_parent=True,exact_r1_scene_seed_pair=False))
    out.mkdir(parents=True)
    snapshot_rows={}
    for name in audit.OWNED_SOURCES:
        path=ROOT/name;target=out/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True)
        checksum=a.file_sha(path);shutil.copyfile(path,target)
        a.require(a.file_sha(path)==a.file_sha(target)==checksum,'policy source changed while proposal was frozen')
        snapshot_rows[name]=dict(original_path=str(path.resolve()),archive=a.pointer(ROOT,target))
    snapshot=out/'source_snapshot.json'
    write(snapshot,dict(schema='r10_periodic_feedback_TRAIN_policy_source_snapshot_v1',sources=snapshot_rows,
        source_files_modified=False,native_GPU_or_collection_started=False))
    proposal=dict(schema=audit.PROPOSAL_VERSION,raw_audit_version=audit.RAW_AUDIT_VERSION,
        created_utc=datetime.now(timezone.utc).isoformat(),proposal_only=True,native_or_GPU_started=False,
        execution_registration_created=False,original_plan=old_proposal['original_plan'],
        complete_primary_lineage=old_proposal['complete_primary_lineage'],
        actual_coverage_diagnosis=a.pointer(ROOT,BASE/'TRAIN_high_cup_counted_actual100_bottleneck_diagnostic_v2/report.json'),
        primary_registrations=old_proposal['primary_registrations'],source_snapshot=a.pointer(ROOT,snapshot),
        task=audit.TASK,proposed_cases=proposed,first_fixed_case_id=cases[0]['case_id'],augmentation_budget=audit.BUDGET,
        requery_configuration=asdict(PlanningRequeryConfig()),periodic_configuration=asdict(PeriodicReplanningConfig()),
        teacher_policy='Fresh original-teacher goal after ten actual post-query feedback ticks while an arm path remains unfinished; actual gripper/release/recovery deferrals recorded; failed-planning waits independently retain every command.',
        periodic_teacher_policy_is_exact_author_schedule=False,replica2_is_exact_r1_scene_seed_pair=False,
        comparison_to_author='Separate extra TRAIN teacher schedule; per-query feedback clock and deferred actuation are not the author global step_id%10 loop.',
        all100_outcomes_retained=True,outcome_selected=False,infrastructure_retry=False,
        original_primary_cases_receipts_or_raw_labels_modified=False,original_Context_primary_view_changed=False,
        preservation=dict(minimum_completed_parent_sources=8,maximum_position_tolerance_m=.03,
            maximum_orientation_tolerance_rad=.15,failed_and_interrupted_commands_never_marked_completed=True),
        preserved_primary_and_counted_r1=dict(registration=a.pointer(ROOT,old_registration_path),
            manifest=a.pointer(ROOT,BASE/'TRAIN_high_cup_requery_all100_candidate_v2/manifest.json'),
            whole100_audit=a.pointer(ROOT,BASE/'TRAIN_high_cup_requery_v2_full100_actual_audit_v1/actual_manifest_audit.json'),
            bottleneck_diagnosis=a.pointer(ROOT,BASE/'TRAIN_high_cup_counted_actual100_bottleneck_diagnostic_v2/report.json')),
        limitations=['New replica2 scenes use new deterministic seeds, so outcomes cannot be interpreted as exact r1 paired-scene causal effects.',
            'No native operation, sufficient phase support, public perception or completion benefit established by CPU tests.',
            'Only a future explicitly approved execution registry may start these100 cases. No hidden infrastructure retry or outcome-selected expansion.',
            'New typed collection is separate from all existing3100 acquisitions and the original Context K1 view.'])
    path=out/'proposal.json';write(path,proposal)
    return a.pointer(ROOT,path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();print(json.dumps(prepare(args.out)))
