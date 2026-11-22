"""Convert every real TRAIN receipt for diagnostics; never fit a skill/model."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import analyze_multitask_smoke_r10 as common
from scripts import collect_multitask_training_r10 as c
from our_mani.multitask.artifact_fitting import load_train_cases,interaction_source
from our_mani.multitask.interaction import _completion_reference,InteractionArtifactError


def audit(manifest_path):
    manifest,cases=load_train_cases(ROOT,manifest_path);rows=[];errors=[]
    for case in cases:
        entry=case['entry'];row=dict(task=entry['task'],source_id=entry['source_id'],
            parent_source_group=entry['parent_source_group'],behavior=entry['behavior'],
            interaction_eligible=entry['interaction_eligible'],frames=len(case['arrays']['times']))
        if not entry['interaction_eligible']:
            rows.append(dict(row,status='hold_audited_not_interaction',events=[]));continue
        try:
            source,detail=interaction_source(case);events=[]
            for event in source.completion_events:
                indices=np.flatnonzero(source.command_ids==event.origin_command_id)
                valid=0;excluded=Counter();terminal=0;binding=event.binding
                try:_completion_reference(source,event,0.);terminal=1
                except (InteractionArtifactError,KeyError):excluded['terminal_reference_unavailable']+=1
                for index in indices:
                    if (source.references[index]!=binding or not event.attempt_start_time-1e-8<=source.times[index]<event.attempt_end_time-1e-8):
                        excluded['outside_attempt_or_reference_changed']+=1;continue
                    try:
                        if binding.reference_kind not in ('world','phase_origin'):
                            source.body_tracks[binding.reference_id].at(source.times[index],
                                orientation_required=binding.reference_kind=='observed_se3',
                                same_mode_as=event.reference_mode,not_before=source.times[index])
                        valid+=1
                    except (InteractionArtifactError,KeyError):excluded['invalid_mode_gap_or_orientation']+=1
                events.append(dict(phase=event.phase,evidence=event.evidence,command_id=event.origin_command_id,
                    reference_mode=event.reference_mode,duration_seconds=event.attempt_end_time-event.attempt_start_time,
                    eligible_zero_lag_curve_rows=valid,terminal_endpoint_rows=terminal,excluded_rows=dict(excluded)))
            rows.append(dict(row,status='converted_no_fit',events=events,audit=detail,
                phase_event_counts=dict(Counter(event['phase'] for event in events))))
        except Exception as error:errors.append(dict(row,error=type(error).__name__+': '+str(error)))
    return dict(version='r10_real_TRAIN_interaction_conversion_audit_v1',collector_manifest=common.pointer(manifest_path),
        script=common.pointer(__file__),adapter_source=common.pointer(ROOT/'our_mani/multitask/artifact_fitting.py'),
        interaction_source=common.pointer(ROOT/'our_mani/multitask/interaction.py'),acquisition_receipts_verified=len(cases),
        teacher_cases=sum(case['entry']['teacher_used'] for case in cases),hold_cases=sum(not case['entry']['teacher_used'] for case in cases),
        conversion_errors=errors,rows=rows,accepted_phase_events=sum(len(row.get('events',[])) for row in rows),
        minimum_eight_parent_support_not_established=True,TRAIN_only=True,VAL_read=False,TEST_read=False,
        model_fitting_performed=False,native_or_GPU_launched=False,original_data_modified=False,
        formal_comparison_ready=False,duration_is_command_level_skill_attempt_not_total_phase=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=c.BASE/'TRAIN_smoke_collection_v6/manifest.json')
    parser.add_argument('--report',type=Path,default=c.BASE/'v6_interaction_adapter_audit.json')
    args=parser.parse_args();report=audit(args.manifest);c.write_json(args.report,report,overwrite=True)
    print({key:report[key] for key in ('acquisition_receipts_verified','teacher_cases','hold_cases','accepted_phase_events','conversion_errors')})
if __name__=='__main__':main()
