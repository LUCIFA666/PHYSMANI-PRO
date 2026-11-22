"""Strict actual400 combined goal/semantic production consumer; no native/GPU."""
import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.contracts import Pose
from our_mani.multitask import acquisition_lineage as a,release_preflight as release,semantic_acquire as semantic
from our_mani.multitask import interaction,artifact_fitting as adapter,command_goal_reference as goal,basketball_command_goal_reference as basketball
from our_mani.multitask.reference_time import load_rule as load_reference


def run(report_path,output):
    begun=time.perf_counter();output=a.local(ROOT,output);a.require(not output.exists(),'fresh actual400 consumer output required');output.mkdir()
    files={};report=a.read(a.checked_pointer(ROOT,a.pointer(ROOT,a.local(ROOT,report_path)),files))
    configuration=a.read(a.checked_pointer(ROOT,report['configuration'],files))
    manifest=a.read(a.checked_pointer(ROOT,report['prefit400_source_proof'],files))
    a.require(report['actual_rollout_sources']==400 and report['distinct_original_parent_groups']==200,
        'original200+aux200/200parents required')
    a.require(report['all675_original_deposit_commands_retained'] is True,'all675 original closed/open commands required')
    reference=load_reference(ROOT,configuration['reference_time_derivation_rule'])
    code={str(Path(m.__file__).resolve()):a.file_sha(m.__file__) for m in (release,semantic,interaction,adapter,goal,basketball)}
    code[str(Path(__file__).resolve())]=a.file_sha(__file__);cases=[]
    for row in manifest['case_proofs']:
        path=a.checked_pointer(ROOT,row['canonical_receipt'],files);record=a.read(path)
        with np.load(a.checked_pointer(ROOT,row['cache'],files),allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        command=a.read(a.checked_pointer(ROOT,row['command_receipt'],files))
        cases.append(dict(path=path,record=record,entry=record['entry'],arrays=arrays,command_receipt=command,
            command_receipt_verified=True,command_receipt_sha256=row['command_receipt']['sha256']))
    inputs=release._Inputs(ROOT)
    for field in ('fitting_core_source_snapshot','fitting_producer_source_snapshot'):inputs.source_snapshot(configuration[field])
    rule,semantic_sources=release._semantic_sources(inputs,configuration,manifest,cases,reference,{})
    goal_rule,sources=release._command_goal_sources(inputs,configuration,manifest,cases,reference,{},rule)
    a.require(set(sources)==set(semantic_sources),'the two typed evidence views must cover the same400 actual raw sources')
    a.require(len(sources)==400 and len({s.grouping_source for s in sources.values()})==200,
        'consumer must reconstruct all400 with original parent grouping')
    a.require(sum(len(s.command_goal_reference_evidence.references)+len(s.command_goal_reference_evidence.pending_commands)
        for s in sources.values())==675,'recomputed complete675 deposit command scope differs')
    results={};negatives=[]
    for task,row in report['tasks'].items():
        values=[s for s in sources.values() if s.task_name==task]
        a.require(len(values)==200 and len({s.grouping_source for s in values})==100,'consumer task200/100parent coverage differs')
        fitting=a.read(a.checked_pointer(ROOT,row['detailed_fit_report'],files));support={}
        for label,proof in fitting['phase_fits'].items():
            phase,anchor,role,kind,manipulated,mode,basis=label.split('|')
            template=SimpleNamespace(phase=phase,anchor_id=anchor,reference_role=role,reference_kind=kind,
                manipulated_role=manipulated,reference_mode=mode,reference_time_basis=basis,
                reference_time_rule_sha256=proof.get('reference_time_rule_sha256'),
                lag_seconds=proof.get('selected_lag',0.))
            key,parents,attempts=release._completion_support(values,template)
            a.require(parents==set(proof['completed_parent_sources']) and len(attempts)==proof['completed_attempt_count'],
                'pending or passing source support differs from actual events/segments')
            folds={parent:i%2 for i,parent in enumerate(sorted(parents))}
            a.require(proof.get('source_to_fold',folds)==folds,'r0/r1 must share exact original-parent fold')
            terminal=[]
            for source in values:
                events=(source.semantic_acquire_evidence.segments if phase=='acquire' else source.completion_events)
                for event in events:
                    attempt=(source.source_id+':semantic:'+event.segment_id if phase=='acquire'
                        else source.source_id+':command'+str(event.origin_command_id))
                    if attempt not in attempts:continue
                    if basis=='command_goal_reference':
                        ref=next(r for r in source.command_goal_reference_evidence.references if r.command_id==event.origin_command_id)
                        reference_pose=Pose.from_xyzw(ref.intent_reference_xyzw)
                    else:reference_pose=interaction._completion_reference(source,event,template.lag_seconds)
                    tcp=Pose.from_xyzw(event.terminal_tcp_xyzw)
                    terminal.append(dict(source=source.grouping_source,attempt=attempt,position=reference_pose.rotation.T@(tcp.position-reference_pose.position),
                        rotation=reference_pose.rotation.T@tcp.rotation))
            a.require(len(terminal)==len(attempts),'every admitted attempt needs its actual terminal label')
            residual={parent:[] for parent in parents}
            for fold in (0,1):
                train=[r for r in terminal if folds[r['source']]!=fold]
                held=[r for r in terminal if folds[r['source']]==fold]
                counts={parent:sum(r['source']==parent for r in train) for parent in {r['source'] for r in train}}
                weights=np.asarray([1./counts[r['source']] for r in train]);weights/=weights.sum()
                position=np.average([r['position'] for r in train],axis=0,weights=weights)
                rotation=Rotation.from_matrix(np.asarray([r['rotation'] for r in train])).mean(weights).as_matrix()
                for r in held:residual[r['source']].append(dict(position=float(np.linalg.norm(position-r['position'])),
                    orientation=float(Rotation.from_matrix(rotation@r['rotation'].T).magnitude())))
            actual={parent:{field:float(np.median([r[field] for r in residual[parent]])) for field in ('position','orientation')}
                for parent in parents}
            a.require(all(abs(actual[parent][field]-proof['terminal_parent_residuals'][parent][field])<1e-12
                for parent in parents for field in ('position','orientation')),
                'independent terminal-only raw reconstruction differs from full-curve fitter crossfit')
            fit_config=configuration['interaction'];pt=max(fit_config['minimum_position_tolerance_m'],float(np.quantile(
                [r['position'] for r in actual.values()],fit_config['tolerance_quantile'])))
            rt=max(fit_config['minimum_orientation_tolerance_rad'],float(np.quantile(
                [r['orientation'] for r in actual.values()],fit_config['tolerance_quantile'])))
            a.require(abs(pt-proof['position_tolerance_m'])<1e-12 and abs(rt-proof['orientation_tolerance_rad'])<1e-12,
                'actual held-parent tolerance differs')
            support[label]=dict(parents=len(parents),attempts=len(attempts),parent_folds=folds,
                independent_raw_terminal_crossfit_tolerance_m=pt,orientation_tolerance_rad=rt,
                cap_passed=pt<=.03 and rt<=.15,full_fit_passed=proof['passed'])
        if 'library' not in row:
            metadata=dict(semantic_acquire_derivation=dict(default_legacy_behavior_changed=False,
                original_events_command_ids_masks_and_censoring_preserved=True,
                rule_sha256_values=[rule.sha256],helper_sha256_values=[rule.helper['sha256']],
                per_source={s.source_id:release._semantic_source_proof(s) for s in values}))
            metadata.update(command_goal_reference_derivation=dict(schema=goal.SCHEMA,
                rule_sha256_values=[goal_rule.sha256],helper_sha256=a.file_sha(goal.__file__),
                original_raw_completion_metric_preserved=True,
                per_source={s.source_id:goal.evidence_proof(s.command_goal_reference_evidence) for s in values}),
                public_completion_verifier_calibrated=False)
            status='fit_pending_no_library_promoted_source_proof_only'
        else:
            library=interaction.InteractionLibrary.load(a.checked_pointer(ROOT,row['library'],files));metadata=library.metadata
            status='actual200_source_library_consumer_verified'
        release._semantic_library_metadata(metadata,values,rule)
        release._command_goal_library_metadata(metadata,values,goal_rule)
        if 'library' in row:
            for template in library.templates:release._command_goal_geometry(values,template)
            template=next(t for t in library.templates if t.reference_time_basis=='command_goal_reference')
            changed=template.completion_geometry_xyzw.copy();changed[2]+=.01
            try:release._command_goal_geometry(values,replace(template,completion_geometry_xyzw=changed))
            except ValueError:negatives.append(task+':changed_public_canonical_geometry')
            else:raise AssertionError('invented canonical public geometry accepted')
        results[task]=dict(status=status,teacher_rollouts=200,parents=100,
            empty_or_censored_source_count=sum(not s.semantic_acquire_evidence.segments for s in values),support=support)
        for change in ('omit_aux','omit_censored','rawSHA','helper','rule','status','raw_preserved'):
            bad_metadata=deepcopy(metadata);body=bad_metadata['semantic_acquire_derivation']
            aux=next(s for s in values if s.augmentation_id.endswith('_r1'))
            empty=next((s for s in values if not s.semantic_acquire_evidence.segments),aux)
            if change=='omit_aux':body['per_source'].pop(aux.source_id)
            elif change=='omit_censored':body['per_source'].pop(empty.source_id)
            elif change=='rawSHA':body['per_source'][aux.source_id]['raw_bindings']['cache']['sha256']='f'*64
            elif change=='helper':body['helper_sha256_values']=['f'*64]
            elif change=='rule':body['rule_sha256_values']=['f'*64]
            elif change=='status':body['per_source'][aux.source_id]['status']='invented_success'
            else:body['original_events_command_ids_masks_and_censoring_preserved']=False
            try:release._semantic_library_metadata(bad_metadata,values,rule)
            except ValueError:negatives.append(task+':'+change)
            else:raise AssertionError('tampered actual400 metadata accepted: '+change)
        for change in ('omit_aux','intent','raw_metric','helper','rule','public_scope','discard_goal','discard_semantic'):
            bad=deepcopy(metadata);body=bad['command_goal_reference_derivation']
            aux=next(s for s in values if s.augmentation_id.endswith('_r1'))
            populated=next(s for s in values if s.command_goal_reference_evidence.references)
            if change=='omit_aux':body['per_source'].pop(aux.source_id)
            elif change=='intent':body['per_source'][populated.source_id]['references'][0]['intent_reference_xyzw'][0]+=.01
            elif change=='raw_metric':body['original_raw_completion_metric_preserved']=False
            elif change=='helper':body['helper_sha256']='f'*64
            elif change=='rule':body['rule_sha256_values']=['f'*64]
            elif change=='public_scope':bad['public_completion_verifier_calibrated']=True
            elif change=='discard_goal':bad.pop('command_goal_reference_derivation')
            else:bad.pop('semantic_acquire_derivation')
            try:
                release._command_goal_library_metadata(bad,values,goal_rule)
                release._semantic_library_metadata(bad,values,rule)
            except ValueError:negatives.append(task+':goal_combo:'+change)
            else:raise AssertionError('tampered combined-view metadata accepted: '+change)
    aux_case=next(c for c in cases if c['entry'].get('augmentation_replica')==1
        and any(row['raw_phase']=='acquire' for row in c['command_receipt']['commands']))
    old_pointer=a.pointer(ROOT,ROOT/'artifacts/framework/multitask_r10/semantic_acquire_rule_v6_aux200_setup/semantic_acquire_original_only_negative_rule.json')
    old=semantic.load_rule(ROOT,old_pointer)
    try:adapter.interaction_source(aux_case,reference_time_rule=reference,semantic_acquire_rule=old)
    except ValueError as error:
        a.require('outside' in str(error) or 'unknown' in str(error),'old rule rejected for an unrelated reason')
        negatives.append('current_original_only_actual_aux_still_unknown_no_legacy_fallback')
    else:raise AssertionError('old original-only rule silently admitted auxiliary data')
    try:goal.derive_source(aux_case,adapter.interaction_source(aux_case)[0],replace(goal_rule,auxiliary_contexts={}))
    except ValueError as error:
        a.require('unknown actual execution plan' in str(error),'goal unknown-aux rejected for unrelated reason')
        negatives.append('goal_requires_explicit_approved_auxiliary_context')
    else:raise AssertionError('goal rule silently admitted undeclared auxiliary data')
    for change in ('array','ledger'):
        bad=deepcopy(aux_case)
        if change=='array':bad['arrays']['measured_tcp_xyzw'][0,0]+=.01
        else:next(row for row in bad['command_receipt']['commands'] if row['raw_phase']=='acquire')['raw_waypoint_xyzw'][0]+=.01
        try:adapter.interaction_source(bad,reference_time_rule=reference,semantic_acquire_rule=rule,command_goal_rule=goal_rule)
        except ValueError:negatives.append('actual_aux_mutation:'+change)
        else:raise AssertionError('tampered actual auxiliary data accepted')
    inputs.include(files);inputs.rehash_all()
    for name,sha in code.items():a.require(a.file_sha(name)==sha,'source changed during400 consumer proof')
    result=dict(schema='r10_actual400_combined_command_goal_v4_semantic_v6_consumer_audit_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter()-begun,production_fit=a.pointer(ROOT,a.local(ROOT,report_path)),
        actual_teacher_sources_rebuilt=400,distinct_original_parent_groups=200,producer_consumer_all_source_proofs_identical=True,
        task_libraries=results,negative_cases_rejected=negatives,source_sha256=code,verified_files_sha256=inputs.files,
        old_v4_unknown_aux_evidence_preserved=True,combined_goal_and_semantic_views_rebuilt=True,formal_primary2800_or_combined3000_bundle_created=False,
        Context_K1_changed=False,native_GPU_or_collection_started=False,formal_model_policy_release=False)
    p=output/'report.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return dict(report=a.pointer(ROOT,p),tasks=results,negative_cases=len(negatives))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();print(json.dumps(run(args.report,args.out),ensure_ascii=False))
