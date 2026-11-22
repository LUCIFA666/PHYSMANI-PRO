"""Export current TRAIN fits after all original goals pass a hierarchy replay."""
from pathlib import Path
from dataclasses import asdict
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.contracts import Pose
from our_mani.multitask import interaction,artifact_fitting as adapter,command_goal_reference as goal
from our_mani.multitask.remove_command_goal_candidates_v46 import INTENT_BASIS
from our_mani.multitask.remove_runtime_v48 import SCHEMA,TASKS
from our_mani.multitask.evaluation import atomic_json,file_sha,verify_inputs


def pointer(path):return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))


def main():
    base=ROOT/'artifacts/framework/multitask_r10';out=base/'remove_source_qualified_libraries_v48'
    if out.exists():raise FileExistsError('preserve previous export')
    audit_path=base/'remove_original_goal_hierarchy_audit_v45/report.json';audit=json.loads(audit_path.read_text())
    if not audit['source_qualification_passed'] or audit['original_parents']!=200 or audit['commands']!=901:raise ValueError('all200 original raw waypoint goals must reproduce')
    verify_inputs(ROOT,audit['inputs_sha256'])
    report_path=base/'remove_reference_current_original200_CPU_candidate_v47/report.json';report=json.loads(report_path.read_text());verify_inputs(ROOT,report['verified_files_sha256'])
    prefit_path=ROOT/report['prefit_proof']['path']
    if file_sha(prefit_path)!=report['prefit_proof']['sha256']:raise ValueError('current actual prefit proof changed')
    prefit=json.loads(prefit_path.read_text());sources={};canonical={};stage_counts={}
    for proof in prefit['case_proofs']:
        raw=proof['original_raw_bindings'];path=ROOT/raw['canonical_receipt']['path'];record=json.loads(path.read_text());cache=ROOT/raw['cache']['path'];cp=ROOT/raw['commands']['path']
        for key in ('canonical_receipt','cache','commands'):
            if file_sha(ROOT/raw[key]['path'])!=raw[key]['sha256']:raise ValueError('current raw source changed')
        with np.load(cache,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        case=dict(path=path,record=record,entry=record['entry'],arrays=arrays,command_receipt=json.loads(cp.read_text()),command_receipt_verified=True,command_receipt_sha256=raw['commands']['sha256'])
        source=adapter.interaction_source(case)[0];sources.setdefault(source.task_name,[]).append(source)
        references={r['command_id']:r for r in proof['candidate_references']};commands={c['command_id']:c for c in source.command_ledger}
        for event in source.completion_events:
            cid=event.origin_command_id;ref=references.get(cid)
            if ref is None or (event.phase=='approach' and commands[cid]['raw_stage']=='wp0'):continue
            b=ref['binding'];key='|'.join((event.phase,b['anchor_id'],b['reference_role'],b['reference_kind'],b['manipulated_role'],ref['reference_mode'],INTENT_BASIS))
            intent=Pose.from_xyzw(ref['intent_reference_xyzw']);wp=Pose.from_xyzw(event.committed_world_waypoint_xyzw)
            local=Pose(intent.rotation.T@(wp.position-intent.position),intent.rotation.T@wp.rotation)
            for progress in (0.,1.):canonical.setdefault((source.task_name,key),[]).append(dict(source=source.grouping_source,rollout_id=source.source_id,
                attempt_id=source.source_id+':command'+str(cid),progress=progress,position=local.position,rotation=local.rotation,opening=0.,terminal_event=progress==1.))
    out.mkdir();results={};manifest=base/'TRAIN_artifacts_full_goalv6_semanticv9_3100_v1/TRAIN_fit_manifest.json'
    config=json.loads((base/'command_goal_reference_rule_v6_counted_final_setup_v1/artifact_fit_configuration_full3100.json').read_text());fit_config=interaction.InteractionFitConfig(**config['interaction'])
    rule_sha=file_sha(audit_path)
    for task in TASKS:
        rows=sources[task]
        if len(rows)!=100 or len({r.grouping_source for r in rows})!=100:raise ValueError('all original parents required')
        fit_path=ROOT/report['tasks'][task]['candidate']['path'];fit=json.loads(fit_path.read_text())
        if file_sha(fit_path)!=report['tasks'][task]['candidate']['sha256'] or not fit['all_required_phases_and_observed_partitions_passed']:raise ValueError('current original-cap fit failed')
        templates=[];basis_rules=[]
        for key,d in fit['phase_fits'].items():
            if not d['passed'] or d['completed_parent_count']<8 or d['position_tolerance_m']>.03 or d['orientation_tolerance_rad']>.15:raise ValueError('original interaction fit gate failed')
            phase,anchor,role,kind,manipulated,mode,basis=key.split('|');curve=d['fitted_curve']
            t=dict(phase=phase,anchor_id=anchor,reference_role=role,reference_kind=kind,manipulated_role=manipulated,
                reference_mode=mode,reference_time_basis=interaction.COMMAND_GOAL if basis==INTENT_BASIS else basis,
                reference_time_rule_sha256=rule_sha if basis==INTENT_BASIS else None,lag_seconds=d['selected_lag'],
                progress=curve['progress'],positions=curve['positions'],rotations=Rotation.from_quat(curve['rotation_xyzw']).as_matrix().tolist(),
                opening=curve['opening'],duration_seconds=d['duration_seconds'],duration_budget_seconds=max(d['duration_seconds'],d['duration_budget_seconds']),
                position_tolerance_m=d['position_tolerance_m'],orientation_tolerance_rad=d['orientation_tolerance_rad'])
            if basis==INTENT_BASIS:
                values=canonical[(task,key)];p,r,_=interaction._fit_completed_curve(values,np.array([0.,1.]))
                t['completion_geometry_xyzw']=Pose(p[-1],r[-1]).xyzw().tolist();basis_rules.append(rule_sha)
            templates.append(t)
        metadata=dict(schema=interaction.LIBRARY_SCHEMA,task_name=task,training_complete=True,fit_split='train',selection_split='train',TEST_used_for_fitting=False,synthetic=False,
            TRAIN_source_ids=sorted(s.source_id for s in rows),TRAIN_parent_source_groups=sorted({s.grouping_source for s in rows}),
            TRAIN_source_manifest_path=str(manifest),TRAIN_source_manifest_sha256=file_sha(manifest),
            fitting_module_sha256=file_sha(interaction.__file__),task_spec_module_sha256=file_sha(ROOT/'our_mani/multitask/specs.py'),
            fit_configuration=asdict(fit_config),development_support_override=False,formal_comparison_ready=False,
            source_grouped_lag_selection=fit['phase_fits'],completed_skill_fitting=fit,
            fitted_phase_order=[p.name for p in interaction.get_task_spec(task).phase_skills],omitted_optional_phases=[],
            command_goal_reference_derivation=dict(helper_sha256=file_sha(goal.__file__),rule_sha256_values=sorted(set(basis_rules)),
                qualified_extension=dict(source=pointer(ROOT/'our_mani/multitask/remove_command_goal_candidates_v46.py'),
                    export_source=pointer(Path(__file__)),source_goal_audit=pointer(audit_path),current_prefit_proof=pointer(prefit_path))),
            public_completion_verifier_calibrated=False,all_original_failed_censored_sources_retained=True,
            wp0_completion_is_prefix_not_phase_endpoint=True,original_rule_caps_changed=False)
        library=interaction.InteractionLibrary(dict(metadata=metadata,templates=templates));path=out/task/'interaction.json';library.save(path)
        proof=dict(schema=SCHEMA,task=task,source_goals_reconstructed_passed=True,original_parents=100,all_numeric_partitions_passed=True,
            all_original_censored_failed_parents_preserved=True,interaction_library=pointer(path),source_goal_audit=pointer(audit_path),
            current_full_curve_fit=pointer(fit_path),current_prefit_proof=pointer(prefit_path),
            fitter=pointer(ROOT/'our_mani/multitask/remove_command_goal_candidates_v46.py'),exporter=pointer(Path(__file__)),
            formal_release=False,native_success_claimed=False,causal_runtime_role_selection_requires_native_smoke=True,TEST_read=False)
        proof_path=out/task/'qualification.json';atomic_json(proof_path,proof);results[task]=dict(interaction=pointer(path),qualification=pointer(proof_path))
    atomic_json(out/'report.json',dict(complete=True,tasks=results,original_parents=200,models_for_TRAIN_executor_development_only=True,formal_release=False,TEST_read=False))
    print(json.dumps(dict(complete=True,tasks=list(results),original_parents=200,formal_release=False)),flush=True)


if __name__=='__main__':main()
