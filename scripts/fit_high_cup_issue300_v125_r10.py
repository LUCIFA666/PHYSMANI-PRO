"""Audit all300 actual acquisitions, then fit original transport issue intent."""
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import replace
import json,sys,time,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import high_cup_issue_goal_v122 as derive,artifact_fitting as adapter,interaction
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs

B=ROOT/'artifacts/framework/multitask_r10';OUT=B/'high_cup_all300_issue_operation_v125'


def ptr(path):
    path=Path(path).resolve();return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))


def main():
    if OUT.exists():raise FileExistsError('preserve previous fit or failure')
    OUT.mkdir();start=time.perf_counter()
    try:
        rule_path=OUT/'explicit_transport_issue_rule.json'
        atomic_json(rule_path,dict(schema=derive.SCHEMA,TRAIN_only=True,VAL_used=False,TEST_used=False,
            tasks={derive.TASK:{'transport':'original_issue_pose'}},helper=ptr(derive.__file__),
            original_intent_rule=ptr(B/'command_goal_reference_rule_v6_counted_final_setup_v1/command_goal_rule.json'),
            combined_source_manifest=ptr(B/'high_cup_combined300_operation_fit_v116/TRAIN_combined_source_manifest.json'),
            original_fit_gates=dict(minimum_completed_parent_sources=8,maximum_position_tolerance_m=.03,
                maximum_orientation_tolerance_rad=.15)))
        print(json.dumps(dict(stage='reverify_actual_typed_old200_and_periodic100')),flush=True)
        rule=derive.load_rule(ROOT,ptr(rule_path));files=dict(rule.files);sources=[];rejections=[]
        for index,(name,row) in enumerate(rule.cases.items()):
            path=Path(name);record=json.loads(path.read_text());entry=record['entry']
            with np.load(ROOT/entry['cache'],allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
            raw=json.loads((ROOT/entry['teacher_command_receipt']).read_text())
            case=dict(path=path,record=record,entry=entry,arrays=arrays,command_receipt=raw,
                command_receipt_verified=True,command_receipt_sha256=record['files_sha256'][entry['teacher_command_receipt']])
            source,_=adapter.interaction_source(case)
            if index==0:
                changed_record=copy.deepcopy(record);changed_record['alignment']['native_success_latched']=not record['alignment']['native_success_latched']
                changed_arrays=dict(arrays);changed_arrays['native_pose_valid']=arrays['native_pose_valid'].copy()
                changed_arrays['native_pose_valid'][0,0]=not bool(arrays['native_pose_valid'][0,0])
                changed_tcp=source.measured_tcp_xyzw.copy();changed_tcp[0,0]+=.001
                for label,bad_case,bad_source in (
                    ('canonical_success_label_tamper',dict(case,record=changed_record),source),
                    ('raw_native_mask_tamper',dict(case,arrays=changed_arrays),source),
                    ('measured_curve_tamper',case,replace(source,measured_tcp_xyzw=changed_tcp)),
                    ('unregistered_plan_alias',dict(case,path=path.parent/'other_case.json'),source)):
                    try:derive.derive_source(bad_case,bad_source,rule)
                    except ValueError as e:rejections.append(dict(case=ptr(path),test=label,rejected=True,reason=str(e)))
                    else:raise AssertionError('actual raw tamper was not rejected: '+label)
            sources.append(derive.derive_source(case,source,rule))
            if (index+1)%100==0:print(json.dumps(dict(stage='derive_original_issue_intent',acquisitions=index+1)),flush=True)
        atomic_json(OUT/'actual_raw_negative_checks.json',dict(checks=rejections,all_rejected=True,
            original_files_modified=False,native_started=False,GPU_started=False))
        for p in (Path(__file__),Path(derive.__file__),Path(adapter.__file__),Path(interaction.__file__)):
            files[str(p.resolve())]=file_sha(p)
        config_path=B/'command_goal_reference_rule_v6_counted_final_setup_v1/artifact_fit_configuration_full3100.json'
        files[str(config_path.resolve())]=file_sha(config_path)
        config=interaction.InteractionFitConfig(**json.loads(config_path.read_text())['interaction'])
        assert (config.minimum_completed_parent_sources,config.maximum_position_tolerance_m,config.maximum_orientation_tolerance_rad)==(8,.03,.15)
        manifest=OUT/'TRAIN_source_manifest.json'
        atomic_json(manifest,dict(schema=derive.SCHEMA,entries=[r['entry'] for r in rule.cases.values()],
            explicit_rule=ptr(rule_path),actual_original_parents=100,teacher_acquisitions=300,
            all_original_failed_censored_sources_commands_events_masks_retained=True,
            exact_periodic_policy_identity_preserved=True,context_primary_view_modified=False,
            source_curation_by_outcome=False,verified_files_sha256=files,VAL_used=False,TEST_used=False))
        model=None
        try:
            library=interaction.fit_interaction_library(sources,config,TRAIN_source_manifest_path=manifest)
            fit=library.metadata['completed_skill_fitting'];model=OUT/'interaction.json';library.save(model)
            assert interaction.InteractionLibrary.load(model).sha256==library.sha256
        except interaction.InteractionFittingPending as e:fit=e.diagnostics
        fit_path=OUT/'completed_skill_fit.json';atomic_json(fit_path,fit);verify_inputs(ROOT,files)
        phases={k:{x:v.get(x) for x in ('passed','reason','completed_parent_count','completed_attempt_count',
            'position_tolerance_m','orientation_tolerance_rad','duration_seconds','duration_budget_seconds')} for k,v in fit['phase_fits'].items()}
        report=dict(schema=derive.SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),model=None if model is None else ptr(model),
            manifest=ptr(manifest),actual_original_parents=100,teacher_acquisitions=300,phases=phases,
            detailed_fit=ptr(fit_path),required_phase_failures=fit['required_phase_failures'],failed_fit_keys=fit['failed_fit_keys'],
            actual_raw_negative_checks=ptr(OUT/'actual_raw_negative_checks.json'),
            original_numeric_fitter_and_gates_unchanged=True,original_raw_completion_metrics_preserved=True,
            transport_issue_intent_is_not_actual_future_dynamics=True,old_rule_and_raw_sources_not_modified=True,
            verified_files_sha256=files,native_started=False,GPU_started=False,formal_release=False,seconds=time.perf_counter()-start)
        atomic_json(OUT/'report.json',report)
        print(json.dumps(dict(fit_complete=True,model_exported=model is not None,phases=phases)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'preparation_failed.json',dict(type=type(error).__name__,message=str(error),
            native_started=False,GPU_started=False,old_sources_modified=False));raise


if __name__=='__main__':main()
