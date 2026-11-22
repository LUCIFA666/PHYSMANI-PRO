"""Typed old200 + periodic100 operation fit, retaining all100 real parents."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import artifact_fitting as adapter,interaction,acquisition_lineage as lineage
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import checked_sha,verify_inputs

TASK='place_cups_on_rotating_frame_high_speed';B=ROOT/'artifacts/framework/multitask_r10'
OUT=B/'high_cup_combined300_operation_fit_v115'


def main():
    if OUT.exists():raise FileExistsError('preserve previous current fit or failure')
    OUT.mkdir();files={};started=time.perf_counter()
    def used(path,expected=None):
        p=Path(path);p=p if p.is_absolute() else ROOT/p;p=p.resolve()
        checksum=file_sha(p) if expected is None else checked_sha(p,expected)[0]
        if str(p) in files:assert files[str(p)]==checksum
        files[str(p)]=checksum;return p
    def ptr(p):return dict(path=str(Path(p).resolve().relative_to(ROOT)),sha256=file_sha(p))
    try:
        strict_path=used(B/'high_cup_periodic_complete_audit_v114/strict_raw100_audit.json')
        strict=json.loads(strict_path.read_text());assert strict['audit_passed'] and strict['complete']
        verify_inputs(ROOT,strict['verified_files_sha256']);files.update(strict['verified_files_sha256'])
        periodic_path=used(B/'TRAIN_high_cup_periodic10_all100_candidate_v2/manifest.json')
        periodic=json.loads(periodic_path.read_text());assert periodic==strict['acquisition_projection']
        old_fit_path=used(B/'TRAIN_artifacts_full_goalv6_semanticv9_3100_v1/TRAIN_fit_manifest.json')
        old_fit=json.loads(old_fit_path.read_text());old_pointer=old_fit['collector_manifest']
        old_path=used(old_pointer['path'],old_pointer['sha256'])
        print(json.dumps(dict(stage='verify_original_typed3100_lineage',new_periodic100_complete=True)),flush=True)
        old_manifest,old_proof=lineage.load_acquisition_manifest(ROOT,old_path)
        assert old_proof is not None;verify_inputs(ROOT,old_proof['verified_files_sha256']);files.update(old_proof['verified_files_sha256'])
        selected=[];entry_by_key={}
        for manifest,kind,proof in ((old_manifest,'original_typed_bundle',old_proof),(periodic,'new_periodic10_raw100',strict)):
            for entry in manifest['entries']:
                if entry['task']==TASK and entry.get('interaction_eligible',True):
                    assert entry['source_kind']=='TRAIN_native_generated_augmentation'
                    key=(entry['task'],entry['raw_episode_id']);assert key not in entry_by_key
                    entry_by_key[key]=(entry,kind)
            for row in manifest['case_receipts']:
                p=used(row['path'],row['sha256']);record=json.loads(p.read_text());entry=record['entry'];key=(entry['task'],entry['raw_episode_id'])
                if key not in entry_by_key or entry['task']!=TASK or not entry.get('interaction_eligible',True):continue
                assert entry==entry_by_key[key][0]
                if kind=='original_typed_bundle':
                    assert record['plan_sha256']==lineage.actual_case_plan_sha(manifest,proof,record['case_id'])
                else:assert record['plan_sha256']==periodic['plan_sha256']
                selected.append((row,record,kind))
        assert len(selected)==300 and Counter(record['entry']['behavior_replica'] for _,record,_ in selected)==Counter({0:100,1:100,2:100})
        sources=[];entries=[];new_native=Counter();new_support=defaultdict(set)
        for row,record,kind in selected:
            entry=record['entry'];outputs=record['files_sha256']
            for name,checksum in outputs.items():used(name,checksum)
            cache=used(entry['cache'],entry['cache_sha256']);command_name=entry['teacher_command_receipt']
            commands=used(command_name,outputs[command_name]);raw=json.loads(commands.read_text())
            with np.load(cache,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
            source,_=adapter.interaction_source(dict(path=ROOT/row['path'],record=record,entry=entry,arrays=arrays,
                command_receipt=raw,command_receipt_verified=True,command_receipt_sha256=outputs[command_name]))
            sources.append(source);entries.append(entry)
            if kind=='new_periodic10_raw100':
                success=record['alignment']['native_success_latched'];assert isinstance(success,bool);new_native[success]+=1
                for event in source.completion_events:new_support[(event.phase,event.reference_mode)].add(source.grouping_source)
        roster_path=used(ROOT/'artifacts/framework/protocol/train_manifest.json')
        expected={'train:'+TASK+':'+row['raw_episode_id'] for row in json.loads(roster_path.read_text())['episodes'] if row['task']==TASK}
        assert len(sources)==300 and {s.grouping_source for s in sources}==expected and len(expected)==100
        config_path=used(B/'command_goal_reference_rule_v6_counted_final_setup_v1/artifact_fit_configuration_full3100.json')
        config=interaction.InteractionFitConfig(**json.loads(config_path.read_text())['interaction'])
        for p in (Path(__file__),Path(adapter.__file__),Path(interaction.__file__),Path(lineage.__file__)):used(p)
        manifest=OUT/'TRAIN_combined_source_manifest.json'
        atomic_json(manifest,dict(schema='r10_highcup_explicit_old200_plus_periodic100_operation_view_v115',entries=entries,
            typed_original_bundle=ptr(old_path),strict_periodic100_proof=ptr(strict_path),periodic100_manifest=ptr(periodic_path),
            original_real_parents=100,teacher_acquisitions=300,all_original_failed_censored_sources_retained=True,
            periodic_policy_not_exact_author_reproduction=True,context_primary_view_modified=False,
            verified_files_sha256=files,VAL_used=False,TEST_used=False))
        summary=dict(new_periodic100_native_successes=new_native[True],new_periodic100_native_failures=new_native[False],
            native_success_field='alignment.native_success_latched',new_periodic_typed_completed_support={':'.join(k):len(v) for k,v in new_support.items()},
            supersedes_v114_summary_only=True,v114_strict_raw_audit_remains_valid=True,
            v114_top_level_success_field_absent_zero_invalid=True,v114_reference_mode_alias_unknown_superseded=True)
        atomic_json(OUT/'corrected_periodic100_summary.json',summary)
        print(json.dumps(dict(stage='fit_all300_teacher_acquisitions',real_parents=100,**summary)),flush=True)
        library=None
        try:
            library=interaction.fit_interaction_library(sources,config,TRAIN_source_manifest_path=manifest)
            fit=library.metadata['completed_skill_fitting'];model=OUT/'interaction.json';library.save(model)
        except interaction.InteractionFittingPending as e:fit=e.diagnostics
        fit_path=OUT/'current_completed_skill_fit.json';atomic_json(fit_path,fit);verify_inputs(ROOT,files)
        phases={k:{x:v.get(x) for x in ('passed','reason','completed_parent_count','completed_attempt_count',
            'position_tolerance_m','orientation_tolerance_rad','duration_seconds','duration_budget_seconds')} for k,v in fit['phase_fits'].items()}
        atomic_json(OUT/'report.json',dict(schema='r10_highcup_all300_actual_operation_fit_v115',created_utc=datetime.now(timezone.utc).isoformat(),
            actual_original_parents=100,teacher_acquisitions=300,source_manifest=ptr(manifest),strict_raw100_audit=ptr(strict_path),
            corrected_new_periodic_summary=summary,model=None if library is None else ptr(model),
            phases=phases,required_phase_failures=fit['required_phase_failures'],failed_fit_keys=fit['failed_fit_keys'],
            detailed_fit=ptr(fit_path),all_original_sources_and_raw_labels_retained=True,original_gates_not_loosened=True,
            independent_augmentation_count_not_claimed_as_parent_count=True,native_started=False,GPU_started=False,formal_release=False,
            verified_files_sha256=files,seconds=time.perf_counter()-started))
        print(json.dumps(dict(fit_completed=True,model_exported=library is not None,phases=phases,formal_release=False)),flush=True)
    except BaseException as e:
        atomic_json(OUT/'preparation_failed.json',dict(type=type(e).__name__,message=str(e),
            original_sources_not_modified=True,native_started=False,GPU_started=False));raise


if __name__=='__main__':main()
