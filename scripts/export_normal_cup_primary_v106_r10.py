"""Export an explicit initial-sequence cup policy with full original evidence."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import interaction,artifact_fitting,release_preflight
from our_mani.multitask import normal_cup_primary_export_v105 as exporter
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import checked_sha,verify_inputs

B=ROOT/'artifacts/framework/multitask_r10';OUT=B/'normal_cup_primary_policy_v106'


def main():
    if OUT.exists():raise FileExistsError('preserve prior exported or failed artifact')
    OUT.mkdir();files={}
    def used(path,expected=None):
        path=Path(path);path=path if path.is_absolute() else ROOT/path
        path=path.resolve();checksum=file_sha(path) if expected is None else checked_sha(path,expected)[0]
        name=str(path)
        if name in files:assert files[name]==checksum
        files[name]=checksum;return path
    def pointer(path):
        path=Path(path).resolve();return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))
    def loadptr(p):return used(p['path'],p['sha256'])
    try:
        prior=used(B/'command_goal_actual700_full_curve_CPU_draft_v1/report.json')
        producer=json.loads(prior.read_text());verify_inputs(ROOT,producer['verified_files_sha256'])
        files.update(producer['verified_files_sha256'])
        detail=loadptr(producer['tasks'][exporter.TASK]['actual_full_curve_fit']);draft=json.loads(detail.read_text())
        production_path=used(B/'TRAIN_artifacts_full_goalv6_semanticv9_3100_v1/report.json')
        production=json.loads(production_path.read_text())['per_task'][exporter.TASK]['completed_skill_fitting']
        config_path=used(B/'command_goal_reference_rule_v6_counted_final_setup_v1/artifact_fit_configuration_full3100.json')
        config=interaction.InteractionFitConfig(**json.loads(config_path.read_text())['interaction'])
        templates,scope=exporter.admit_templates(draft,production,config)
        case_path=used(B/'full14_pending_reference_readonly_diagnostic_v6'/exporter.TASK/'cases.json')
        case_rows=json.loads(case_path.read_text());sources=[];entries=[]
        for row in case_rows:
            path=loadptr(row['canonical_receipt']);record=json.loads(path.read_text())
            cache=loadptr(row['cache']);commands=loadptr(row['command_receipt'])
            with np.load(cache,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
            source,_=artifact_fitting.interaction_source(dict(path=path,record=record,entry=record['entry'],
                arrays=arrays,command_receipt=json.loads(commands.read_text()),command_receipt_verified=True,
                command_receipt_sha256=row['command_receipt']['sha256']))
            sources.append(source);entries.append(record['entry'])
        roster_path=used(ROOT/'artifacts/framework/protocol/train_manifest.json')
        roster=json.loads(roster_path.read_text())['episodes']
        expected={'train:'+exporter.TASK+':'+row['raw_episode_id'] for row in roster if row['task']==exporter.TASK}
        assert len(sources)==len({s.grouping_source for s in sources})==len(expected)==100
        assert {s.grouping_source for s in sources}==expected
        support={}
        for template in templates:
            key,parents,attempts=release_preflight._completion_support(sources,template)
            label='|'.join(key);original=production['phase_fits'][label]
            assert parents==set(original['completed_parent_sources'])
            assert len(attempts)==original['completed_attempt_count'] and len(parents)>=8
            support[label]=dict(independent_raw_completed_parents=len(parents),independent_raw_completed_attempts=len(attempts))
        for path in (Path(__file__),Path(exporter.__file__),Path(interaction.__file__),
                Path(artifact_fitting.__file__),Path(release_preflight.__file__),ROOT/'tests/test_normal_cup_primary_export_v105.py'):
            used(path)
        manifest=OUT/'TRAIN_source_manifest.json'
        atomic_json(manifest,dict(schema=exporter.SCHEMA,entries=entries,case_proofs=case_rows,
            all100_original_sources_retained=True,all_original_commands_events_and_failures_retained=True,
            source_domain_admission=scope,independent_actual_completion_support=support,
            original_producer=pointer(prior),original_full_numeric_report=pointer(detail),
            original_production_diagnostic=pointer(production_path),
            original_full_partition_gate_passed=False,verified_inputs_sha256=files,
            native_started=False,GPU_started=False,formal_VAL_TEST=False))
        proof=dict(actual_numeric_curve_producer='command_goal_reference.fit_diagnostic + unchanged interaction curve solver',
            original_producer=pointer(prior),original_curve_report=pointer(detail),
            original_production_report=pointer(production_path),
            admission_source=pointer(Path(exporter.__file__)),export_source=pointer(Path(__file__)),
            original_full_scope_fit_pending_preserved=True,new_numeric_refit_performed=False)
        value=exporter.payload(templates,scope,sources=sources,manifest=manifest,fit_configuration=config,
            production_report=production,export_proof=proof)
        library=interaction.InteractionLibrary(value);model=OUT/'interaction.json';library.save(model)
        loaded=interaction.InteractionLibrary.load(model);assert loaded.sha256==library.sha256
        assert all(t.reference_mode!='released' for t in loaded.templates)
        verify_inputs(ROOT,files)
        atomic_json(OUT/'report.json',dict(schema=exporter.SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),
            task=exporter.TASK,model=pointer(model),manifest=pointer(manifest),policy_scope=scope,
            independent_raw_completed_support=support,old_full_scope_gate_passed=False,
            all_original_sources_and_labels_retained=True,phases=list(library.phase_order),
            tests_each_environment=5,original_numeric_kernel_reused=True,verified_files_sha256=files,
            original_support_and_error_caps_not_loosened=True,
            native_started=False,GPU_started=False,formal_release=False))
        print(json.dumps(dict(exported=True,phases=list(library.phase_order),independent_support=support,
            old_full_scope_gate_passed=False,native_started=False)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'preparation_failed.json',dict(type=type(error).__name__,message=str(error),
            native_started=False,GPU_started=False,old_inputs_modified=False))
        raise


if __name__=='__main__':main()
