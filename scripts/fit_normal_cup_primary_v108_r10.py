"""Refit all100 current raw sources under a separately declared policy domain."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask import interaction,artifact_fitting,release_preflight
from our_mani.multitask import normal_cup_primary_fitting_v107 as fitting
from our_mani.multitask.evaluation import atomic_json,file_sha,digest
from our_mani.multitask.input_stat_cache_v79 import checked_sha,verify_inputs

B=ROOT/'artifacts/framework/multitask_r10';OUT=B/'normal_cup_primary_policy_v108'


def main():
    if OUT.exists():raise FileExistsError('preserve previous model or failed preparation')
    OUT.mkdir();files={};started=time.perf_counter()
    def used(path,expected=None):
        p=Path(path);p=p if p.is_absolute() else ROOT/p;p=p.resolve()
        value=file_sha(p) if expected is None else checked_sha(p,expected)[0]
        if str(p) in files:assert files[str(p)]==value
        files[str(p)]=value;return p
    def ptr(path):
        p=Path(path).resolve();return dict(path=str(p.relative_to(ROOT)),sha256=file_sha(p))
    def loadptr(p):return used(p['path'],p['sha256'])
    try:
        cases_path=used(B/'full14_pending_reference_readonly_diagnostic_v6'/fitting.TASK/'cases.json')
        rows=json.loads(cases_path.read_text());sources=[];entries=[]
        for row in rows:
            path=loadptr(row['canonical_receipt']);record=json.loads(path.read_text())
            cache=loadptr(row['cache']);command=loadptr(row['command_receipt'])
            with np.load(cache,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
            source,_=artifact_fitting.interaction_source(dict(path=path,record=record,entry=record['entry'],arrays=arrays,
                command_receipt=json.loads(command.read_text()),command_receipt_verified=True,
                command_receipt_sha256=row['command_receipt']['sha256']))
            sources.append(source);entries.append(record['entry'])
        roster_path=used(ROOT/'artifacts/framework/protocol/train_manifest.json')
        expected={'train:'+fitting.TASK+':'+row['raw_episode_id'] for row in json.loads(roster_path.read_text())['episodes'] if row['task']==fitting.TASK}
        assert len(sources)==len(expected)==100 and {s.grouping_source for s in sources}==expected
        config_path=used(B/'command_goal_reference_rule_v6_counted_final_setup_v1/artifact_fit_configuration_full3100.json')
        config=interaction.InteractionFitConfig(**json.loads(config_path.read_text())['interaction'])
        for p in (Path(__file__),Path(fitting.__file__),Path(interaction.__file__),
                Path(artifact_fitting.__file__),Path(release_preflight.__file__),ROOT/'our_mani/multitask/specs.py'):
            used(p)
        manifest=OUT/'TRAIN_source_manifest.json'
        atomic_json(manifest,dict(schema=fitting.SCHEMA,entries=entries,case_proofs=rows,
            declared_policy_domain='initial_sequence_no_released_cup_recovery',
            all_original100_failed_censored_and_rare_mode_sources_retained=True,
            all_original_completion_events_commands_masks_unmodified=True,
            verified_files_sha256=files,native_started=False,GPU_started=False,formal_VAL_TEST=False))
        print(json.dumps(dict(stage='current_original_full_scope_fit',original_parents=100)),flush=True)
        try:interaction.fit_interaction_library(sources,config,TRAIN_source_manifest_path=manifest)
        except interaction.InteractionFittingPending as e:
            original=e.diagnostics
        else:raise AssertionError('expected old insufficient released-recovery support must remain visible')
        atomic_json(OUT/'original_full_scope_fit_still_pending.json',original)
        print(json.dumps(dict(stage='explicit_initial_sequence_fit',original_failed_keys=original['failed_fit_keys'])),flush=True)
        library=fitting.fit_primary_interactions(sources,config,TRAIN_source_manifest_path=manifest,synthetic=False)
        # Exact full original diagnostics must survive the domain change:
        # all raw curation, failed partitions, parent CV numbers and censoring.
        assert digest(library.metadata['completed_skill_fitting'])==digest(original)
        supports={}
        for template in library.templates:
            key,parents,attempts=release_preflight._completion_support(sources,template);label='|'.join(key)
            row=original['phase_fits'][label]
            assert parents==set(row['completed_parent_sources']) and len(attempts)==row['completed_attempt_count']
            supports[label]=dict(completed_parents=len(parents),completed_attempts=len(attempts))
        model=OUT/'interaction.json';library.save(model)
        assert fitting.PrimarySequenceLibrary.load(model).sha256==library.sha256
        try:interaction.InteractionLibrary.load(model)
        except interaction.InteractionArtifactError:pass
        else:raise AssertionError('old full-scope loader must not silently accept a distinct policy domain')
        verify_inputs(ROOT,files)
        atomic_json(OUT/'report.json',dict(schema=fitting.SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),
            task=fitting.TASK,model=ptr(model),manifest=ptr(manifest),policy_scope=library.metadata['initial_sequence_policy_scope'],
            actual_original_numeric_fitter_invoked=True,actual_new_domain_fitter_invoked=True,
            complete_original_diagnostics_and_all_failed_partitions_identical=True,
            original_full_scope_fit_still_pending=ptr(OUT/'original_full_scope_fit_still_pending.json'),
            independent_raw_completed_support=supports,old_loader_refuses_new_schema=True,
            all_original100_parents_and_raw_labels_retained=True,original_gates_not_loosened=True,
            native_started=False,GPU_started=False,formal_release=False,verified_files_sha256=files,
            seconds=time.perf_counter()-started))
        print(json.dumps(dict(fitted=True,phases=list(library.phase_order),supports=supports,
            full_original_gate_passed=False,native_started=False,seconds=time.perf_counter()-started)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'preparation_failed.json',dict(type=type(error).__name__,message=str(error),
            native_started=False,GPU_started=False,old_inputs_not_modified=True))
        raise


if __name__=='__main__':main()
