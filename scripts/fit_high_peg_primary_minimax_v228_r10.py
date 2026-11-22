"""Retain all100 original teachers; test a different TRAIN-only terminal estimator."""
from pathlib import Path
from datetime import datetime,timezone
import json
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask import acquisition_lineage as a,artifact_fitting as adapter,command_goal_reference as goal
from our_mani.multitask import interaction
from our_mani.multitask import high_peg_primary_fitting_v221 as fitter,minimax_terminal_v220 as minimax
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs

BASE=ROOT/'artifacts/framework/multitask_r10'
OUT=BASE/'high_peg_primary_minimax_v228'
TASK='insert_onto_rotating_peg_high_speed'


def main():
    if OUT.exists():raise FileExistsError('preserve completed or failed fit')
    OUT.mkdir();began=time.perf_counter();files={}
    prior_path=BASE/'TRAIN_artifacts_full_goalv6_semanticv9_3100_v1/report.json'
    prior=a.read(a.checked_pointer(ROOT,a.pointer(ROOT,prior_path),files))
    config_path=Path(prior['fitting_configuration']['path'])
    config=a.read(a.checked_pointer(ROOT,prior['fitting_configuration'],files))
    fit=interaction.InteractionFitConfig(**config['interaction'])
    assert fit.minimum_completed_parent_sources==8 and fit.maximum_position_tolerance_m==.03 and fit.maximum_orientation_tolerance_rad==.15
    rule=goal.load_rule(ROOT,config['command_goal_reference_rule'],purpose='production_fitting');files.update(rule.verified_files_sha256)
    old=prior['per_task'][TASK]['completed_skill_fitting']
    proofs=old['command_goal_reference_derivation']['per_source'];assert len(proofs)==100
    sources=[];entries=[];source_audits=[]
    try:
        for index,(source_id,proof) in enumerate(proofs.items(),1):
            raw=proof['raw_bindings'];p=a.checked_pointer(ROOT,raw['canonical_receipt'],files);record=a.read(p);entry=record['entry']
            assert entry['split']=='train' and entry['task']==TASK
            cache=a.checked_pointer(ROOT,raw['cache'],files)
            with np.load(cache,allow_pickle=False) as stored:arrays={key:stored[key].copy() for key in stored.files}
            commands=a.read(a.checked_pointer(ROOT,raw['commands'],files))
            case=dict(path=p,record=record,entry=entry,arrays=arrays,command_receipt=commands,
                command_receipt_verified=True,command_receipt_sha256=raw['commands']['sha256'])
            source,audit=adapter.interaction_source(case,command_goal_rule=rule)
            assert source.source_id==source_id
            sources.append(source);entries.append(dict(entry,source_id=source.source_id))
            evidence=source.command_goal_reference_evidence
            if evidence is not None:files.update(evidence.derivation_proof['verified_files_sha256'])
            source_audits.append(dict(source_id=source.source_id,parent=source.grouping_source,raw_bindings=raw,adapter_audit=audit))
            if index%10==0:atomic_json(OUT/f'source_progress_{index:03d}.json',dict(stage='verify_all_original100_sources',completed=index,complete=False,native_started=False,GPU_started=False))
        assert len(sources)==100 and len({s.grouping_source for s in sources})==100
        for p in [Path(__file__),Path(fitter.__file__),Path(minimax.__file__),Path(interaction.__file__),ROOT/'tests/test_minimax_terminal_v220.py']:
            files[str(p.resolve())]=file_sha(p)
        manifest=OUT/'TRAIN_source_manifest.json'
        atomic_json(manifest,dict(schema='r10_highpeg_all100_primary_minimax_TRAIN_manifest_v228',entries=entries,
            fit_split='train',VAL_used=False,TEST_used=False,source_audits=source_audits,
            original_all_partitions_report=dict(path=str(prior_path.relative_to(ROOT)),sha256=file_sha(prior_path)),
            original_raw_commands_events_masks_and_failures_preserved=True,verified_files_sha256=files))
        files[str(manifest)]=file_sha(manifest)
        model=None
        try:
            library=fitter.fit_primary_interactions(sources,fit,TRAIN_source_manifest_path=manifest,synthetic=False)
            details=library.metadata['completed_skill_fitting'];path=OUT/'interaction.json';library.save(path)
            reloaded=fitter.PrimarySequenceLibrary.load(path);assert reloaded.sha256==library.sha256
            model=dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))
            status='all_initial_sequence_original_gates_passed_new_estimator'
        except interaction.InteractionFittingPending as error:
            details=error.diagnostics;status='pending_original_numeric_gates'
        atomic_json(OUT/'phase_fit_diagnostics.json',details)
        verify_inputs(ROOT,files)
        report=dict(schema='r10_highpeg_original100_primary_minimax_insert_TRAIN_fit_v228',created_utc=datetime.now(timezone.utc).isoformat(),
            stage=status,model=model,manifest=dict(path=str(manifest.relative_to(ROOT)),sha256=file_sha(manifest)),
            original_teacher_parents=100,original_sources=100,
            numeric_estimator_changed_only_for_insert_terminal_translation=True,
            all_terminal_rows_constrained_no_outcome_filtering=True,
            original8_point03_point15_gates_unchanged=True,
            policy_scope='initial sequence; released-ring recovery unconditionally outside new policy',
            original_full_scope_fit_passed_not_claimed=True,
            minimax_solver_audits=fitter.MINIMAX_AUDIT,
            phase_fits={k:{n:v.get(n) for n in ['passed','reason','completed_parent_count','position_tolerance_m','orientation_tolerance_rad']} for k,v in details['phase_fits'].items()},
            required_phase_failures=details['required_phase_failures'],failed_fit_keys=details['failed_fit_keys'],
            verified_files_sha256=files,seconds=time.perf_counter()-began,
            native_started=False,GPU_started=False,formal_VAL_TEST=False)
        atomic_json(OUT/'report.json',report);atomic_json(OUT/'status.json',dict(stage=status,complete=True,model=model))
        print(json.dumps(dict(stage=status,model=model,seconds=report['seconds'])),flush=True)
    except BaseException as error:
        atomic_json(OUT/'preparation_failed.json',dict(type=type(error).__name__,message=str(error),verified_sources=len(sources),native_started=False,GPU_started=False));raise


if __name__=='__main__':main()
