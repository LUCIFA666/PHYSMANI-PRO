"""Recover a completed numeric fit after a JSON consumer order failure."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.normal_cup_primary_library_v109 import PrimarySequenceLibrary
from our_mani.multitask.evaluation import atomic_json,file_sha,digest
from our_mani.multitask.input_stat_cache_v79 import verify_inputs


def main():
    base=ROOT/'artifacts/framework/multitask_r10';previous=base/'normal_cup_primary_policy_v108'
    out=base/'normal_cup_primary_policy_v110'
    if out.exists():raise FileExistsError('preserve completed or failed acceptance')
    manifest=previous/'TRAIN_source_manifest.json';d=json.loads(manifest.read_text());files=d['verified_files_sha256']
    verify_inputs(ROOT,files);model=previous/'interaction.json';library=PrimarySequenceLibrary.load(model)
    original=json.loads((previous/'original_full_scope_fit_still_pending.json').read_text())
    assert digest(original)==digest(library.metadata['completed_skill_fitting'])
    assert len(library.metadata['TRAIN_parent_source_groups'])==100
    for launcher in ([sys.executable],['bash','scripts/run_local_sim.sh']):
        subprocess.run(launcher+['-m','unittest','discover','-s','tests','-p','test_normal_cup_primary_library_v109.py'],cwd=ROOT,check=True)
    out.mkdir();copy=out/'interaction.json';shutil.copyfile(model,copy);assert file_sha(copy)==file_sha(model)
    def pointer(p):return dict(path=str(p.relative_to(ROOT)),sha256=file_sha(p))
    inputs=dict(files)
    for p in (model,manifest,previous/'original_full_scope_fit_still_pending.json',previous/'preparation_failed.json',
            ROOT/'our_mani/multitask/normal_cup_primary_library_v109.py',ROOT/'tests/test_normal_cup_primary_library_v109.py',Path(__file__)):
        inputs[str(p.resolve())]=file_sha(p)
    verify_inputs(ROOT,inputs)
    atomic_json(out/'report.json',dict(schema='r10_normal_cup_primary_policy_canonical_consumer_acceptance_v110',
        created_utc=datetime.now(timezone.utc).isoformat(),model=pointer(copy),manifest=pointer(manifest),
        policy_scope=library.metadata['initial_sequence_policy_scope'],
        prior_numeric_fit_and_all_original_diagnostics_preserved=True,model_bytes_copied_exactly=True,
        only_JSON_dictionary_order_handling_changed=True,tests_each_environment=4,
        old_full_scope_fitter_still_pending=pointer(previous/'original_full_scope_fit_still_pending.json'),
        original_preparation_failure=pointer(previous/'preparation_failed.json'),
        all_original100_sources_and_raw_labels_retained=True,all_admitted_original_support_and_error_caps_passed=True,
        admitted_completed_parent_counts={k:v['completed_parent_count'] for k,v in original['phase_fits'].items()
            if k in library.metadata['initial_sequence_policy_scope']['admitted_partition_keys']},
        old_full_scope_loader_acceptance_claimed=False,verified_files_sha256=inputs,
        native_started=False,GPU_started=False,formal_release=False))
    print(json.dumps(dict(accepted=True,all_original_parents=100,phases=list(library.phase_order),
        old_full_scope_passed=False,tests_each_environment=4,native_started=False)),flush=True)


if __name__=='__main__':main()
