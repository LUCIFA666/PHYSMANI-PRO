"""Fit the independently versioned RGB-q90 TRAIN candidate; no native launch."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.colour_only_candidate import (verify_native_colour_cases,
    load_visual_patch,bound,pointer,sha,read,write)
from our_mani.multitask.artifact_fitting import colour_samples
from our_mani.multitask.public_roles import ColourFitConfig
from our_mani.multitask.colour_distribution import (FIXED_CONFIG,FEATURE,fit_prototypes,
    parent_crossfit,fit_candidate,colour_descriptor,DevelopmentQuantileRoleResolver)
from our_mani.multitask.specs import get_task_spec


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    args=parser.parse_args();root=ROOT;base=root/'artifacts/framework/multitask_r10';out=(root/args.out).resolve()
    if not out.is_relative_to(base) or out.exists():raise ValueError('fresh desktop artifact namespace required')
    out.mkdir(parents=True);started=datetime.now(timezone.utc).isoformat()
    original=base/'TRAIN_colour_only_visual_supplement_actual_candidate_v1'
    study=base/'TRAIN_colour_distribution_feature_study_v1'
    proof=base/'TRAIN_primary3000_counted_highcup100_bundle_v1/bundle_audit.json'
    annotation=base/'TRAIN_colour_visual_patch_annotation_candidate_v1/annotation.json'
    peer=base/'TRAIN_colour_visual_patch_independent_peer_v1/report.json'
    evidence=dict(native_acquisition_proof=pointer(root,proof),
        native_colour_ledger=pointer(root,original/'actual_colour_input_ledger.json'),
        visual_annotation=pointer(root,annotation),visual_peer=pointer(root,peer),
        original_parent_assignment=pointer(root,original/'TRAIN_parent_crossfit.json'),
        feature_study=pointer(root,study/'report.json'))
    source_paths=[Path(__file__),root/'our_mani/multitask/colour_distribution.py',
        root/'our_mani/multitask/public_roles.py',root/'our_mani/multitask/colour_only_candidate.py',
        root/'our_mani/multitask/artifact_fitting.py',root/'our_mani/multitask/specs.py']
    bindings=[pointer(root,path) for path in source_paths]+list(evidence.values())+[
        pointer(root,original/'colour_only_manifest.json'),pointer(root,study/'q90_rgb_lab.json')]
    write(out/'source_bindings_before.json',bindings)
    for ref in bindings:
        path=bound(root,ref);destination=out/'sources'/ref['sha256']/path.name
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(path.read_bytes())
    cases,ledger,parents=verify_native_colour_cases(root,proof)
    if ledger!=read(original/'actual_colour_input_ledger.json')['entries']:raise ValueError('native colour source ledger changed')
    config=ColourFitConfig(**FIXED_CONFIG);samples,excluded=colour_samples(root,cases,config)
    native_count=len(samples);visual,_=load_visual_patch(root,annotation,parents,peer);samples.append(visual)
    assignment=read(original/'TRAIN_parent_crossfit.json')['parent_to_fold']
    crossfit=parent_crossfit(samples,config,assignment);write(out/'parent_crossfit.json',crossfit)
    candidate=fit_candidate(samples,config,root=root,
        TRAIN_manifest=pointer(root,original/'colour_only_manifest.json'),evidence=evidence,
        crossfit=pointer(root,out/'parent_crossfit.json'))
    write(out/'candidate.json',candidate)
    previous=read(study/'q90_rgb_lab.json');prototypes=fit_prototypes(samples,config)
    centre_equal=all(np.array_equal(prototypes[name]['prototype_lab_normalised'],row['centre'])
        and prototypes[name]['acceptance_radius']==row['acceptance_radius']
        and prototypes[name]['TRAIN_parent_source_groups']==row['TRAIN_parent_sources']
        for name,row in previous['prototypes'].items())
    descriptors=np.stack([colour_descriptor(s.rgb_pixels_255) for s in samples])
    feature_sha=hashlib.sha256(np.ascontiguousarray(descriptors,dtype='<f8').tobytes()).hexdigest()
    replay=dict(feature_matrix_exact_equal=feature_sha==previous['feature_matrix_sha256'],
        full_prototypes_and_radii_exact_equal=centre_equal,
        all_heldout_predictions_exact_equal=crossfit['held_predictions']==previous['held_predictions'],
        all_parent_scores_exact_equal=crossfit['held_parent_scores']==previous['held_parent_scores'],
        original_native_role_component_map_exact_equal=candidate['role_components']==
            read(original/'candidate.json')['native_role_components'],feature_matrix_sha256=feature_sha)
    if not all(value for key,value in replay.items() if key.endswith('equal')):raise ValueError('independent module drifted from actual feature study')
    for task in ('insert_onto_rotating_peg','insert_onto_rotating_peg_high_speed',
                 'push_moving_button','push_moving_button_high_speed'):
        DevelopmentQuantileRoleResolver(get_task_spec(task),candidate,root=root,
            shared_semantic_permission=True,allow_development_candidate=True)
    patches={name:next(s.rgb_pixels_255 for s in samples if s.colour_name==name) for name in prototypes}
    np.savez_compressed(out/'representative_TRAIN_patches.npz',**patches)
    write(out/'independent_module_replay.json',replay)
    for ref in bindings:bound(root,ref)
    for row in ledger:
        for key in ('canonical','raw_child','colour_sidecar'):bound(root,row[key])
    report=dict(schema='r10_TRAIN_RGB_q90_colour_candidate_actual_fit_report_v1',
        started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),
        TRAIN_only=True,VAL_used=False,TEST_used=False,native_GPU_or_new_collection_started=False,
        native_acquisitions_reverified=3100,native_colour_samples=native_count,visual_colour_samples=1,
        native_colour_parents=400,total_colour_parents=len(assignment),classes=len(prototypes),
        all_original_samples_retained=True,excluded_native_samples=excluded,
        original_frozen_parent_assignment_reused=True,accuracy_selected_folds=False,
        feature=FEATURE,fit_configuration=FIXED_CONFIG,crossfit_passed=crossfit['passed'],
        macro_parent_correct_and_supported_fraction=crossfit['macro_parent_correct_and_supported_fraction'],
        required_fraction=.95,full_fit_complete=True,independent_replay=replay,
        all_bound_source_colour_raw_bytes_rehashed_before_after=True,
        unrelated_motion_array_audit_reused_not_rehashed=True,
        native_component_map_unchanged=True,manual_visual_patch_has_no_component_authority=True,
        runtime_integration_not_performed=True,runtime_admission_granted=False,formal_release_issued=False,
        remaining_work=['explicit shared-protocol version and factory/coverage consumer integration after ROOT review',
            'current RGBD development closed-loop verification; no VAL/TEST tuning of feature',
            'formal task-specific admission using actual new-source/model evidence'],
        source_bindings=bindings,outputs={name:pointer(root,out/name) for name in (
            'candidate.json','parent_crossfit.json','independent_module_replay.json','representative_TRAIN_patches.npz')})
    write(out/'report.json',report)
    print(json.dumps(dict(report=pointer(root,out/'report.json'),candidate=pointer(root,out/'candidate.json'),
        crossfit_passed=crossfit['passed'],macro=crossfit['macro_parent_correct_and_supported_fraction'])),flush=True)


if __name__=='__main__':main()
