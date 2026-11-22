"""Independent TRAIN pixel-feature study; no native collection/runtime admission."""
from collections import Counter,defaultdict
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.colour_only_candidate import (verify_native_colour_cases,load_visual_patch,
    bound,pointer,sha,read,write)
from our_mani.multitask.artifact_fitting import colour_samples
from our_mani.multitask.public_roles import ColourFitConfig,_lab
from our_mani.multitask.evaluation import digest

FEATURES=('legacy_median_lab','q90_rgb_lab','brightest20pct_median_lab','median_lab_plus_q90_brightness')


def descriptor(pixels,name):
    pixels=np.asarray(pixels,float)
    if pixels.ndim!=2 or pixels.shape[1]!=3 or not len(pixels) or not np.isfinite(pixels).all() or np.any((pixels<0)|(pixels>255)):
        raise ValueError('finite public RGB255 patch required')
    lab=_lab(pixels)
    if name=='legacy_median_lab':return np.median(lab,axis=0)
    if name=='q90_rgb_lab':return _lab(np.quantile(pixels,.9,axis=0)[None])[0]
    brightness=np.max(pixels,axis=1)
    if name=='brightest20pct_median_lab':return np.median(lab[brightness>=np.quantile(brightness,.8)],axis=0)
    if name=='median_lab_plus_q90_brightness':return np.r_[np.median(lab,axis=0),np.quantile(brightness,.9)/255.]
    raise ValueError('unknown declared feature')


def fit(samples,x,config):
    classes=sorted({s.colour_name for s in samples});prototypes={}
    for name in classes:
        indices=[i for i,s in enumerate(samples) if s.colour_name==name]
        parents=sorted({samples[i].source_group for i in indices})
        if len(parents)<config.minimum_sources_per_colour:raise ValueError('insufficient TRAIN parents: '+name)
        centres=np.stack([np.median([x[i] for i in indices if samples[i].source_group==group],axis=0) for group in parents])
        centre=np.median(centres,axis=0);radii=np.linalg.norm(centres-centre,axis=1)
        prototypes[name]=dict(centre=centre.tolist(),acceptance_radius=max(config.minimum_acceptance_radius,
            float(np.quantile(radii,config.within_colour_quantile))),TRAIN_parent_sources=parents)
    nearest=[]
    for name in classes:
        distance,other=min((float(np.linalg.norm(np.asarray(prototypes[name]['centre'])-prototypes[other]['centre'])),other)
            for other in classes if other!=name)
        nearest.append(dict(colour=name,nearest_colour=other,distance=distance))
    separation=min(row['distance'] for row in nearest)
    if separation<=2*config.minimum_margin:
        return None,dict(reason='TRAIN prototypes not separable at original margin',nearest_pairs=nearest,minimum_distance=separation)
    for row in nearest:prototypes[row['colour']]['acceptance_radius']=min(prototypes[row['colour']]['acceptance_radius'],row['distance']/2)
    return prototypes,dict(minimum_prototype_distance=separation,nearest_pairs=nearest)


def evaluate(samples,x,config,assignment):
    groups=sorted({s.source_group for s in samples});classes=sorted({s.colour_name for s in samples});folds=[];held=[]
    for fold in (0,1):
        fit_ids=[i for i,s in enumerate(samples) if assignment[s.source_group]!=fold]
        held_ids=[i for i,s in enumerate(samples) if assignment[s.source_group]==fold]
        support={name:len({samples[i].source_group for i in fit_ids if samples[i].colour_name==name}) for name in classes}
        if any(n<config.minimum_sources_per_colour for n in support.values()):
            folds.append(dict(fold=fold,valid=False,reason='parent support below original8',support=support));continue
        model,diagnostic=fit([samples[i] for i in fit_ids],x[fit_ids],config)
        if model is None:
            folds.append(dict(fold=fold,valid=False,support=support,diagnostic=diagnostic));continue
        for index in held_ids:
            distances=sorted((float(np.linalg.norm(x[index]-np.asarray(row['centre']))),name) for name,row in model.items())
            distance,predicted=distances[0];margin=distances[1][0]-distance
            supported=bool(distance<=model[predicted]['acceptance_radius'] and margin>=config.minimum_margin)
            s=samples[index];held.append(dict(fold=fold,source_group=s.source_group,task=s.task_name,role=s.role_id,
                true_colour=s.colour_name,predicted_colour=predicted,distance=distance,margin=margin,supported=supported,
                correct_and_supported=bool(supported and predicted==s.colour_name)))
        folds.append(dict(fold=fold,valid=True,support=support,fit_parent_sources=sorted({samples[i].source_group for i in fit_ids}),
            held_parent_sources=sorted({samples[i].source_group for i in held_ids}),diagnostic=diagnostic))
    valid=len(held)==len(samples) and all(row['valid'] for row in folds)
    parent_scores={group:float(np.mean([r['correct_and_supported'] for r in held if r['source_group']==group])) for group in groups} if valid else {}
    macro=float(np.mean(list(parent_scores.values()))) if valid else None
    model,full=fit(samples,x,config)
    task_scores={task:float(np.mean([score for group,score in parent_scores.items() if group.split(':')[1]==task]))
        for task in sorted({s.task_name for s in samples})} if valid else {}
    confusion=defaultdict(Counter)
    for row in held:confusion[row['true_colour']][row['predicted_colour']]+=1
    return dict(valid=valid,passed=bool(valid and macro>=.95 and model is not None),
        macro_heldout_parent_correct_and_supported_fraction=macro,required_fraction=.95,folds=folds,
        per_task_macro_parent_fraction=task_scores,full_fit_complete=model is not None,full_fit_diagnostic=full,
        prototypes=model,held_predictions=held,held_parent_scores=parent_scores,
        held_confusion={name:dict(values) for name,values in confusion.items()})


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True);args=parser.parse_args()
    root=ROOT;base=root/'artifacts/framework/multitask_r10';out=(root/args.out).resolve()
    if not out.is_relative_to(base) or out.exists():raise ValueError('fresh desktop artifact output required')
    out.mkdir(parents=True)
    proof_path=base/'TRAIN_primary3000_counted_highcup100_bundle_v1/bundle_audit.json'
    annotation_path=base/'TRAIN_colour_visual_patch_annotation_candidate_v1/annotation.json'
    peer_path=base/'TRAIN_colour_visual_patch_independent_peer_v1/report.json'
    config_path=base/'command_goal_reference_rule_v6_counted_final_setup_v1/artifact_fit_configuration_full3100.json'
    original=base/'TRAIN_colour_only_visual_supplement_actual_candidate_v1'
    fold_path=original/'TRAIN_parent_crossfit.json'
    refs=[pointer(root,p) for p in (Path(__file__),root/'our_mani/multitask/colour_only_candidate.py',
        root/'our_mani/multitask/public_roles.py',root/'our_mani/multitask/artifact_fitting.py',
        proof_path,annotation_path,peer_path,config_path,fold_path,original/'report.json',original/'actual_colour_input_ledger.json')]
    write(out/'feature_declarations.json',dict(schema='r10_TRAIN_public_colour_feature_study_declarations_v1',features=list(FEATURES),
        declared_before_actual_fit=True,all_folds_fixed_from_original_class_label_only_assignment=True,
        maximum_feature_dimensions=4,no_learned_rescaling_or_repeated_features=True,
        original_Lab_normalisation_preserved=True,minimum_margin=.02,minimum_parent_sources=8,minimum_accuracy=.95,
        VAL_used=False,TEST_used=False,source_inputs=refs))
    cases,ledger,parents=verify_native_colour_cases(root,proof_path)
    config=ColourFitConfig(**read(config_path)['colour'])
    if asdict(config)!=dict(minimum_sources_per_colour=8,within_colour_quantile=.95,minimum_acceptance_radius=.05,
        minimum_margin=.02,minimum_visible_pixels=20,maximum_observation_age=.1):raise ValueError('frozen numerical rules changed')
    samples,excluded=colour_samples(root,cases,config);visual,annotation=load_visual_patch(root,annotation_path,parents,peer_path)
    native_count=len(samples);samples.append(visual)
    assignment=read(fold_path)['parent_to_fold']
    if set(assignment)!={s.source_group for s in samples}:raise ValueError('whole-parent folds or actual sample roster changed')
    results={}
    for name in FEATURES:
        x=np.stack([descriptor(s.rgb_pixels_255,name) for s in samples]);result=evaluate(samples,x,config,assignment)
        result['feature_name']=name;result['descriptor_dimensions']=x.shape[1]
        result['feature_matrix_sha256']=sha256_array(x)
        write(out/(name+'.json'),result)
        results[name]={k:v for k,v in result.items() if k not in ('prototypes','held_predictions','held_parent_scores','held_confusion','folds')}
        results[name]['folds']=[dict(fold=f['fold'],valid=f['valid'],support=f['support'],minimum_distance=f.get('diagnostic',{}).get('minimum_prototype_distance',f.get('diagnostic',{}).get('minimum_distance'))) for f in result['folds']]
        print(json.dumps(dict(feature=name,valid=result['valid'],passed=result['passed'],macro=result['macro_heldout_parent_correct_and_supported_fraction'])),flush=True)
    for reference in refs:bound(root,reference)
    for row in ledger:
        for key in ('canonical','raw_child','colour_sidecar'):bound(root,row[key])
    report=dict(schema='r10_TRAIN_public_colour_features_actual_parent_crossfit_study_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        TRAIN_only=True,VAL_used=False,TEST_used=False,native_GPU_or_new_collection_started=False,
        native_acquisitions_reverified=3100,native_colour_samples=native_count,visual_colour_samples=1,
        native_colour_parents=400,total_colour_parents=len(assignment),colour_classes=len({s.colour_name for s in samples}),
        all_original_samples_retained=True,excluded_native_visible_samples=excluded,
        data_reused_original_colour_ledger=pointer(root,original/'actual_colour_input_ledger.json'),
        new_ledger_matches_original=ledger==read(original/'actual_colour_input_ledger.json')['entries'],
        whole_parent_assignment=pointer(root,fold_path),alternate_assignment_or_accuracy_selected_folds=False,
        fixed_config=asdict(config),required_fraction=.95,features=results,
        passing_feature_names=[name for name,result in results.items() if result['passed']],
        native_runtime_component_map_not_changed=True,visual_patch_is_not_native_semantic_authority=True,
        feature_selection_uses_TRAIN_only=True,formal_release_issued=False,runtime_admission_granted=False,
        independent_feature_module_or_production_artifact_created=False,source_inputs=refs,
        all_bound_colour_raw_and_source_bytes_rehashed_before_after=True,
        motion_array_and_unrelated137GB_audit_reused_not_rehashed=True)
    write(out/'report.json',report);print(json.dumps(dict(report=pointer(root,out/'report.json'),passing=report['passing_feature_names'])))


def sha256_array(array):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array,dtype='<f8').tobytes()).hexdigest()


if __name__=='__main__':main()
