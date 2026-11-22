"""TRAIN-only RGB quantile colour candidate, explicitly separate from v1.

The feature uses the same three normalised Lab coordinates as the original
resolver.  Only the pooled-pixel statistic changes.  No fitting result here
authorises formal evaluation: the resolver requires explicit development use.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .public_roles import (ColourFitConfig, PublicRoleResolutionError,
                           PublicRoleResolver, _lab, _normalise_words, _sha)

SCHEMA = 'r10_TRAIN_public_RGB_q90_Lab_colour_candidate_v1'
FEATURE = dict(name='componentwise_RGB_q90_then_normalised_CIELAB', quantile=.9,
               dimensions=3, RGB_units='0_to_255', Lab_normalisation=[100.,128.,128.],
               learned_scaling=False, maximum_pixels=512,
               pooling_order='observation_images_insertion_order_then_row_major_mask',
               sampling='deterministic_evenly_spaced_indices_inclusive_endpoints')
FIXED_CONFIG = dict(minimum_sources_per_colour=8, within_colour_quantile=.95,
                    minimum_acceptance_radius=.05, minimum_margin=.02,
                    minimum_visible_pixels=20, maximum_observation_age=.1)
REQUIRED_FRACTION = .95
EVIDENCE_KEYS = {'native_acquisition_proof','native_colour_ledger','visual_annotation',
                 'visual_peer','original_parent_assignment','feature_study'}
_VISUAL_ROLE = '__visual_colour_prototype_patch_only__'
_VISUAL_COMPONENT = '__no_runtime_component_authority__'


def _require(value, message):
    if not value:
        raise PublicRoleResolutionError(message)


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checked(root, pointer):
    _require(isinstance(pointer,dict) and set(pointer)=={'path','sha256'},
             'exact colour evidence pointer required')
    path=(root/pointer['path']).resolve()
    _require(path.is_relative_to(root) and _file_sha(path)==pointer['sha256'],
             'colour evidence bytes changed: '+str(path))
    return path


def _config(config):
    _require(isinstance(config,ColourFitConfig) and asdict(config)==FIXED_CONFIG,
             'the original 8/.02/.95 and public-observation rules remain fixed')
    return config


def sampled_pixels(pixels):
    """Match the existing native TRAIN sidecar's deterministic <=512 sampler."""
    pixels=np.asarray(pixels,float)
    _lab(pixels)  # original strict RGB255 validation, no clipping/coercion
    indices=np.linspace(0,len(pixels)-1,min(512,len(pixels)),dtype=int)
    return pixels[indices]


def colour_descriptor(pixels):
    pixels=sampled_pixels(pixels)
    return _lab(np.quantile(pixels,.9,axis=0)[None])[0]


def fit_prototypes(samples,config):
    """Original source balancing/radius/margin, with the declared q90 feature."""
    _config(config);samples=tuple(samples)
    _require(samples and all(s.split=='train' for s in samples),'TRAIN-only colour samples required')
    classes=sorted({s.colour_name for s in samples})
    _require(len(classes)>=2,'at least two colour classes required')
    descriptors=np.stack([colour_descriptor(s.rgb_pixels_255) for s in samples])
    prototypes={}
    for name in classes:
        indices=[i for i,s in enumerate(samples) if s.colour_name==name]
        groups=sorted({samples[i].source_group for i in indices})
        _require(len(groups)>=config.minimum_sources_per_colour,'insufficient TRAIN parents for '+name)
        centres=np.stack([np.median([descriptors[i] for i in indices
                                    if samples[i].source_group==group],axis=0) for group in groups])
        centre=np.median(centres,axis=0)
        radius=max(config.minimum_acceptance_radius,
                   float(np.quantile(np.linalg.norm(centres-centre,axis=1),config.within_colour_quantile)))
        aliases=sorted({name}|{_normalise_words(alias) for i in indices for alias in samples[i].aliases})
        prototypes[name]=dict(prototype_lab_normalised=centre.tolist(),acceptance_radius=radius,
                              aliases=aliases,TRAIN_parent_source_groups=groups)
    aliases=[alias for row in prototypes.values() for alias in row['aliases']]
    _require(len(aliases)==len(set(aliases)),'ambiguous colour aliases')
    for name,row in prototypes.items():
        nearest=min(float(np.linalg.norm(np.asarray(row['prototype_lab_normalised'])-
                    prototypes[other]['prototype_lab_normalised'])) for other in classes if other!=name)
        _require(nearest>2*config.minimum_margin,'TRAIN prototypes not separable at original margin')
        row['acceptance_radius']=min(row['acceptance_radius'],nearest/2)
    return prototypes


def classify_descriptor(descriptor,prototypes,config):
    _config(config);descriptor=np.asarray(descriptor,float)
    _require(descriptor.shape==(3,) and np.isfinite(descriptor).all(),'finite three-dimensional descriptor required')
    distances=sorted((float(np.linalg.norm(descriptor-row['prototype_lab_normalised'])),name)
                     for name,row in prototypes.items())
    _require(len(distances)>=2,'ambiguity checking requires two prototypes')
    distance,name=distances[0];margin=distances[1][0]-distance
    return dict(colour_name=name,distance=distance,margin=margin,
                valid=bool(distance<=prototypes[name]['acceptance_radius'] and margin>=config.minimum_margin))


def parent_crossfit(samples,config,parent_to_fold):
    """Evaluate an already frozen whole-parent assignment, without reassigning."""
    _config(config);samples=tuple(samples);groups={s.source_group for s in samples}
    _require(set(parent_to_fold)==groups and set(parent_to_fold.values())=={0,1},
             'exact predeclared two-fold parent roster required')
    folds=[];held=[];classes=sorted({s.colour_name for s in samples})
    for fold in (0,1):
        fit_samples=[s for s in samples if parent_to_fold[s.source_group]!=fold]
        held_samples=[s for s in samples if parent_to_fold[s.source_group]==fold]
        support={name:len({s.source_group for s in fit_samples if s.colour_name==name}) for name in classes}
        _require(all(n>=8 for n in support.values()),'fold has fewer than eight TRAIN parents')
        prototypes=fit_prototypes(fit_samples,config)
        for sample in held_samples:
            prediction=classify_descriptor(colour_descriptor(sample.rgb_pixels_255),prototypes,config)
            held.append(dict(fold=fold,source_group=sample.source_group,task=sample.task_name,
                role=sample.role_id,true_colour=sample.colour_name,predicted_colour=prediction['colour_name'],
                distance=prediction['distance'],margin=prediction['margin'],supported=prediction['valid'],
                correct_and_supported=prediction['valid'] and prediction['colour_name']==sample.colour_name))
        fit_groups=sorted({s.source_group for s in fit_samples});held_groups=sorted({s.source_group for s in held_samples})
        _require(not set(fit_groups)&set(held_groups),'parent leakage in colour crossfit')
        folds.append(dict(fold=fold,fit_parent_sources=fit_groups,held_parent_sources=held_groups,
                          fit_parent_support_by_colour=support))
    scores={group:float(np.mean([r['correct_and_supported'] for r in held if r['source_group']==group]))
            for group in sorted(groups)}
    macro=float(np.mean(list(scores.values())))
    return dict(schema='r10_RGB_q90_colour_frozen_parent_crossfit_v1',valid=True,
        passed=macro>=REQUIRED_FRACTION,macro_parent_correct_and_supported_fraction=macro,
        required_fraction=REQUIRED_FRACTION,parent_to_fold=dict(parent_to_fold),folds=folds,
        held_predictions=held,held_parent_scores=scores,all_sample_count=len(samples),
        VAL_used=False,TEST_used=False)


def fit_candidate(samples,config,*,root,TRAIN_manifest,evidence,crossfit):
    """Build a new candidate; raw provenance is separately retained in evidence."""
    _config(config);root=Path(root).resolve();samples=tuple(samples)
    manifest_path=_checked(root,TRAIN_manifest);manifest=json.loads(manifest_path.read_text())
    _require(manifest.get('split')=='train' and manifest.get('VAL_used') is False
        and manifest.get('TEST_used') is False and manifest.get('entries')
        and all(row.get('split')=='train' for row in manifest['entries']),
        'explicit TRAIN-only colour manifest and every entry required')
    declared={':'.join(('train',r['task'],r.get('parent_raw_episode_id') or r['raw_episode_id']))
              for r in manifest['entries'] if r.get('split')=='train'}
    groups=sorted({s.source_group for s in samples})
    _require(set(groups)<=declared,'undeclared colour parent source')
    _require(set(evidence)==EVIDENCE_KEYS,'complete actual TRAIN colour evidence required')
    for reference in evidence.values():_checked(root,reference)
    actual_crossfit=json.loads(_checked(root,crossfit).read_text())
    expected_crossfit=parent_crossfit(samples,config,actual_crossfit['parent_to_fold'])
    _require(actual_crossfit==expected_crossfit and actual_crossfit['passed'] is True,
             'actual frozen-parent crossfit did not pass the original .95 gate')
    components={}
    for sample in samples:
        if sample.role_id==_VISUAL_ROLE:
            _require(sample.component_names==(_VISUAL_COMPONENT,), 'visual patch must have no component authority')
            continue
        key=sample.task_name+'|'+sample.role_id;value=sorted(sample.component_names)
        _require(key not in components or components[key]==value,'native component identity changed')
        components[key]=value
    sources={name:dict(path=str(path.relative_to(root)),sha256=_file_sha(path))
             for name,path in [('distribution',Path(__file__).resolve()),
                               ('public_roles',Path(__file__).with_name('public_roles.py').resolve())]}
    payload=dict(metadata=dict(schema=SCHEMA,development_candidate=True,formal_release=False,
        runtime_admission_granted=False,fit_complete=True,fit_split='train',VAL_used=False,TEST_used=False,
        feature=FEATURE,fit_configuration=asdict(config),TRAIN_parent_source_groups=groups,
        TRAIN_manifest=TRAIN_manifest,evidence=evidence,parent_crossfit=crossfit,source_bindings=sources,
        teacher_labels_used_only_in_TRAIN=True,shared_semantic_binding_is_extra_prior=True,
        pure_RGBD_detector_claimed=False,visual_patch_has_runtime_component_authority=False),
        role_components=components,colours=fit_prototypes(samples,config))
    payload['artifact_sha256']=_sha(payload)
    return payload


class DevelopmentQuantileRoleResolver(PublicRoleResolver):
    """Explicit opt-in candidate adapter; never an automatic v1 schema upgrade."""
    def __init__(self,task_spec,artifact,*,root,shared_semantic_permission,
                 allow_development_candidate=False):
        root=Path(root).resolve();m=artifact.get('metadata',{})
        _require(allow_development_candidate is True and shared_semantic_permission is True,
                 'explicit development and shared semantic prior permissions required')
        _require(m.get('schema')==SCHEMA and m.get('development_candidate') is True
            and m.get('formal_release') is False and m.get('runtime_admission_granted') is False
            and m.get('fit_complete') is True and m.get('fit_split')=='train'
            and m.get('VAL_used') is False and m.get('TEST_used') is False
            and m.get('feature')==FEATURE and m.get('visual_patch_has_runtime_component_authority') is False
            and artifact.get('artifact_sha256')==_sha({k:v for k,v in artifact.items() if k!='artifact_sha256'}),
            'frozen explicit q90 TRAIN development candidate required')
        _require(set(m.get('source_bindings',{}))=={'distribution','public_roles'},'complete colour source bindings required')
        actual={'distribution':Path(__file__).resolve(),'public_roles':Path(__file__).with_name('public_roles.py').resolve()}
        for name,path in actual.items():_require(_checked(root,m['source_bindings'][name])==path,'wrong colour execution source')
        manifest=json.loads(_checked(root,m['TRAIN_manifest']).read_text())
        _require(manifest.get('split')=='train' and manifest.get('VAL_used') is False
            and manifest.get('TEST_used') is False and manifest.get('entries')
            and all(row.get('split')=='train' for row in manifest['entries']),
            'every bound colour source manifest entry must remain TRAIN-only')
        declared={':'.join(('train',r['task'],r.get('parent_raw_episode_id') or r['raw_episode_id']))
                  for r in manifest['entries']}
        _require(set(m['TRAIN_parent_source_groups'])<=declared,'undeclared colour parent source')
        _require(set(m.get('evidence',{}))==EVIDENCE_KEYS,'complete actual TRAIN colour evidence required')
        for ref in m['evidence'].values():_checked(root,ref)
        cv=json.loads(_checked(root,m['parent_crossfit']).read_text())
        scores=cv.get('held_parent_scores',{});held=cv.get('held_predictions',[])
        _require(cv.get('valid') is True and cv.get('passed') is True and cv.get('required_fraction')==.95
            and cv.get('VAL_used') is False and cv.get('TEST_used') is False
            and set(scores)==set(m['TRAIN_parent_source_groups'])==set(cv.get('parent_to_fold',{}))
            and cv.get('all_sample_count')==len(held)>0
            and all(abs(scores[group]-float(np.mean([r['correct_and_supported'] for r in held
                if r['source_group']==group])))<1e-12 for group in scores)
            and abs(cv.get('macro_parent_correct_and_supported_fraction',-1)-np.mean(list(scores.values())))<1e-12
            and np.mean(list(scores.values()))>=.95,'colour crossfit proof is incomplete or failed')
        _require(all(_VISUAL_COMPONENT not in components for components in artifact['role_components'].values()),
                 'visual colour patch cannot grant runtime semantic component authority')
        self.spec,self.artifact=task_spec,artifact;self.config=_config(ColourFitConfig(**m['fit_configuration']))

    def _role_colour(self,role,observation,template):
        components=self.artifact['role_components'].get(self.spec.task_name+'|'+role)
        _require(components,'TRAIN artifact lacks native component identity for '+role)
        selected=[c for c in template.components if c.name in components]
        _require({c.name for c in selected}==set(components),'current component topology differs from TRAIN prior')
        ids={identity for c in selected for identity in c.instance_ids};values=[]
        images,masks=observation['observation/images'],observation['observation/instance_masks']
        # Deliberately matches train_color_annotations: insertion order, row-major,
        # then the exact same <=512 evenly spaced sampler before quantile.
        for camera,image in images.items():
            _require(camera in masks,'current RGB is missing its instance mask')
            mask=np.asarray(masks[camera]);rgb=np.asarray(image)
            if mask.ndim==3 and mask.shape[2]==3:
                channels=np.rint(mask*255.).astype(np.int64)
                mask=channels[...,0]+256*channels[...,1]+65536*channels[...,2]
            _require(rgb.ndim==3 and rgb.shape[:2]==mask.shape and rgb.shape[-1]==3,
                     'misaligned current RGB/instance mask')
            pixels=rgb[np.isin(mask,tuple(ids))]
            if len(pixels):values.append(pixels)
        count=sum(len(v) for v in values)
        if count<self.config.minimum_visible_pixels:
            return dict(valid=False,reason='insufficient_current_visible_colour_pixels',pixels=count)
        prediction=classify_descriptor(colour_descriptor(np.concatenate(values)),self.artifact['colours'],self.config)
        return dict(prediction,pixels=count,descriptor_feature=FEATURE['name'],
            provenance='current_RGBD_pixels_with_shared_semantic_instance_component_bindings',
            colour_artifact_sha256=self.artifact['artifact_sha256'],development_candidate=True)
