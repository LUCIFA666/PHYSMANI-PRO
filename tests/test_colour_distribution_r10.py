"""CPU boundary tests. Temporary evidence is synthetic, not runtime admission."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from our_mani.contracts import ObjectState,Pose,RobotState
from our_mani.multitask.interaction import PolicyState
from our_mani.multitask.specs import get_task_spec
from our_mani.multitask.public_roles import (ColourFitConfig,TRAINColourSample,
    PublicRoleResolver,PublicRoleResolutionError,_lab,_sha)
from our_mani.multitask.colour_distribution import (FEATURE,FIXED_CONFIG,colour_descriptor,
    sampled_pixels,fit_prototypes,parent_crossfit,fit_candidate,DevelopmentQuantileRoleResolver)

ROOT=Path(__file__).resolve().parents[1]


def write(path,payload):
    path.write_text(json.dumps(payload,sort_keys=True,allow_nan=False))
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class ColourDistributionTests(unittest.TestCase):
    def samples(self):
        patches={'red':np.r_[np.tile([129,0,0],(20,1)),np.tile([251,0,0],(4,1))],
                 'maroon':np.tile([125,0,0],(24,1)),'blue':np.tile([0,0,255],(24,1))}
        samples=[TRAINColourSample('insert_onto_rotating_peg','train:insert_onto_rotating_peg:'+str(i),
            'peg'+str(j),('paint',),name,pixels) for i in range(20) for j,(name,pixels) in enumerate(patches.items())]
        assignment={'train:insert_onto_rotating_peg:'+str(i):i%2 for i in range(20)}
        return samples,assignment,patches

    def candidate(self):
        temporary=TemporaryDirectory(dir=ROOT/'runtime');self.addCleanup(temporary.cleanup)
        directory=Path(temporary.name);samples,assignment,patches=self.samples()
        config=ColourFitConfig(**FIXED_CONFIG)
        manifest=write(directory/'manifest.json',dict(split='train',VAL_used=False,TEST_used=False,
            entries=[dict(split='train',task='insert_onto_rotating_peg',raw_episode_id=str(i)) for i in range(20)]))
        evidence={key:write(directory/(key+'.json'),dict(synthetic_CPU_fixture=True)) for key in (
            'native_acquisition_proof','native_colour_ledger','visual_annotation','visual_peer',
            'original_parent_assignment','feature_study')}
        crossfit=write(directory/'crossfit.json',parent_crossfit(samples,config,assignment))
        candidate=fit_candidate(samples,config,root=ROOT,TRAIN_manifest=manifest,evidence=evidence,crossfit=crossfit)
        return candidate,directory,patches

    def resolver(self,artifact=None,**kwargs):
        if artifact is None:artifact=self.candidate()[0]
        return DevelopmentQuantileRoleResolver(get_task_spec('insert_onto_rotating_peg'),artifact,
            root=ROOT,shared_semantic_permission=True,allow_development_candidate=True,**kwargs)

    def runtime_fixture(self):
        artifact,_,patches=self.candidate();resolver=self.resolver(artifact)
        roles=['peg0','peg1','peg2']
        bodies={role:ObjectState(role,0.,Pose(np.array([i*.1,0.,0.])),np.zeros((3,3)),np.zeros(3),np.zeros(3))
                for i,role in enumerate(roles)}
        robot=RobotState(0.,Pose(np.zeros(3)),np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7))
        state=PolicyState(robot,bodies,{'bodies':{role:dict(directly_visible=True,registration_valid=True) for role in roles}},
                          {role:(role,) for role in roles})
        templates=[SimpleNamespace(role_id=role,components=(SimpleNamespace(name='paint',instance_ids=(i+1,)),))
                   for i,role in enumerate(roles)]
        rgb=np.concatenate([p.reshape(4,6,3) for p in patches.values()],axis=1)
        mask=np.concatenate([np.full((4,6),i+1) for i in range(3)],axis=1)
        observation={'timestamp':0.,'observation/images':{'front':rgb},'observation/instance_masks':{'front':mask}}
        return resolver,state,observation,templates

    def test_quantile_keeps_original_three_dimensional_Lab_units_and_albedo(self):
        _,_,patches=self.samples()
        old=np.linalg.norm(np.median(_lab(patches['red']),axis=0)-np.median(_lab(patches['maroon']),axis=0))
        new=np.linalg.norm(colour_descriptor(patches['red'])-colour_descriptor(patches['maroon']))
        self.assertLess(old,.04);self.assertGreater(new,.04)
        np.testing.assert_array_equal(colour_descriptor(patches['red']),_lab(np.array([[251,0,0]]))[0])
        self.assertEqual(colour_descriptor(patches['red']).shape,(3,))

    def test_sampler_exactly_matches_native_inclusive_evenly_spaced_rule(self):
        pixels=np.arange(3000).reshape(1000,3)%256
        expected=pixels[np.linspace(0,999,512,dtype=int)]
        np.testing.assert_array_equal(sampled_pixels(pixels),expected)
        np.testing.assert_array_equal(colour_descriptor(pixels),_lab(np.quantile(expected,.9,axis=0)[None])[0])

    def test_invalid_RGB_fails_without_clipping(self):
        for pixels in ([],[[1,2]],[[256,0,0]],[[-1,0,0]],[[float('nan'),0,0]]):
            with self.subTest(pixels=pixels),self.assertRaises(PublicRoleResolutionError):colour_descriptor(pixels)

    def test_original_rules_cannot_be_relaxed(self):
        samples,_,_=self.samples()
        with self.assertRaises(PublicRoleResolutionError):
            fit_prototypes(samples,ColourFitConfig(**{**FIXED_CONFIG,'minimum_margin':.01}))
        with self.assertRaises(PublicRoleResolutionError):
            fit_prototypes(samples[:21],ColourFitConfig(**FIXED_CONFIG))

    def test_crossfit_is_whole_parent_and_uses_frozen_roster(self):
        samples,assignment,_=self.samples();cv=parent_crossfit(samples,ColourFitConfig(**FIXED_CONFIG),assignment)
        self.assertEqual(cv['macro_parent_correct_and_supported_fraction'],1.)
        self.assertEqual(len(cv['held_predictions']),60)
        for fold in cv['folds']:
            self.assertFalse(set(fold['fit_parent_sources'])&set(fold['held_parent_sources']))
            self.assertEqual(set(fold['fit_parent_support_by_colour'].values()),{10})
        with self.assertRaises(PublicRoleResolutionError):
            parent_crossfit(samples,ColourFitConfig(**FIXED_CONFIG),dict(list(assignment.items())[:-1]))
        with self.assertRaises(PublicRoleResolutionError):
            parent_crossfit(samples,ColourFitConfig(**FIXED_CONFIG),{g:int(i<5) for i,g in enumerate(assignment)})

    def test_candidate_cannot_silently_replace_original_schema_or_grant_formal(self):
        artifact,_,_=self.candidate()
        self.assertFalse(artifact['metadata']['formal_release']);self.assertFalse(artifact['metadata']['runtime_admission_granted'])
        with self.assertRaises(PublicRoleResolutionError):
            PublicRoleResolver(get_task_spec('insert_onto_rotating_peg'),artifact,shared_semantic_permission=True)
        with self.assertRaises(PublicRoleResolutionError):
            DevelopmentQuantileRoleResolver(get_task_spec('insert_onto_rotating_peg'),artifact,root=ROOT,shared_semantic_permission=True)
        artifact['metadata']['formal_release']=True;artifact['artifact_sha256']=_sha({k:v for k,v in artifact.items() if k!='artifact_sha256'})
        with self.assertRaises(PublicRoleResolutionError):self.resolver(artifact)

    def test_mutated_feature_source_and_evidence_rejected(self):
        artifact,directory,_=self.candidate()
        wrong=deepcopy(artifact);wrong['metadata']['feature']={**FEATURE,'quantile':.8}
        wrong['artifact_sha256']=_sha({k:v for k,v in wrong.items() if k!='artifact_sha256'})
        with self.assertRaises(PublicRoleResolutionError):self.resolver(wrong)
        wrong=deepcopy(artifact);wrong['metadata']['source_bindings']['distribution']=write(directory/'fake_source.py',{})
        wrong['artifact_sha256']=_sha({k:v for k,v in wrong.items() if k!='artifact_sha256'})
        with self.assertRaises(PublicRoleResolutionError):self.resolver(wrong)
        (directory/'visual_annotation.json').write_text('{"changed": true}')
        with self.assertRaises(PublicRoleResolutionError):self.resolver(artifact)

    def test_failed_or_forged_parent_scores_rejected(self):
        artifact,directory,_=self.candidate();cv=json.loads((directory/'crossfit.json').read_text())
        cv['held_parent_scores'][next(iter(cv['held_parent_scores']))]=0.
        artifact['metadata']['parent_crossfit']=write(directory/'crossfit.json',cv)
        artifact['artifact_sha256']=_sha({k:v for k,v in artifact.items() if k!='artifact_sha256'})
        with self.assertRaises(PublicRoleResolutionError):self.resolver(artifact)

    def test_mixed_manifest_entries_rejected_despite_train_header_and_flags(self):
        artifact,directory,_=self.candidate();path=directory/'manifest.json'
        manifest=json.loads(path.read_text());manifest['entries'].append(dict(
            split='validation',task='insert_onto_rotating_peg',raw_episode_id='hidden_VAL'))
        changed=write(path,manifest);samples,_,_=self.samples()
        with self.assertRaises(PublicRoleResolutionError):
            fit_candidate(samples,ColourFitConfig(**FIXED_CONFIG),root=ROOT,TRAIN_manifest=changed,
                evidence=artifact['metadata']['evidence'],crossfit=artifact['metadata']['parent_crossfit'])
        artifact['metadata']['TRAIN_manifest']=changed
        artifact['artifact_sha256']=_sha({k:v for k,v in artifact.items() if k!='artifact_sha256'})
        with self.assertRaises(PublicRoleResolutionError):self.resolver(artifact)

    def test_visual_patch_cannot_create_runtime_component_permission(self):
        artifact,_,_=self.candidate()
        artifact['role_components']['insert_onto_rotating_peg|peg0']=['__no_runtime_component_authority__']
        artifact['artifact_sha256']=_sha({k:v for k,v in artifact.items() if k!='artifact_sha256'})
        with self.assertRaises(PublicRoleResolutionError):self.resolver(artifact)

    def test_current_public_RGB_changes_selected_role_without_native_target_input(self):
        resolver,state,observation,templates=self.runtime_fixture()
        binding,report=resolver.resolve('put ring on red spoke',state,observation,templates)
        self.assertEqual(binding['selected_peg'],('peg0',))
        observation['native_target_spoke']='peg0'  # unused, explicitly hostile extra input
        observation['observation/images']['front']=np.roll(observation['observation/images']['front'],6,axis=1)
        binding,report=resolver.resolve('put ring on red spoke',state,observation,templates)
        self.assertEqual(binding['selected_peg'],('peg1',))
        self.assertFalse(report['native_target_or_variation_read']);self.assertFalse(report['teacher_labels_used_at_inference'])

    def test_fresh_registration_visible_topology_and_ambiguity_remain_required(self):
        resolver,state,observation,templates=self.runtime_fixture()
        stale=replace(state,bodies={k:replace(v,timestamp=-.11) for k,v in state.bodies.items()})
        with self.assertRaises(PublicRoleResolutionError):resolver.resolve('red spoke',stale,observation,templates)
        missing=replace(state,perception_metadata={'bodies':{}})
        with self.assertRaises(PublicRoleResolutionError):resolver.resolve('red spoke',missing,observation,templates)
        with self.assertRaises(PublicRoleResolutionError):resolver.resolve('red spoke',state,{**observation,'timestamp':.05},templates)
        wrong=deepcopy(templates);wrong[0].components[0].name='different'
        with self.assertRaises(PublicRoleResolutionError):resolver.resolve('red spoke',state,observation,wrong)
        observation['observation/images']['front'][:]=[251,0,0]
        with self.assertRaises(PublicRoleResolutionError):resolver.resolve('red spoke',state,observation,templates)

    def test_invisible_pixels_and_unknown_language_do_not_guess(self):
        resolver,state,observation,templates=self.runtime_fixture()
        observation['observation/instance_masks']['front'][:,:6]=0
        with self.assertRaises(PublicRoleResolutionError):resolver.resolve('red spoke',state,observation,templates)
        with self.assertRaises(PublicRoleResolutionError):resolver.resolve('purple spoke',state,observation,templates)


if __name__=='__main__':unittest.main()
