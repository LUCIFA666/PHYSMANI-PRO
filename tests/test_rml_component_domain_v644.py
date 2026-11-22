from pathlib import Path
import json
import unittest
import numpy as np
from our_mani.multitask.evaluation import file_sha
from our_mani.multitask.rml_component_domain_v640 import ComponentDomain,for_train,for_evaluation
from our_mani.multitask.button_command_library_v641 import ButtonCommandLibrary
from our_mani.multitask.button_intent_library_v71 import ButtonIntentLibrary
from our_mani.multitask.rml_component_factory_v643 import load_library
from our_mani.multitask.colour_component_domain_v642 import QualifiedQuantileRoleResolver
from our_mani.multitask.specs import TASK_SPECS
ROOT=Path(__file__).resolve().parents[1];TASK='push_moving_button'


class Tests(unittest.TestCase):
    def fixture(self,raw=None):
        path=ROOT/'artifacts/framework/protocol/train_manifest.json';rows=json.loads(path.read_text())['episodes']
        raw=raw or next(r['raw_episode_id'] for r in rows if r['task']==TASK)
        files={str(path):file_sha(path)};domain=for_train(ROOT,files,TASK,raw,'context_geometry')
        c=json.loads((ROOT/'artifacts/framework/multitask_r10/physmani_skills_v536_setup'/f'{TASK}_configuration.json').read_text())
        return c,domain,files

    def test_new_command_consumer_retains_every_original_target_without_claiming_physical_press_completion(self):
        c,domain,_=self.fixture();path=ROOT/c['learned_artifacts'][TASK]['interaction']['path']
        old=ButtonIntentLibrary(ROOT,path,purpose='train_development');new=ButtonCommandLibrary(ROOT,path,domain=domain)
        for a,b in zip(old.templates,new.templates):
            np.testing.assert_allclose(a.at(1.)[0].xyzw(),b.at(1.)[0].xyzw(),atol=1e-14)
            np.testing.assert_array_equal(a.opening,b.opening)
        self.assertFalse(new.metadata['completed_skill_support_claimed'])
        self.assertFalse(new.metadata['public_press_completion_granted'])
        self.assertEqual(new.runtime_component_domain['split'],'train')
        with self.assertRaises(ValueError):ButtonIntentLibrary(ROOT,path,purpose='evaluation')

    def test_material_identity_is_independent_of_which_fixed_TRAIN_case_constructs_it(self):
        c,domain,_=self.fixture();rows=json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
        raw=[r['raw_episode_id'] for r in rows if r['task']==TASK][1]
        _,second,_=self.fixture(raw)
        a=load_library(ROOT,c,TASK,domain);b=load_library(ROOT,c,TASK,second)
        self.assertEqual(a.sha256,b.sha256)
        self.assertNotEqual(a.runtime_component_domain['raw_episode_id'],b.runtime_component_domain['raw_episode_id'])

    def test_forged_evaluation_capability_is_rejected_before_loading_models(self):
        c,domain,files=self.fixture();fake=ComponentDomain('evaluation',TASK,'context_geometry','test','episode0','f'*64,object())
        with self.assertRaises(ValueError):load_library(ROOT,c,TASK,fake)
        with self.assertRaises(ValueError):for_evaluation(ROOT,{},files,'test',TASK,'episode0','context_geometry')
        rows=json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
        fifth=[r['raw_episode_id'] for r in rows if r['task']==TASK][4]
        with self.assertRaises(ValueError):for_train(ROOT,files,TASK,fifth,'context_geometry')

    def test_qualified_colour_loader_preserves_original_artifact_and_all_role_thresholds(self):
        c,domain,_=self.fixture();artifact=json.loads((ROOT/c['learned_artifacts'][TASK]['colour']['path']).read_text())
        snapshot=json.dumps(artifact,sort_keys=True)
        resolver=QualifiedQuantileRoleResolver(TASK_SPECS[TASK],artifact,root=ROOT,shared_semantic_permission=True,domain=domain)
        self.assertEqual(snapshot,json.dumps(artifact,sort_keys=True))
        self.assertFalse(resolver.formal_runtime_admission)
        self.assertEqual(resolver.config.minimum_sources_per_colour,8)
        self.assertEqual(resolver.config.minimum_margin,.02)


if __name__=='__main__':unittest.main()
