from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch,Mock
import json,unittest
from our_mani.multitask import rml_case_preflight_v663 as c


class Tests(unittest.TestCase):
    def fixture(self,root,split='validation'):
        task='moving_basketball_in_hoop';worker=root/'worker.py';worker.write_text('pass\n')
        config=dict(baseline_repo=str(root/'native'),common_author_RML_profile={'native_sources_sha256':{}},dt=.05,simulation_seconds=10.,
            gripper_velocity=.2,baseline_decision_ticks=10,physmani_geometry_executor_v517={},
            direct_RML_policy_components_v669={task:{}})
        (root/'config.json').write_text(json.dumps(config));(root/'inputs.json').write_text(json.dumps(dict(files={'unused':'0'*64})))
        manifest=root/'manifest.json';manifest.write_text('{}')
        source=root/('val_package_compressed' if split=='validation' else 'test_alias')/'data.h5'
        source.parent.mkdir(parents=True);source.write_bytes(b'public fixture source')
        row=dict(task=task,raw_episode_id='fixture0',path=str(source))
        args=SimpleNamespace(task=task,raw_episode_id='fixture0',method='context_geometry',split=split,
            config='config.json',frozen_inputs='inputs.json',manifest='manifest.json',out='artifacts/framework/multitask_r10/formal/case0',
            replica=0,exploration_seed=None,record_demo=False,demo_only=False,development_demo=False,
            suite='suite.json',suite_lock='suite_inputs.json',validation_release=None)
        suite=dict(source_manifests={split:{'path':'manifest.json','sha256':c.file_sha(manifest)}},validation_release_path='VAL_release.json')
        return args,row,worker,suite

    def scopes(self,root,worker,suite):
        from contextlib import ExitStack
        stack=ExitStack()
        stack.enter_context(patch.object(c,'verify_inputs'))
        stack.enter_context(patch.object(c,'_checked',side_effect=lambda r,f,p:Path(p).resolve()))
        stack.enter_context(patch.object(c,'common_binding',return_value={'native_sources_sha256':{}}))
        stack.enter_context(patch.object(c,'fingerprint',return_value='profile'))
        stack.enter_context(patch.object(c,'executor_binding',return_value={}))
        stack.enter_context(patch.object(c,'policy_binding',return_value={}))
        stack.enter_context(patch.object(c,'_suite_route',return_value=(suite,{},dict(worker={'path':'worker.py','sha256':c.file_sha(worker)}),'policy',{})))
        stack.enter_context(patch.object(c,'locked_pointer',side_effect=lambda r,f,p:(Path(r)/p['path']).resolve()))
        return stack

    def test_admission_failure_prevents_formal_manifest_selection_and_model_loading(self):
        with TemporaryDirectory() as d:
            root=Path(d);args,row,worker,suite=self.fixture(root)
            with self.scopes(root,worker,suite),patch.object(c,'for_evaluation',side_effect=ValueError('full192 not admitted')) as admit,patch.object(c,'cpu_preflight') as model,patch.object(c,'for_train') as train:
                with self.assertRaisesRegex(ValueError,'full192'):c.prepare(root,args,worker)
                self.assertEqual(admit.call_args.args[3],'validation')
                model.assert_not_called();train.assert_not_called()

    def test_actual_validation_domain_reaches_new_component_loader_without_TRAIN_relabel(self):
        with TemporaryDirectory() as d:
            root=Path(d);args,row,worker,suite=self.fixture(root)
            domain=Mock(purpose='evaluation')
            with self.scopes(root,worker,suite),patch.object(c,'for_evaluation',return_value=(domain,row,dict(policy_sha256='policy'))) as admit,patch.object(c,'cpu_preflight',return_value={}) as model,patch.object(c,'for_train') as train:
                prepared=c.prepare(root,args,worker)
                self.assertEqual(prepared['identity']['split'],'validation')
                self.assertFalse(prepared['identity']['demo_only']);self.assertTrue(prepared['formal'])
                self.assertIs(model.call_args.args[4],domain)
                self.assertEqual(admit.call_args.args[3],'validation');train.assert_not_called()

    def test_TEST_without_complete_VAL_receipt_cannot_reach_source_admission(self):
        with TemporaryDirectory() as d:
            root=Path(d);args,row,worker,suite=self.fixture(root,'test')
            with self.scopes(root,worker,suite),patch.object(c,'for_evaluation') as admit:
                with self.assertRaisesRegex(ValueError,'complete VAL release'):c.prepare(root,args,worker)
                admit.assert_not_called()

    def test_development_scope_cannot_enter_formal_validation(self):
        with TemporaryDirectory() as d:
            root=Path(d);args,row,worker,suite=self.fixture(root);args.development_demo=True
            with patch.object(c,'verify_inputs') as verify:
                with self.assertRaisesRegex(ValueError,'formal scope'):c.prepare(root,args,worker)
                verify.assert_not_called()


if __name__=='__main__':unittest.main()
