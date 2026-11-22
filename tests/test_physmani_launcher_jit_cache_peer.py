"""Actual saved launcher observation -> current exact JIT profile CPU wiring.

The starting launcher fields are historical native-v8 observations, never
synthetic actual-v10 claims. Only the two cache exports are projected to the
new metadata profile. No launcher, simulator, Torch, Triton or GPU runs here.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

from our_mani.multitask import baseline_jit_artifacts as jit
from our_mani.multitask import baseline_runtime_state as runtime
from our_mani.multitask import baseline_component as component

ROOT=Path(__file__).resolve().parents[1]
NATIVE_PYTHON=Path('/home/ubuntu/physmani-conda/envs/physmani5090/bin/python')
V8=ROOT/'artifacts/framework/multitask_r10/PhysMani_early_VAL_approved_v8'
RAW=V8/'executions/cases/3abaa978021bf3143b88/attempt0000/infrastructure_error.json'
RAW_SHA='d530a91a38327b3c189295217583dfb99f81e6eaa066677939abe08ddb11eb69'
CONTRACT_SHA='d9262222abf7ac0ded033e78b6ccb31efd176ede98d1865aa8f32bc02cc6940e'
DRAFT_PROFILE=ROOT/'artifacts/framework/multitask_r10/PhysMani_early_VAL_dependency_review_v10/jit_artifact_profile_draft.json'
DRAFT_PROFILE_SHA='3cff1dc73ef70f83918f0ffefb49546ef93b0124dcedb1d9418eb4a28feebc7a'


class ActualLauncherJITCacheWiringPeerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert jit.file_sha(RAW)==RAW_SHA
        assert jit.file_sha(V8/'execution_contract.json')==CONTRACT_SHA
        raw=json.loads(RAW.read_text())
        cls.actual_v8=raw['recorded_initial_conditions']['baseline_execution_dependency_witness']['actual_launcher_environment']
        assert cls.actual_v8['source']=='actual_python_child_after_runtime_script'
        cls.config=json.loads((V8/'early_config.json').read_text())
        cls.dependencies=component.observer_dependencies(ROOT,dict(path=str((V8/'execution_contract.json').relative_to(ROOT)),sha256=CONTRACT_SHA))
        assert jit.file_sha(DRAFT_PROFILE)==DRAFT_PROFILE_SHA
        cls.profile=json.loads(DRAFT_PROFILE.read_text())
        assert cls.profile['schema']==jit.PROFILE
        assert cls.profile['source_bindings_sha256'][str(Path(jit.__file__).resolve())]==jit.file_sha(jit.__file__)
        assert cls.profile['cache_paths']==dict(inductor=str(ROOT/'runtime/framework/physmani_r10_JIT_v10/inductor'),
            triton=str(ROOT/'runtime/framework/physmani_r10_JIT_v10/triton'))
        cls.initial_Torch_imported='torch' in sys.modules
        cls.initial_Triton_imported='triton' in sys.modules

    def projected_launcher(self,paths=None):
        value=deepcopy(self.actual_v8);paths=self.profile['cache_paths'] if paths is None else paths
        value['environment'].update(TORCHINDUCTOR_CACHE_DIR=paths['inductor'],TRITON_CACHE_DIR=paths['triton'])
        return value

    def validate(self,value,profile=None,dependencies=None):
        runtime.validate_launcher_environment(value,self.dependencies if dependencies is None else dependencies,
            ROOT,self.config,str(NATIVE_PYTHON),jit_profile=self.profile if profile is None else profile)

    def test_real_saved_launcher_fields_plus_exact_v10_metadata_paths_pass_without_other_projection(self):
        projected=self.projected_launcher()
        changed={key for key,value in projected['environment'].items() if self.actual_v8['environment'][key]!=value}
        self.assertEqual(changed,{'TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'})
        self.assertEqual(projected['programs'],self.actual_v8['programs'])
        self.validate(projected)
        self.assertFalse(self.profile['actual_native_or_GPU_execution'])
        self.assertFalse(self.profile['reviewed_before_execution'])
        self.assertEqual('torch' in sys.modules,self.initial_Torch_imported)
        self.assertEqual('triton' in sys.modules,self.initial_Triton_imported)

    def test_old_or_foreign_profile_paths_rejected_even_when_environment_matches_that_profile(self):
        for relative in ('physmani_r10_JIT_v8','physmani_r10_JIT_v9','foreign_cache','../outside_cache'):
            paths={key:str(ROOT/'runtime/framework'/relative/key) for key in ('inductor','triton')}
            profile=deepcopy(self.profile);profile['cache_paths']=paths
            with self.subTest(path=relative),self.assertRaisesRegex(ValueError,'JIT profile widened ROOT paths'):
                self.validate(self.projected_launcher(paths),profile)
        paths=dict(inductor='/tmp/foreign_inductor',triton='/tmp/foreign_triton')
        profile=deepcopy(self.profile);profile['cache_paths']=paths
        with self.assertRaisesRegex(ValueError,'JIT profile widened ROOT paths'):
            self.validate(self.projected_launcher(paths),profile)

    def test_correct_v10_profile_rejects_actual_old_exports_and_each_unsynchronized_export(self):
        with self.assertRaisesRegex(ValueError,'runtime/project/compiler/cache path differs'):self.validate(self.actual_v8)
        for key in ('TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'):
            for version in ('v8','v9'):
                value=self.projected_launcher()
                value['environment'][key]=value['environment'][key].replace('_v10/','_'+version+'/')
                with self.subTest(key=key,version=version),self.assertRaises(ValueError):self.validate(value)

    def test_v10_projection_keeps_exact_existing_environment_and_program_guards(self):
        for key,value in (('CUDA_VISIBLE_DEVICES','1'),('MAX_JOBS','7'),('PYTHONPATH',str(ROOT)),('CC','/usr/bin/gcc')):
            changed=self.projected_launcher();changed['environment'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate(changed)
        changed=self.projected_launcher();changed['environment']['UNKNOWN_CACHE_EXPORT']='not_allowed'
        with self.assertRaises(ValueError):self.validate(changed)
        changed=self.projected_launcher();changed['environment'].pop('TRITON_CACHE_DIR')
        with self.assertRaises(ValueError):self.validate(changed)
        changed=self.projected_launcher();changed['programs']['python']['sha256']='f'*64
        with self.assertRaises(ValueError):self.validate(changed)
        dependencies=dict(self.dependencies);dependencies.pop(str(NATIVE_PYTHON.resolve()))
        with self.assertRaises(ValueError):self.validate(self.projected_launcher(),dependencies=dependencies)

    def test_actual_historical_program_bytes_and_origin_record_remain_unchanged(self):
        for name,item in self.actual_v8['programs'].items():
            with self.subTest(program=name):
                self.assertEqual(jit.file_sha(item['path']),item['sha256'])
                self.assertEqual(self.dependencies[str(Path(item['path']).resolve())],item['sha256'])
        self.assertEqual(jit.file_sha(RAW),RAW_SHA)
        self.assertEqual(self.actual_v8['environment']['TORCHINDUCTOR_CACHE_DIR'],
            str(ROOT/'runtime/framework/physmani_r10_JIT_v8/inductor'))
        self.assertEqual(self.actual_v8['environment']['TRITON_CACHE_DIR'],
            str(ROOT/'runtime/framework/physmani_r10_JIT_v8/triton'))


if __name__=='__main__':unittest.main()
