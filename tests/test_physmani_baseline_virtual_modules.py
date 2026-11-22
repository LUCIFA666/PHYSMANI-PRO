"""Real CPU torch objects and strict virtual-owner/cache boundary regressions."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, MappingProxyType
import unittest
from unittest.mock import patch

import torch

from our_mani.multitask import baseline_component as b


class StockTorchVirtualModuleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.owners={name:sys.modules[name] for name in ('torch._ops','torch._classes')}
        self.modules={'torch':torch,**self.owners,'torch.ops':torch.ops,'torch.classes':torch.classes}
        self.dependencies={str(Path(module.__file__).resolve()):b.file_sha(module.__file__)
                           for module in [torch,*self.owners.values()]}
        self.dependencies[str(Path(sys.executable).resolve())]=b.file_sha(sys.executable)
        self.contract=self.root/'contract.json'
        self.contract.write_text(json.dumps(dict(dependency_groups={'local_sources':self.dependencies})))
        self.pointer=dict(path='contract.json',sha256=b.file_sha(self.contract))
        self.backend=b.ProjectedBaselineBackend(root=self.root,execution_contract=self.pointer,inner=SimpleNamespace())
        self.backend.witness={'imported_dependency_files_sha256':{},'loaded_native_files_sha256':{}}
        self.cuda_before=torch.cuda.is_initialized()

    def tearDown(self):
        self.assertEqual(torch.cuda.is_initialized(),self.cuda_before)
        self.temp.cleanup()

    def observe(self,modules=None):
        original=Path.read_text
        def read(path,*args,**kwargs):
            return '' if str(path)=='/proc/self/maps' else original(path,*args,**kwargs)
        with patch.dict(sys.modules,self.modules if modules is None else modules,clear=True),patch.object(Path,'read_text',read):
            self.backend._imports()

    def test_real_CPU_torch_registry_types_resolve_and_emit_exact_bound_owner_witnesses(self):
        self.observe()
        witness=self.backend.witness;aliases=witness['virtual_module_owner_bindings']
        self.assertEqual(set(aliases),{'torch.ops','torch.classes'})
        b.validate_virtual_owner_bindings(aliases,self.dependencies,witness['imported_dependency_files_sha256'])
        for key,(owner,stock,literal,attribute) in b.TORCH_VIRTUAL_OWNERS.items():
            self.assertIs(type(self.modules[key]),vars(self.owners[owner])[stock])
            self.assertEqual(aliases[key]['literal_placeholder'],literal)
            self.assertEqual(aliases[key]['owner_file']['sha256'],b.file_sha(self.owners[owner].__file__))
            self.assertTrue(aliases[key]['registry_identity_verified'])

    def test_same_stock_type_new_instance_fails_registry_identity(self):
        replacement=type(torch.ops)()
        with patch.dict(sys.modules,dict(self.modules,**{'torch.ops':replacement}),clear=True),self.assertRaisesRegex(ValueError,'registry identity'):
            b.torch_virtual_owner('torch.ops',replacement)

    def test_fake_type_even_with_registry_and_parent_attributes_is_rejected(self):
        fake=SimpleNamespace(__file__='_ops.py',__name__='torch.ops')
        with patch.object(torch,'ops',fake),patch.object(self.owners['torch._ops'],'ops',fake),\
             patch.dict(sys.modules,dict(self.modules,**{'torch.ops':fake}),clear=True),\
             self.assertRaisesRegex(ValueError,'stock type'):
            b.torch_virtual_owner('torch.ops',fake)

    def test_literal_and_real_owner_origin_cannot_be_replaced(self):
        with patch.object(type(torch.ops),'__file__','other.py'),\
             self.assertRaisesRegex(ValueError,'placeholder'):
            b.torch_virtual_owner('torch.ops',torch.ops)
        other=self.root/'torch/_ops.py';other.parent.mkdir();other.write_text('# false owner\n')
        with patch.object(self.owners['torch._ops'],'__file__',str(other)),\
             self.assertRaisesRegex(ValueError,'owner origin'):
            b.torch_virtual_owner('torch.ops',torch.ops)

    def test_changed_bound_owner_SHA_is_rejected(self):
        self.dependencies[str(Path(self.owners['torch._ops'].__file__).resolve())]='f'*64
        self.contract.write_text(json.dumps(dict(dependency_groups={'local_sources':self.dependencies})))
        self.backend.execution_contract['sha256']=b.file_sha(self.contract)
        with self.assertRaisesRegex(ValueError,'outside frozen closure'):self.observe()

    def test_unreviewed_relative_missing_absolute_and_unknown_registry_alias_still_rejected(self):
        for name,file in [('unknown','_ops.py'),('unknown',str(self.root/'missing.py')),('unknown',str(self.root/'relative.py'))]:
            with self.subTest(file=file),self.assertRaises(ValueError):
                self.observe({'unknown':SimpleNamespace(__file__=file)})
        with self.assertRaisesRegex(ValueError,'unreviewed relative'):
            self.observe({**self.modules,'unreviewed.stock_alias':torch.ops})
        # Even an existing declared file must not gain generic relative-file
        # resolution as a side effect of the two reviewed torch aliases.
        allowed=self.root/'relative.py';allowed.write_text('# bound ordinary file\n')
        self.dependencies[str(allowed)]=b.file_sha(allowed)
        self.contract.write_text(json.dumps(dict(dependency_groups={'local_sources':self.dependencies})))
        self.backend.execution_contract['sha256']=b.file_sha(self.contract)
        previous=Path.cwd()
        try:
            os.chdir(self.root)
            with self.assertRaisesRegex(ValueError,'unreviewed relative'):
                self.observe({'ordinary':SimpleNamespace(__file__='relative.py')})
        finally:os.chdir(previous)

    def test_raw_typed_witness_cannot_lose_alias_owner_hash_or_stock_identity(self):
        self.observe();original=self.backend.witness['virtual_module_owner_bindings'];imports=self.backend.witness['imported_dependency_files_sha256']
        for mutation in ('missing','SHA','type','registry','owner'):
            value=json.loads(json.dumps(original))
            if mutation=='missing':value.pop('torch.ops')
            elif mutation=='SHA':value['torch.ops']['owner_file']['sha256']='f'*64
            elif mutation=='type':value['torch.ops']['stock_type']='fake._Ops'
            elif mutation=='registry':value['torch.ops']['registry_identity_verified']=False
            else:value['torch.ops']['owner_file']['path']=str(self.root/'_ops.py')
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                b.validate_virtual_owner_bindings(value,self.dependencies,imports)

    def test_unverified_maps_never_populate_cache_and_root_SHA_identity_is_exact(self):
        key=(str(self.root.resolve()),self.pointer['sha256'])
        b._VERIFIED_DEPENDENCIES.pop(key,None)
        b.observer_dependencies(self.root,self.pointer)
        self.assertNotIn(key,b._VERIFIED_DEPENDENCIES)
        # Test an already verified immutable cache without granting a fixture
        # execution contract; the complete real contract path is tested in the
        # existing contract suite.
        cached=MappingProxyType(dict(self.dependencies));b._VERIFIED_DEPENDENCIES[key]=cached
        with patch.object(b,'normalized',side_effect=AssertionError('cached normalization')):
            self.assertIs(b.observer_dependencies(self.root,self.pointer),cached)
        with self.assertRaises(TypeError):cached['new']='f'*64
        other=self.root/'other_root';other.mkdir();(other/'contract.json').write_bytes(self.contract.read_bytes())
        self.assertIsNot(b.observer_dependencies(other,self.pointer),cached)
        self.contract.write_text(self.contract.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'immutable project pointer changed'):
            b.observer_dependencies(self.root,self.pointer)
        changed=dict(self.pointer,sha256=b.file_sha(self.contract))
        self.assertIsNot(b.observer_dependencies(self.root,changed),cached)
        self.assertNotIn((str(self.root.resolve()),changed['sha256']),b._VERIFIED_DEPENDENCIES)
        b._VERIFIED_DEPENDENCIES.pop(key,None)


if __name__=='__main__':unittest.main()
