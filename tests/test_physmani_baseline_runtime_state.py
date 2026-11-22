from copy import deepcopy
import ast
from contextlib import ExitStack
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from our_mani.multitask import baseline_runtime_state as state
from our_mani.multitask.evaluation import file_sha


def encoded(value=b'No',key=state.MARKER):
    return struct.pack('<iB',1,1)+struct.pack('<i',len(key))+key+struct.pack('<i',len(value))+value+b'\x00'


class RuntimeStateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.path=self.root/'coppelia/system/settings.dat';self.path.parent.mkdir(parents=True)
        self.path.write_bytes(encoded());self.override=patch.object(state,'STATE_FILE',self.path.resolve());self.override.start()
        self.env_override=patch.dict(os.environ,{'COPPELIASIM_ROOT':str(self.path.parent.parent)});self.env_override.start()
        self.config={'baseline_repo':str(self.root/'repo')}
        for path in [self.root/'our_mani/multitask/simulation.py',self.root/'coppelia/libcoppeliaSim.so',
             *[self.root/'repo/third_party/PyRep/pyrep'/name for name in ('pyrep.py','backend/sim.py','backend/simConst.py')]]:
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'CPU fixture, no native import')
        self.evidence=self.root/'evidence.json';self.evidence.write_text('{}')
        self.value=state.draft_permission(self.root,self.config,{'path':'evidence.json','sha256':file_sha(self.evidence)})
        self.permission_path=self.root/'permission.json'

    def tearDown(self):self.env_override.stop();self.override.stop();self.temp.cleanup()

    def pointer(self):
        self.permission_path.write_text(json.dumps(self.value))
        return {'path':'permission.json','sha256':file_sha(self.permission_path)}

    def test_permission_draft_never_grants_execution(self):
        pointer=self.pointer()
        state.load_permission(self.root,pointer,self.config,require_reviewed=False)
        with self.assertRaises(ValueError):state.load_permission(self.root,pointer,self.config)
        self.value['reviewed_before_execution']=True
        state.load_permission(self.root,self.pointer(),self.config)

    def test_permission_cannot_widen_path_values_or_fields(self):
        for key,value in [('resolved_path',str(self.path.parent/'other.dat')),('allowed_values',['No','Yes','Maybe']),
            ('native_headless_required',False),('directory_integrity_for_every_other_file',False),('unknown_setting',True)]:
            original=deepcopy(self.value);self.value[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                state.load_permission(self.root,self.pointer(),self.config,require_reviewed=False)
            self.value=original

    def test_native_API_source_or_evidence_change_rejected(self):
        pointer=self.pointer();path=self.root/'repo/third_party/PyRep/pyrep/backend/sim.py';path.write_text('changed')
        with self.assertRaises(ValueError):state.load_permission(self.root,pointer,self.config,require_reviewed=False)
        self.value=state.draft_permission(self.root,self.config,{'path':'evidence.json','sha256':file_sha(self.evidence)})
        pointer=self.pointer();self.evidence.write_text('{"changed":true}')
        with self.assertRaises(ValueError):state.load_permission(self.root,pointer,self.config,require_reviewed=False)

    def test_exact_marker_parser_rejects_any_extra_real_setting(self):
        for payload in [encoded(b'Maybe'),encoded(key=b'UNKNOWN_CRASH_MARKER_001'),encoded()+b'physics=changed',
            b'\x02'+encoded()[1:],encoded()[:-1],encoded()[:-1]+b'\x01']:
            with self.subTest(payload=payload.hex()),self.assertRaises(ValueError):state.parse_crash_marker(payload)
        for value in (b'No',b'Yes'):self.assertEqual(state.parse_crash_marker(encoded(value))['value'],value.decode())

    def test_snapshot_is_exact_path_stable_read_and_read_only(self):
        before=self.path.read_bytes();observed=state.snapshot(self.path)
        self.assertEqual(self.path.read_bytes(),before);state.validate_snapshot(observed,self.value)
        other=self.path.parent/'usrset.txt';other.write_bytes(before)
        with self.assertRaises(ValueError):state.snapshot(other)
        self.path.write_bytes(before+b'unknown-setting')
        with self.assertRaises(ValueError):state.snapshot(self.path)
        diagnostic=state.diagnostic_snapshot()
        self.assertFalse(diagnostic['complete']);self.assertIn('parse_error',diagnostic)
        self.assertEqual(diagnostic['hex'],self.path.read_bytes().hex())
        with self.assertRaises(ValueError):state.validate_snapshot(diagnostic,self.value)

    def fake_sim(self,*,headless=True,times=(1.,1.)):
        iterator=iter(times);calls=[]
        def clock():calls.append(('clock',));return next(iterator)
        def query(parameter):calls.append(('query',parameter));return headless
        return SimpleNamespace(sim_boolparam_headless=34,sim_gui_headless=65536,
            simGetSimulationTime=clock,simGetBoolParameter=query),calls

    def test_headless_uses_only_public_read_APIs_no_step_or_setter(self):
        sim,calls=self.fake_sim();witness=state.observe_headless(sim=sim)
        self.assertEqual(calls,[('clock',),('query',34),('clock',)])
        self.assertTrue(witness['value']);self.assertTrue(witness['physical_time_unchanged'])
        self.assertFalse(witness['task_ground_truth_read'])

    def test_actual_installed_PyRep_API_AST_preserves_C_integer_return(self):
        path=Path('/home/ubuntu/projects/PhysMani/third_party/PyRep/pyrep/backend/sim.py')
        tree=ast.parse(path.read_text())
        node=next(item for item in tree.body if isinstance(item,ast.FunctionDef) and item.name=='simGetBoolParameter')
        calls=[]
        lib=SimpleNamespace(simGetBoolParameter=lambda parameter:(calls.append(('native_read',parameter)) or 1))
        namespace={'lib':lib,'_check_return':lambda value:calls.append(('check',value))}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        sim,_=self.fake_sim();sim.simGetBoolParameter=namespace['simGetBoolParameter']
        witness=state.observe_headless(sim=sim)
        self.assertEqual(calls,[('native_read',34),('check',1)])
        self.assertIs(type(witness['raw_value']),int);self.assertEqual(witness['raw_type'],'int')

    def test_false_headless_changed_clock_and_wrong_API_constant_rejected(self):
        for sim in (self.fake_sim(headless=False)[0],self.fake_sim(headless=1.0)[0],self.fake_sim(times=(1.,1.05))[0]):
            with self.assertRaises(ValueError):state.observe_headless(sim=sim)
        sim,_=self.fake_sim();sim.sim_boolparam_headless=999
        with self.assertRaises(ValueError):state.observe_headless(sim=sim)
        sim,_=self.fake_sim()
        with patch.dict(os.environ,{'COPPELIASIM_ROOT':str(self.root/'unknown_runtime')}),self.assertRaises(ValueError):
            state.observe_headless(sim=sim)

    def test_case_before_after_known_values_and_limits_remain_honest(self):
        pointer=self.pointer();before=state.snapshot(self.path);sim,_=self.fake_sim();headless=state.observe_headless(sim=sim)
        initial=dict(permission=pointer,before=before,headless=headless)
        self.path.write_bytes(encoded(b'Yes'))
        witness=dict(schema=state.WITNESS,**deepcopy(initial),after=state.snapshot(self.path),limitations=state.LIMITATIONS)
        state.validate_case_witness(witness,self.value,pointer,initial)
        for change in ('headless','before','after','limits'):
            bad=deepcopy(witness)
            if change=='headless':bad['headless']['value']=False
            elif change=='before':bad['before']['parsed']['value']='Yes'
            elif change=='after':bad['after']['parsed']['sha256']='0'*64
            else:bad['limitations']=[]
            with self.subTest(change=change),self.assertRaises(ValueError):state.validate_case_witness(bad,self.value,pointer,initial)

    def launcher_fixture(self):
        prefix=self.root/'env';base=self.root/'base_conda';paths={}
        for name in state.SHELL_PROGRAMS:
            path=prefix/'bin'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(name.encode());paths[name]=path
        for name in ('conda','python'):
            path=base/'bin'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(('base '+name).encode())
        interpreter=paths['python'];repo=Path(self.config['baseline_repo'])
        environment={key:'' for key in state.LAUNCH_ENV_KEYS}
        environment.update(PATH=str(self.root/'usr/bin')+':'+str(prefix/'bin'),
            LD_LIBRARY_PATH=':'.join(map(str,[self.path.parent.parent,self.root/'usr/lib/x86_64-linux-gnu',prefix/'lib',prefix/'targets/x86_64-linux/lib'])),
            PYTHONPATH=':'.join(map(str,[self.root,repo,repo/'third_party/RLBench',repo/'third_party/PyRep'])),
            COPPELIASIM_ROOT=str(self.path.parent.parent),CONDA_PREFIX=str(prefix),CUDA_HOME=str(prefix),
            QT_QPA_PLATFORM_PLUGIN_PATH=str(self.path.parent.parent),PHYSMANI_PROJECT_ROOT=str(repo),PHYS_MANI_REPO_ROOT=str(repo),
            CC=str(prefix/'bin/x86_64-conda-linux-gnu-cc'),CXX=str(prefix/'bin/x86_64-conda-linux-gnu-c++'),
            CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCH_CUDA_ARCH_LIST='12.0',
            MAX_JOBS='6',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',XDG_CACHE_HOME=str(self.root/'runtime/framework/cache'),
            TORCH_HOME=str(self.root/'runtime/framework/torch'),TMPDIR=str(self.root/'runtime/framework/tmp'))
        deps={str(path.resolve()):file_sha(path) for path in (*paths.values(),base/'bin/conda',base/'bin/python',Path('/bin/sh').resolve())}
        stack=ExitStack();stack.enter_context(patch.object(state,'BASE_CONDA',base))
        stack.enter_context(patch.object(state.sys,'executable',str(interpreter)))
        stack.enter_context(patch.object(state.shutil,'which',side_effect=lambda name:str(paths[name])))
        stack.enter_context(patch.dict(os.environ,environment,clear=True))
        self.addCleanup(stack.close)
        return deps,interpreter,environment,paths

    def test_actual_child_environment_and_external_programs_are_bound_separately_from_preparation_env(self):
        deps,interpreter,environment,_=self.launcher_fixture()
        witness=state.observe_launcher_environment(deps,self.root,self.config,interpreter)
        self.assertEqual(witness['environment'],environment)
        self.assertEqual(witness['source'],'actual_python_child_after_runtime_script')
        self.assertEqual(len(witness['programs']),19)
        state.validate_launcher_environment(witness,deps,self.root,self.config,interpreter)

    def test_launcher_JIT_profile_and_actual_exports_share_one_exact_authority(self):
        from our_mani.multitask.baseline_jit_artifacts import cache_paths
        deps,interpreter,environment,_=self.launcher_fixture()
        paths=cache_paths(self.root);profile=dict(cache_paths=paths)
        exports=dict(TORCHINDUCTOR_CACHE_DIR=paths['inductor'],TRITON_CACHE_DIR=paths['triton'])
        with patch.dict(os.environ,exports):
            witness=state.observe_launcher_environment(deps,self.root,self.config,interpreter,jit_profile=profile)
        self.assertEqual({k:witness['environment'][k] for k in exports},exports)
        state.validate_launcher_environment(witness,deps,self.root,self.config,interpreter,jit_profile=profile)
        for relative in ('physmani_r10_JIT_v8','physmani_r10_JIT_v9','unreviewed_JIT'):
            foreign={k:str(self.root/'runtime/framework'/relative/k) for k in paths}
            bad_profile=dict(cache_paths=foreign);bad_witness=deepcopy(witness)
            bad_witness['environment'].update(TORCHINDUCTOR_CACHE_DIR=foreign['inductor'],TRITON_CACHE_DIR=foreign['triton'])
            for observed,declared in [(witness,bad_profile),(bad_witness,profile),(bad_witness,bad_profile)]:
                with self.subTest(path=relative),self.assertRaises(ValueError):
                    state.validate_launcher_environment(observed,deps,self.root,self.config,interpreter,jit_profile=declared)

    def test_unbound_external_program_or_changed_runtime_export_cannot_be_signed(self):
        deps,interpreter,_,paths=self.launcher_fixture()
        missing=dict(deps);missing.pop(str(paths['mcookie']))
        with self.assertRaises(ValueError):state.observe_launcher_environment(missing,self.root,self.config,interpreter)
        for key,value in [('OMP_NUM_THREADS','99'),('PYTHONPATH',str(self.root/'unreviewed')),
                          ('CONDA_PREFIX',str(self.root/'other_env'))]:
            with self.subTest(key=key),patch.dict(os.environ,{key:value}),self.assertRaises(ValueError):
                state.observe_launcher_environment(deps,self.root,self.config,interpreter)


class RuntimeInventoryStateTests(unittest.TestCase):
    def setUp(self):
        from tests.test_multitask_baseline_dependencies import BaselineDependencyInventoryTests
        self.fixture=BaselineDependencyInventoryTests();self.fixture.setUp()
        self.path=self.fixture.file('native_runtime/system/settings.dat',encoded())
        self.override=patch.object(state,'STATE_FILE',self.path.resolve());self.override.start()
        for name in ('pyrep.py','backend/sim.py','backend/simConst.py'):
            self.fixture.file(self.fixture.repo/'third_party/PyRep/pyrep'/name,b'# native CPU API source')
        evidence=self.fixture.file('marker_evidence.json',b'{}')
        self.permission=self.fixture.file('marker_permission.json',json.dumps(state.draft_permission(
            self.fixture.root,self.fixture.config,self.fixture.pointer(evidence))).encode())

    def tearDown(self):self.override.stop();self.fixture.tearDown()

    def test_only_exact_marker_is_excluded_and_actual_bytes_are_observed(self):
        unknown=self.fixture.file('native_runtime/system/unknown_marker.dat',b'ordinary immutable file')
        setting=self.fixture.file('native_runtime/system/usrset.txt',b'physics configuration')
        result=self.fixture.run_inventory(mutable_state_permission=self.fixture.pointer(self.permission))
        runtime=result['dependency_groups']['runtime_and_extension_assets']
        self.assertNotIn(str(self.path),runtime)
        self.assertIn(str(unknown),runtime);self.assertIn(str(setting),runtime)
        self.assertEqual(result['runtime_inventory']['excluded_state_paths'],[str(self.path)])
        self.assertEqual(result['mutable_state_observations']['before']['parsed']['value'],'No')
        self.assertFalse(result['approved'])

    def test_unknown_new_runtime_file_during_inventory_is_rejected(self):
        from our_mani.multitask import baseline_dependencies as inventory
        original=inventory._sha;changed=[]
        def changed_directory(path):
            value=original(path)
            if Path(path)==self.fixture.runtime/'libcoppeliaSim.so' and not changed:
                self.fixture.file('native_runtime/system/new_unknown_marker.dat',encoded());changed.append(True)
            return value
        with patch.object(inventory,'_sha',side_effect=changed_directory),self.assertRaisesRegex(ValueError,'directory file set changed'):
            self.fixture.run_inventory(mutable_state_permission=self.fixture.pointer(self.permission))

    def test_extra_real_setting_in_declared_marker_is_rejected(self):
        self.path.write_bytes(encoded()+b'physics=changed')
        with self.assertRaises(ValueError):
            self.fixture.run_inventory(mutable_state_permission=self.fixture.pointer(self.permission))

    def test_historical_producer_plan_bytes_stay_frozen_without_reinterpreting_old_inputs_as_live_code(self):
        old=self.fixture.file('old_producer_plan.json',json.dumps(dict(schema='old_TRAIN_diagnostic_plan',
            inputs_sha256={'our_mani/multitask/simulation.py':'0'*64})).encode())
        result=self.fixture.run_inventory(calibration_inputs=[self.fixture.pointer(old)],
            mutable_state_permission=self.fixture.pointer(self.permission))
        self.assertEqual(result['gaps'],[])
        self.assertEqual(result['dependency_groups']['calibration_inputs'][str(old)],file_sha(old))
        receipt=result['retained_historical_metadata_not_replayed']
        self.assertEqual(len(receipt),1);self.assertEqual(receipt[0]['path'],str(old))


if __name__=='__main__':unittest.main()
