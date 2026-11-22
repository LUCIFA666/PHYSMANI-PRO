"""Temporary1204 contracts only; no real preflight, lease, camera or Native."""
from contextlib import contextmanager,nullcontext,redirect_stderr
from copy import deepcopy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock,patch

from our_mani.multitask import train_public_held_block_development_v1204 as subject
from scripts import run_TRAIN_public_held_block_development_v1204_r10 as worker

PROJECT=Path(__file__).resolve().parents[1]


def args_for(*,check=True,execute=False):
    task,raw,method=subject.FIXED_CASES[0]
    return NS(check=check,execute=execute,task=task,raw_episode_id=raw,method=method,split='train',
        config=None,frozen_inputs=None,out=subject.ARTIFACT_PREFIX+'fixed0',
        runtime=subject.RUNTIME_PREFIX+'fixed0',overlay_lock=subject.SETUP_PREFIX+'source_overlay_lock.json')


class Fixture:
    def __init__(self,root):
        self.root=Path(root);self.frozen='src/frozen_parent.py'
        inherited={self.frozen:hashlib.sha256(b'frozen parent').hexdigest()}
        self.parent=dict(integration_complete=True,files=inherited)
        for name in subject.source_roster(inherited):
            file=self.root/name;file.parent.mkdir(parents=True,exist_ok=True)
            if name in (subject.FACTORY_SOURCE,subject.FACTORY_TEST):file.write_bytes((PROJECT/name).read_bytes())
            else:file.write_bytes(b'frozen parent' if name==self.frozen else b'synthetic source')
        self.ref=dict(path=subject.PARENT_LOCK,sha256=subject.PARENT_LOCK_SHA)
    def pinned(self):return patch.object(subject,'_parent_lock',return_value=(deepcopy(self.parent),self.ref))
    def files(self):return subject._current_sources(self.root,self.parent)[1]
    def evidence(self,**changes):
        files=self.files();value=dict(schema=subject.CPU_SCHEMA,complete=True,actual_tool_exit_observed=True,
            returncode=0,tests_run=1,skipped=0,native_started=False,GPU_started=False,
            source_hashes_before=files,source_hashes_after=files,parent_source_overlay_lock=self.ref)
        value.update(changes);name=subject.SETUP_PREFIX+'control_CPU_evidence.json'
        file=self.root/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text(json.dumps(value))
        return subject.pointer(self.root,name)
    def lock(self,ready=False):
        evidence=self.evidence() if ready else None
        with self.pinned():return subject.write_source_lock(self.root,args_for().overlay_lock,
            integration_complete=ready,cpu_evidence_ref=evidence)
    def chosen(self):
        a=args_for();route=dict(task=a.task,method=a.method,GPU_required=False,
            configuration=dict(path='config/original.json',sha256='config'),
            frozen_inputs=dict(path=subject.parent.TRAIN_INPUTS,sha256=subject.parent.TRAIN_INPUTS_SHA))
        return dict(task=a.task,raw_episode_id=a.raw_episode_id,method=a.method,original_route=route,
            original_route_sha256=subject.digest(route),original_first_four=['episode0','episode1','episode105','episode106'])


class SourceAndSelectionContracts(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.f=Fixture(self.temp.name)
    def test_source_closure_includes_all_inherited_imports_and1203_source_test(self):
        names=subject.source_roster({name:'sha' for name in subject.parent.OVERLAY_SOURCES})
        self.assertTrue(set(subject.parent.OVERLAY_SOURCES)<=set(names))
        self.assertTrue(set(subject.candidate.SOURCES)<=set(names))
        self.assertTrue(set(subject.EXPLICIT_IMPORT_SOURCES)<=set(names))
        for name in (subject.SOURCE,subject.WORKER,subject.TEST_SOURCE,subject.FACTORY_SOURCE,subject.FACTORY_TEST,subject.PARENT_LOCK):self.assertIn(name,names)
        self.assertEqual(len(names),len(set(names)))
    def test_static_six_are_original_TRAIN_metadata_members_only(self):
        roster=json.loads(subject.parent.raw(PROJECT,subject.parent.MANIFEST,subject.parent.MANIFEST_SHA))
        for task,raw,method in subject.FIXED_CASES:
            first=[row['raw_episode_id'] for row in roster['episodes'] if row['task']==task][:4]
            self.assertIn(raw,first);self.assertIn(method,subject.METHODS)
        self.assertEqual(len(subject.FIXED_CASES),6)
    def test_wrong_split_or_unregistered_source_is_rejected_before_parent_read(self):
        with patch.object(subject.parent,'selection',side_effect=AssertionError('must reject before source read')):
            for field,value in [('split','validation'),('split','test'),('method','physmani'),('raw_episode_id','episode999'),('task','moving_basketball_in_hoop')]:
                a=args_for();setattr(a,field,value)
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):subject.selection(self.f.root,a)
    def test_all_six_pass_exact_original_route_arguments_without_replacement(self):
        with patch.object(subject.parent,'selection',return_value=self.f.chosen()) as parent:
            for i,(task,raw,method) in enumerate(subject.FIXED_CASES):
                a=args_for();a.task,a.raw_episode_id,a.method=task,raw,method
                result=subject.selection(self.f.root,a);self.assertEqual(result['fixed_development_index'],i)
                self.assertIs(parent.call_args.args[1],a)
    def test_pending_lock_is_source_complete_but_never_Native_ready(self):
        self.f.lock()
        with self.f.pinned():locked,ref=subject.read_source_lock(self.f.root,args_for().overlay_lock)
        self.assertFalse(locked['integration_complete']);self.assertFalse(locked['borrowed_frozen_physical_graph_identity'])
        self.assertEqual(locked['factory']['source'],subject.FACTORY_SOURCE)
        with self.assertRaises(ValueError):subject.require_native_ready(dict(source_overlay=locked,root_explicit_execute=True))
    def test_complete_lock_requires_actual_observed_stable_CPU_bytes(self):
        self.f.lock(True)
        with self.f.pinned():locked,ref=subject.read_source_lock(self.f.root,args_for().overlay_lock)
        subject.require_native_ready(dict(source_overlay=locked,root_explicit_execute=True))
        with self.assertRaises(ValueError):subject.require_native_ready(dict(source_overlay=locked,root_explicit_execute=False))
        with self.f.pinned(),self.assertRaises(FileExistsError):subject.write_source_lock(self.f.root,args_for().overlay_lock)
    def test_invalid_CPU_exit_skip_GPU_or_source_drift_cannot_activate(self):
        for key,value in [('actual_tool_exit_observed',False),('returncode',False),('skipped',1),('native_started',True),('GPU_started',True),('source_hashes_after',{})]:
            evidence=self.f.evidence(**{key:value})
            with self.subTest(key=key),self.f.pinned(),self.assertRaises(ValueError):
                subject.write_source_lock(self.f.root,args_for().overlay_lock,integration_complete=True,cpu_evidence_ref=evidence)
    def test_frozen_dependency_and1203_source_drift_are_rejected(self):
        for name in (self.f.frozen,subject.FACTORY_SOURCE,subject.FACTORY_TEST):
            file=self.f.root/name;before=file.read_bytes();file.write_bytes(b'changed')
            try:
                with self.subTest(name=name),self.f.pinned(),self.assertRaises(ValueError):subject.write_source_lock(self.f.root,args_for().overlay_lock)
            finally:file.write_bytes(before)
    def test_lock_cannot_change_fixed_roster_or_claim_formal_graph_acceptance(self):
        self.f.lock();file=self.f.root/args_for().overlay_lock;original=json.loads(file.read_text())
        for key,value in [('fixed_cases',[]),('formal_TRAIN192_acceptance_granted',True),('borrowed_frozen_physical_graph_identity',True),('source_outcome_filtering',True)]:
            changed=deepcopy(original);changed[key]=value;file.write_text(json.dumps(changed))
            with self.subTest(key=key),self.f.pinned(),self.assertRaises(ValueError):subject.read_source_lock(self.f.root,args_for().overlay_lock)
    def test_wrong_source_lock_and_CPU_pointer_roles_do_not_read_VAL(self):
        with patch.object(subject,'raw',side_effect=AssertionError('role must reject before read')):
            with self.assertRaises(ValueError):subject.read_source_lock(self.f.root,'val_package_compressed/x/data.h5')
            with self.assertRaises(ValueError):subject._cpu_evidence(self.f.root,dict(path='artifacts/framework/protocol/validation_manifest.json',sha256='a'*64),{})


class PrepareAndScopeContracts(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.f=Fixture(self.temp.name)
    def test_original742_preflight_entry_model_inputs_and_new1203_identity(self):
        self.f.lock();a=args_for();substrate=importlib.import_module('our_mani.multitask.rml_case_preflight_v742')
        def old_prepare(root,forwarded,entry):
            self.assertEqual(entry,root/subject.LEGACY_WORKER);self.forwarded=forwarded
            return dict(identity=dict(sha256='original_identity'),config=dict(baseline_repo='/frozen/native'),
                args=forwarded,out=root/a.out,component_domain=NS(receipt=lambda:{'purpose':'train_qualification'}),model_preflight={})
        binding=dict(source_sha256=subject.FACTORY_SHA,sources_sha256={subject.FACTORY_SOURCE:subject.FACTORY_SHA})
        with self.f.pinned(),patch.object(subject,'selection',return_value=self.f.chosen()),\
             patch.object(substrate,'prepare',side_effect=old_prepare),patch.object(substrate,'outcome_protocol_binding',return_value={'original741':True}),\
             patch.object(subject.candidate,'binding',return_value=binding):
            prepared=subject.prepare(self.f.root,a)
        self.assertEqual(self.forwarded.split,'train');self.assertEqual(self.forwarded.manifest,subject.parent.MANIFEST)
        self.assertEqual(self.forwarded.frozen_inputs,subject.parent.TRAIN_INPUTS);self.assertIsNone(self.forwarded.source_inputs)
        self.assertNotEqual(prepared['identity']['sha256'],'original_identity')
        self.assertEqual(prepared['execution_protocol']['factory']['source'],subject.FACTORY_SOURCE)
        self.assertFalse(prepared['execution_protocol']['borrowed_frozen_physical_graph_identity'])
    def test_annotation_preserves_every_True_False_fatal_and_motion_diagnostic(self):
        p=dict(execution_protocol=dict(schema=subject.SCHEMA),source_overlay_pointer=dict(path='lock',sha256='x'))
        for success,fatal in [(True,False),(False,False),(False,True)]:
            old=dict(version='original744',official_task_success=success,native_terminal_failure=fatal,motion_pass=False)
            new=subject.annotated_output(p,'case.json',old)
            self.assertIs(new['official_task_success'],success);self.assertIs(new['native_terminal_failure'],fatal)
            self.assertFalse(new['motion_pass']);self.assertEqual(new['version'],subject.SCHEMA+'_case')
            self.assertFalse(new['formal_TRAIN192_acceptance_granted']);self.assertEqual(old['version'],'original744')
        with self.assertRaises(ValueError):subject.annotated_output(p,'case.json',dict(physical_policy_sha256='12c96'))
    def scoped(self,*,throws=False):
        a=args_for(check=False,execute=True);out=self.f.root/a.out;events=[]
        p=dict(source_overlay=dict(integration_complete=True),root_explicit_execute=True,config=dict(baseline_repo='/frozen/native'),
            args=a,out=out,model_preflight=dict(actual_shared16_block_adapter={'checkpoint':'671'}),
            component_domain=NS(receipt=lambda:{'split':'train'}),execution_protocol=dict(schema=subject.SCHEMA),source_overlay_pointer={'path':'newlock','sha256':'sha'})
        runtime=NS(__file__=str(self.f.root/subject.LEGACY_RUNTIME),atomic_json=Mock(return_value='written'),_geometry=object(),_baseline=object())
        original=(runtime.atomic_json,runtime._geometry,runtime._baseline);guard=NS(require_permit=Mock(return_value={'resource_only':True}))
        @contextmanager
        def factory(*args):
            events.append('new1203_construct')
            try:yield ('frontend','policy','executor')
            finally:events.append('new1203_restore')
        real_import=importlib.import_module
        def route(name):return guard if name==subject.GUARD else real_import(name)
        with patch.object(subject,'verify'),patch.object(subject.importlib,'import_module',side_effect=route),\
             patch.object(subject.candidate,'integrated_block_components',side_effect=factory),\
             patch.object(subject.candidate,'binding',return_value={'source_sha256':subject.FACTORY_SHA}):
            if throws:
                with self.assertRaisesRegex(RuntimeError,'original_error'):
                    with subject.runtime_scope(self.f.root,p,runtime,permit=object()):
                        result=runtime._geometry(p,object(),[]);raise RuntimeError('original_error')
            else:
                with subject.runtime_scope(self.f.root,p,runtime,permit=object()):
                    result=runtime._geometry(p,object(),[])
                    with self.assertRaises(ValueError):runtime._geometry(p,object(),[])
                    runtime.atomic_json(out/'case.json',dict(version='old',official_task_success=False,native_terminal_failure=False))
                    with self.assertRaises(ValueError):runtime.atomic_json(out/'case.json',{})
        self.assertEqual((runtime.atomic_json,runtime._geometry,runtime._baseline),original)
        self.assertEqual(events,['new1203_construct','new1203_restore']);self.assertIsNone(result[0])
        self.assertEqual(p['runtime_scope_restoration']['geometry_factory_calls'],1)
        self.assertTrue(p['runtime_scope_restoration']['restored']);self.assertEqual(p['runtime_scope_restoration']['extra_native_calls'],0)
        return p
    def test_only_owned_new_factory_and_writer_are_patched_then_restored(self):self.scoped()
    def test_original_exception_propagates_with_all_scopes_restored(self):
        p=self.scoped(throws=True);self.assertEqual(p['runtime_scope_restoration']['original_exception_type'],'RuntimeError')
    def test_scope_cannot_enter_without_explicit_execute_or_original1106_permit(self):
        p=dict(source_overlay=dict(integration_complete=True),root_explicit_execute=False)
        with self.assertRaises(ValueError):
            with subject.runtime_scope(self.f.root,p,NS(),permit=None):pass
        p['root_explicit_execute']=True
        guard=NS(require_permit=Mock(side_effect=ValueError('formal owner still holds global locks')))
        with patch.object(subject.importlib,'import_module',return_value=guard),self.assertRaises(ValueError):
            with subject.runtime_scope(self.f.root,p,NS(),permit=None):pass


class WorkerContracts(unittest.TestCase):
    def test_default_is_inert_before_selection_guard_or_any_dataset_read(self):
        a=worker.parser().parse_args([])
        with patch.object(subject,'selection',side_effect=AssertionError('default must not select')),\
             patch.object(worker,'resource_guard',side_effect=AssertionError('default must not acquire lease')):
            result=worker.run_case(a)
        self.assertFalse(result['enabled']);self.assertFalse(result['Native_started']);self.assertEqual(len(result['fixed_cases']),6)
    def test_modes_are_mutually_exclusive_and_never_implicitly_Native(self):
        a=args_for(check=True,execute=True)
        with self.assertRaises(ValueError):worker.run_case(a)
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):worker.parser().parse_args(['--check','--execute'])
    def test_prior_CPU_role_rejects_foreign_source_before_any_read(self):
        a=args_for();a.runtime='val_package_compressed/foreign'
        with patch.object(subject,'pointer',side_effect=AssertionError('no foreign source read')):
            with self.assertRaises(ValueError):worker.require_prior_CPU(PROJECT,a,{})
    def test_prior_CPU_must_match_fixed_source_lock_and_dead_inner_and_outer(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);a=args_for(check=False,execute=True);lock={'path':'newlock','sha256':'sha'}
            for name in (subject.WORKER,):
                file=root/name;file.parent.mkdir(parents=True);file.write_text('source')
            cpu=dict(schema=subject.SCHEMA,complete=True,resource_lease_scope_exited=True,source_overlay_lock=lock,
                actual_worker=subject.pointer(root,subject.WORKER),selection=dict(task=a.task,raw_episode_id=a.raw_episode_id,method=a.method),
                process=dict(pid=2147482000,start_ticks=1),frozen_OS_environment=dict(outer_wrapper_pid=2147482001,outer_wrapper_start_ticks=2),
                execution=dict(mode='CPU_check',complete=True,actual_split='train',native_started=False,GPU_started=False,physical_factory_constructed=False))
            file=root/(a.runtime+'/CPU_check/worker_sidecar.json');subject.write_json(file,cpu)
            worker.require_prior_CPU(root,a,lock)
            with self.assertRaises(ValueError):worker.require_prior_CPU(root,a,dict(lock,sha256='changed'))
            with patch.object(worker,'_dead',return_value=False),self.assertRaises(ValueError):worker.require_prior_CPU(root,a,lock)
    def test_current_formal_lock_rejection_never_reaches_new1203_or_Native(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);a=args_for(check=False,execute=True)
            guard=NS(reserved_resources_after_TEST=Mock(side_effect=ValueError('formal queue owns1106')))
            with patch.object(worker,'ROOT',root),patch.object(subject,'selection',return_value={}),\
                 patch.object(subject,'read_source_lock',return_value=(dict(integration_complete=True),{})),\
                 patch.object(worker,'require_prior_CPU',return_value=({},{})),patch.object(worker,'resource_guard',return_value=guard),\
                 patch.object(subject,'prepare',side_effect=AssertionError('busy lease must block prepare')):
                with self.assertRaisesRegex(ValueError,'formal queue'):worker.run_case(a,root=root)
    def fake_run(self,*,check,success=False):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);root=Path(self.temp.name);a=args_for(check=check,execute=not check)
        for name in (subject.WORKER,subject.SOURCE):
            file=root/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text('synthetic worker')
        lock=dict(path='new_source_lock',sha256='lock');chosen=dict(task=a.task,raw_episode_id=a.raw_episode_id,method=a.method)
        prepared=dict(identity={'source':'new1204'},source_substrate_identity={'source':'original742'},
            component_domain=NS(receipt=lambda:{'purpose':'train_qualification'}),model_preflight={'actual671':True},
            execution_protocol=dict(schema=subject.SCHEMA,unchanged_action_outcome_sampling={'original741':True}),
            source_overlay=dict(integration_complete=True),out=root/a.out,source_overlay_pointer=lock,runtime_scope_restoration=None)
        @contextmanager
        def resources(*args,**kw):yield object()
        guard=NS(reserved_resources_after_TEST=resources,require_permit=Mock(return_value={'same_process':True}))
        edges=NS(declared_edge_scope=nullcontext,evaluator_scope=lambda evaluator:nullcontext())
        runtime=NS(run=Mock())
        def actual_run(p):
            subject.write_json(root/a.out/'case.json',dict(version=subject.SCHEMA+'_case',execution_protocol=p['execution_protocol'],
                source_overlay_lock=lock,identity=p['identity'],process_id=os.getpid(),cleanup_completed=True,
                official_task_success=success,native_terminal_failure=False))
        runtime.run.side_effect=actual_run
        @contextmanager
        def scope(*args,**kw):
            yield
            prepared['runtime_scope_restoration']=dict(complete=True,restored=True)
        real_import=importlib.import_module
        def route(name):
            if name=='our_mani.multitask.frozen_input_edges_v463':return edges
            if name=='scripts.evaluate_multitask_case_r10':return NS()
            if name=='scripts.run_task_executor_smoke_author_suite16_v744_r10':return runtime
            if name=='our_mani.multitask.evaluation':return NS(validate_completed_case=Mock())
            if name=='our_mani.multitask.repaired_readiness_v1144':return NS(validate_sampling_case=Mock())
            return real_import(name)
        with patch.object(worker,'ROOT',root),patch.object(subject,'selection',return_value=chosen),\
             patch.object(subject,'read_source_lock',return_value=(dict(integration_complete=True),lock)),\
             patch.object(worker,'require_prior_CPU',return_value=(dict(execution=dict(identity=prepared['identity'])),{'path':'CPU','sha256':'cpu'})),\
             patch.object(worker,'resource_guard',return_value=guard),patch.object(worker,'environment_proof',return_value={'actual820':True}),\
             patch.object(subject,'prepare',return_value=prepared),patch.object(subject,'verify'),\
             patch.object(subject,'runtime_scope',side_effect=scope),patch.object(worker.importlib,'import_module',side_effect=route):
            result=worker.run_case(a,root=root)
        self.assertTrue(result['resource_lease_scope_exited']);self.assertFalse(result['formal_TRAIN192_acceptance_granted'])
        self.assertFalse(result['borrowed_frozen_physical_graph_identity'])
        return result,runtime
    def test_CPU_worker_never_imports_or_constructs_Native(self):
        result,runtime=self.fake_run(check=True)
        runtime.run.assert_not_called();self.assertFalse(result['execution']['native_started'])
        self.assertFalse(result['execution']['physical_factory_constructed'])
    def test_native_false_is_a_complete_development_result_not_infrastructure(self):
        result,runtime=self.fake_run(check=False,success=False)
        runtime.run.assert_called_once();self.assertTrue(result['execution']['complete'])
        self.assertFalse(result['execution']['native_success']);self.assertTrue(result['execution']['native_false_preserved'])
        self.assertFalse(result['execution']['formal_TRAIN192_acceptance_granted'])
    def test_native_true_keeps_the_same_control_closure_and_no_formal_qualification(self):
        result,runtime=self.fake_run(check=False,success=True)
        runtime.run.assert_called_once();self.assertTrue(result['execution']['native_success'])
        self.assertFalse(result['formal_VAL_TEST_qualification'])


if __name__=='__main__':unittest.main()
