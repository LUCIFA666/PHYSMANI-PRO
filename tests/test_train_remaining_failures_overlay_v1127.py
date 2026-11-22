"""CPU synthetic identity/lock/owned-scope tests; no lease or simulator run."""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from our_mani.multitask import train_remaining_failures_overlay_v1127 as overlay
from scripts import diagnose_TRAIN_remaining_failures_v1127_r10 as worker


def args_for(task='insert_onto_rotating_peg'):
    return NS(task=task, raw_episode_id='episode101' if task=='insert_onto_rotating_peg' else 'episode10',
        method='cv_geometry', config=None, frozen_inputs=None,
        out=overlay.ARTIFACT_PREFIX+'case', runtime=overlay.RUNTIME_PREFIX+'case',
        diagnostic_lock=overlay.SETUP_PREFIX+'lock.json', check=True)


class DiagnosticSelection(unittest.TestCase):
    def test_exact_two_sources_only_and_parent_original_route_preserved(self):
        chosen={'original_route':{'unchanged745':True}}
        with patch.object(overlay.parent,'selection',return_value=chosen) as parent:
            for task in ('insert_onto_rotating_peg','beat_the_rotating_buzz'):
                self.assertIs(overlay.selection(Path('/fixture'),args_for(task)),chosen)
            for field,value in (('method','context_geometry'),('method','physmani'),
                                ('raw_episode_id','episode102'),('task','insert_onto_rotating_peg_high_speed')):
                args=args_for();setattr(args,field,value)
                with self.assertRaises(ValueError):overlay.selection(Path('/fixture'),args)
        self.assertEqual(parent.call_count,2)

    def test_prepare_new_identity_honest_actual_worker_nested_parent_and_namespace_restore(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);args=args_for();before=overlay.parent.ARTIFACT_PREFIX;seen=[]
            inherited=dict(identity={'sha256':'parent1122','config_sha256':'config'},
                execution_protocol={'schema':overlay.parent.SCHEMA,'policy':'1121'},
                source_overlay_pointer={'path':overlay.PARENT_LOCK,'sha256':overlay.PARENT_LOCK_SHA})
            locked=dict(files={s:'sha' for s in (overlay.SOURCE,overlay.WORKER,overlay.TEST_SOURCE)})
            ref={'path':args.diagnostic_lock,'sha256':'newlock'}
            def parent_prepare(current,forwarded):
                seen.append((overlay.parent.ARTIFACT_PREFIX,forwarded.overlay_lock));return deepcopy(inherited)
            with patch.object(overlay,'selection',return_value={}), \
                 patch.object(overlay,'read_diagnostic_lock',return_value=(locked,ref)), \
                 patch.object(overlay.parent,'prepare',side_effect=parent_prepare), \
                 patch.object(overlay.parent,'pointer',side_effect=lambda root,name:dict(path=name,sha256='sha')):
                result=overlay.prepare(root,args)
            self.assertEqual(seen,[(overlay.ARTIFACT_PREFIX,overlay.PARENT_LOCK)])
            self.assertEqual(overlay.parent.ARTIFACT_PREFIX,before)
            self.assertEqual(result['execution_protocol']['actual_worker']['path'],overlay.WORKER)
            self.assertEqual(result['execution_protocol']['inherited_1122_execution_protocol'],inherited['execution_protocol'])
            self.assertEqual(result['identity']['source_overlay_lock_sha256'],'newlock')
            self.assertEqual(result['inherited_1122_identity']['sha256'],'parent1122')
            self.assertNotEqual(result['identity']['sha256'],'parent1122')
            self.assertEqual(result['identity']['execution_protocol_schema'],overlay.SCHEMA)

    def test_parent_prepare_exception_preserved_and_namespace_restored(self):
        args=args_for();before=overlay.parent.ARTIFACT_PREFIX;error=RuntimeError('source preflight')
        with TemporaryDirectory() as temporary, patch.object(overlay,'selection',return_value={}), \
             patch.object(overlay,'read_diagnostic_lock',return_value=(
                 {'files':{s:'sha' for s in (overlay.SOURCE,overlay.WORKER,overlay.TEST_SOURCE)}},{})), \
             patch.object(overlay.parent,'prepare',side_effect=error):
            with self.assertRaises(RuntimeError) as raised:overlay.prepare(Path(temporary),args)
        self.assertIs(raised.exception,error)
        self.assertEqual(overlay.parent.ARTIFACT_PREFIX,before)


class DiagnosticLocks(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        for name in ('source.py','worker.py','tests.py'):(self.root/name).write_text(name)
        self.sources=('source.py','worker.py','tests.py','pending_trace.py')
        for name,value in (('OVERLAY_SOURCES',self.sources),('PASSIVE_PINS',{})):
            patcher=patch.object(overlay,name,value);patcher.start();self.addCleanup(patcher.stop)
        patcher=patch.object(overlay,'_parent_lock',return_value=({},dict(path=overlay.PARENT_LOCK,sha256=overlay.PARENT_LOCK_SHA)))
        patcher.start();self.addCleanup(patcher.stop)

    def test_pending_honest_missing_no_native_or_overwrite(self):
        name=overlay.SETUP_PREFIX+'pending.json'
        overlay.write_diagnostic_lock(self.root,name)
        lock,reference=overlay.read_diagnostic_lock(self.root,name)
        self.assertEqual(reference['path'],name)
        self.assertEqual(lock['unpinned_sources'],['pending_trace.py'])
        with self.assertRaises(ValueError):overlay.require_native_ready(dict(diagnostic_overlay=lock))
        with self.assertRaises(FileExistsError):overlay.write_diagnostic_lock(self.root,name)

    def test_complete_lock_requires_all_sources_exact_cpu_map_and_parent_lock(self):
        name=overlay.SETUP_PREFIX+'lock.json'
        with self.assertRaises(ValueError):overlay.write_diagnostic_lock(self.root,name,integration_complete=True)
        (self.root/'pending_trace.py').write_text('ready')
        files={s:overlay.parent.pointer(self.root,s)['sha256'] for s in self.sources}
        cpu=overlay.SETUP_PREFIX+'cpu.json'
        overlay.parent.write_json(self.root/cpu,dict(schema=overlay.CPU_EVIDENCE_SCHEMA,complete=True,
            tests_exit_code=0,native_started=False,GPU_started=False,source_hashes=files,
            parent_source_overlay_lock=dict(path=overlay.PARENT_LOCK,sha256=overlay.PARENT_LOCK_SHA)))
        overlay.write_diagnostic_lock(self.root,name,integration_complete=True,
                                      integration_cpu_evidence=overlay.parent.pointer(self.root,cpu))
        lock,_=overlay.read_diagnostic_lock(self.root,name)
        overlay.require_native_ready(dict(diagnostic_overlay=lock,source_overlay={'integration_complete':True}))
        (self.root/'worker.py').write_text('changed')
        with self.assertRaises(ValueError):overlay.read_diagnostic_lock(self.root,name)

    def test_cpu_boolean_exit_or_missing_parent_cannot_activate(self):
        (self.root/'pending_trace.py').write_text('ready')
        files={s:overlay.parent.pointer(self.root,s)['sha256'] for s in self.sources}
        for i,(code,parent) in enumerate(((False,{'path':overlay.PARENT_LOCK,'sha256':overlay.PARENT_LOCK_SHA}),(0,{}))):
            cpu=overlay.SETUP_PREFIX+f'cpu{i}.json'
            overlay.parent.write_json(self.root/cpu,dict(schema=overlay.CPU_EVIDENCE_SCHEMA,complete=True,
                tests_exit_code=code,native_started=False,GPU_started=False,source_hashes=files,parent_source_overlay_lock=parent))
            with self.assertRaises(ValueError):overlay.write_diagnostic_lock(self.root,overlay.SETUP_PREFIX+f'lock{i}.json',
                integration_complete=True,integration_cpu_evidence=overlay.parent.pointer(self.root,cpu))


class OwnedScopes(unittest.TestCase):
    def test_parent_annotation_called_once_before_single_write_and_new_version_false_preserved(self):
        for task in ('insert_onto_rotating_peg','beat_the_rotating_buzz'):
            with self.subTest(task=task):
                self.scoped_case(task,raise_source=False)

    def test_native_exception_propagates_and_all_owned_scopes_restore(self):
        self.scoped_case('insert_onto_rotating_peg',raise_source=True)

    def test_observer_receipt_error_cannot_mask_original_native_exception(self):
        self.scoped_case('insert_onto_rotating_peg',raise_source=True,receipt_error=True)

    def scoped_case(self,task,*,raise_source,receipt_error=False):
        original_geometry=object();writes=[];order=[]
        runtime=NS(_geometry=original_geometry,atomic_json=lambda path,value:writes.append(value))
        original_writer=runtime.atomic_json;original_annotation=overlay.parent.annotated_output
        prepared=dict(args=args_for(task),out=Path('/new'),source_overlay_pointer={'path':'parent','sha256':'parent'},
            diagnostic_overlay_pointer={'path':'new','sha256':'new'},execution_protocol={'schema':overlay.SCHEMA})
        @contextmanager
        def parent_scope(root,current,module):
            order.append('parent_enter');writer=module.atomic_json
            def write(path,value):return writer(path,overlay.parent.annotated_output(current,path,value))
            with patch.object(module,'_geometry',object()),patch.object(module,'atomic_json',write):
                try:yield
                finally:order.append('parent_exit')
        @contextmanager
        def trace_scope(module,events,**limits):
            self.assertIsNot(module._geometry,original_geometry);order.append('trace_enter')
            try:yield
            finally:events.update(restored=True,dropped_frames=0,errors=[]);order.append('trace_exit')
        class Tap:
            def __init__(self,*args):order.append('tap_construct')
            def __enter__(self):order.append('tap_enter');return self
            def __exit__(self,*args):order.append('tap_exit');return False
            def receipt(self):
                if receipt_error:raise ValueError('receipt failure')
                return dict(restored=True,dropped_events=0,observer_errors=0)
        trace=NS(public_execution_trace_scope=trace_scope);profile=NS(PassiveRingSuccessTap=Tap,PassiveBuzzFailureTap=Tap)
        def imported(name):return trace if name.endswith('v1126') else profile
        events={};error=RuntimeError('original Native exception')
        with patch.object(overlay,'require_native_ready'),patch.object(overlay,'verify'), \
             patch.object(overlay.parent,'runtime_scope',side_effect=parent_scope), \
             patch.object(overlay.importlib,'import_module',side_effect=imported) as modules, \
             patch.object(overlay.parent,'annotated_output',wraps=original_annotation) as annotation:
            if raise_source:
                with self.assertRaises(RuntimeError) as raised:
                    with overlay.runtime_scope(Path('/fixture'),prepared,runtime,events):raise error
                self.assertIs(raised.exception,error)
            else:
                with overlay.runtime_scope(Path('/fixture'),prepared,runtime,events):
                    runtime.atomic_json(Path('/new/case.json'),{'version':'old744','official_task_success':False})
                self.assertEqual(annotation.call_count,1)
                self.assertEqual(len(writes),1)
                self.assertIs(writes[0]['official_task_success'],False)
                self.assertEqual(writes[0]['version'],overlay.SCHEMA+'_case')
                self.assertEqual(writes[0]['inherited_1122_diagnostic_parent_case_version'],overlay.parent.SCHEMA+'_case')
                self.assertEqual(writes[0]['source_overlay_lock'],prepared['diagnostic_overlay_pointer'])
            self.assertEqual(modules.call_count,2)
        self.assertIs(runtime._geometry,original_geometry);self.assertIs(runtime.atomic_json,original_writer)
        self.assertIs(overlay.parent.annotated_output,original_annotation)
        self.assertIs(events['restored'],not receipt_error)
        self.assertEqual(len(events['observer_finalization_errors']),int(receipt_error))
        self.assertEqual(order,['parent_enter','trace_enter','tap_construct','tap_enter','tap_exit','trace_exit','parent_exit'])


class DiagnosticWorker(unittest.TestCase):
    def test_CPU_check_never_installs_native_profile_trace_or_constructs_runtime(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);args=args_for();order=[]
            @contextmanager
            def scope(*args,**kwargs):
                order.append('scope_enter')
                try:yield 'permit'
                finally:order.append('scope_exit')
            guard=NS(reserved_resources_after_TEST=scope,require_permit=Mock(return_value={'actual':'resource'}))
            edges=NS(declared_edge_scope=scope,evaluator_scope=scope)
            parent_worker=NS(environment_proof=Mock(return_value={'same_process':'CPU'}))
            evaluator=NS()
            allowed={'our_mani.multitask.post_completed_TEST_resource_guard_v1106':guard,
                'scripts.run_TRAIN_author_aligned_full16_overlay_v1122_r10':parent_worker,
                'our_mani.multitask.frozen_input_edges_v463':edges,
                'scripts.evaluate_multitask_case_r10':evaluator}
            prepared=dict(identity={'sha256':'newdiagnostic'},inherited_1122_identity={'sha256':'old1122'},
                component_domain=NS(receipt=lambda:{'purpose':'train_qualification'}),
                model_preflight={'source_and_models':'read_only'},execution_protocol={'schema':overlay.SCHEMA})
            lock=dict(integration_complete=True,unpinned_sources=[])
            with patch.object(worker,'ROOT',root),patch.object(overlay,'selection',return_value={'task':args.task}), \
                 patch.object(overlay,'read_diagnostic_lock',return_value=(lock,{'path':args.diagnostic_lock,'sha256':'lock'})), \
                 patch.object(overlay,'prepare',return_value=prepared),patch.object(overlay,'verify'), \
                 patch.object(worker.importlib,'import_module',side_effect=lambda name:allowed[name]), \
                 patch.object(overlay,'runtime_scope',side_effect=AssertionError('no Native hooks')):
                result=worker.run_case(args,root=root)
            self.assertFalse(result['execution']['native_started'])
            self.assertFalse(result['execution']['GPU_started'])
            self.assertFalse(result['execution']['observers_installed'])
            self.assertFalse((root/args.out).exists())
            self.assertTrue((root/args.runtime/'CPU_check/worker_sidecar.json').exists())
            self.assertFalse((root/args.runtime/'native').exists())
            self.assertEqual(guard.require_permit.call_count,2)
            self.assertEqual(order,['scope_enter']*3+['scope_exit']*3)

    def test_pending_diagnostic_lock_rejects_native_before_resource_or_environment(self):
        args=args_for();args.check=False
        with patch.object(overlay,'selection',return_value={}), \
             patch.object(overlay,'read_diagnostic_lock',return_value=({'integration_complete':False},{})), \
             patch.object(worker.importlib,'import_module',side_effect=AssertionError('no lease/native')):
            with self.assertRaisesRegex(ValueError,'complete diagnostic CPU evidence'):
                worker.run_case(args)

    def test_mocked_native_false_case_saved_after_restoration_with_new_sidecar(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);args=args_for();args.check=False
            @contextmanager
            def scope(*args,**kwargs):yield 'permit'
            guard=NS(reserved_resources_after_TEST=scope,require_permit=Mock(return_value={}))
            edges=NS(declared_edge_scope=scope,evaluator_scope=scope)
            source={'path':args.diagnostic_lock,'sha256':'diagnostic'}
            inherited={'path':overlay.PARENT_LOCK,'sha256':overlay.PARENT_LOCK_SHA}
            prepared=dict(identity={'sha256':'newdiagnostic'},execution_protocol={'schema':overlay.SCHEMA},
                          source_overlay_pointer=inherited)
            @contextmanager
            def native_scope(current_root,current,runtime,events):
                try:yield
                finally:events.update(restored=True,passive_native_conditions={'restored':True},
                                      public_execution_trace={'restored':True})
            def run(current):
                overlay.parent.write_json(root/args.out/'case.json',dict(version=overlay.SCHEMA+'_case',
                    execution_protocol=prepared['execution_protocol'],source_overlay_lock=source,
                    parent_source_overlay_lock=inherited,cleanup_completed=True,process_id=os.getpid(),
                    official_task_success=False,native_terminal_failure=False))
            validator=Mock()
            allowed={'our_mani.multitask.post_completed_TEST_resource_guard_v1106':guard,
                'scripts.run_TRAIN_author_aligned_full16_overlay_v1122_r10':NS(environment_proof=lambda root:{}),
                'our_mani.multitask.frozen_input_edges_v463':edges,
                'scripts.evaluate_multitask_case_r10':NS(),
                'scripts.run_task_executor_smoke_author_suite16_v744_r10':NS(run=Mock(side_effect=run)),
                'our_mani.multitask.evaluation':NS(validate_completed_case=validator)}
            with patch.object(worker,'ROOT',root),patch.object(overlay,'selection',return_value={}), \
                 patch.object(overlay,'read_diagnostic_lock',return_value=({'integration_complete':True},source)), \
                 patch.object(overlay,'prepare',return_value=prepared),patch.object(overlay,'verify'), \
                 patch.object(worker.importlib,'import_module',side_effect=lambda name:allowed[name]), \
                 patch.object(overlay,'runtime_scope',side_effect=native_scope):
                result=worker.run_case(args,root=root)
            self.assertIs(result['execution']['native_success'],False)
            self.assertTrue(result['execution']['native_false_preserved'])
            self.assertEqual(result['schema'],overlay.SCHEMA)
            self.assertEqual(result['source_overlay_lock'],source)
            self.assertTrue((root/args.runtime/'native/worker_sidecar.json').exists())
            validator.assert_called_once()


if __name__=='__main__':unittest.main()
