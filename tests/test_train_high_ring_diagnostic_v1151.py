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

from our_mani.multitask import train_high_ring_diagnostic_v1151 as overlay
from scripts import diagnose_TRAIN_high_ring_v1151_r10 as worker

ROOT=Path(__file__).resolve().parents[1]


def args_for(task='insert_onto_rotating_peg_high_speed'):
    return NS(task=task, raw_episode_id='episode102' if task=='insert_onto_rotating_peg_high_speed' else 'episode10',
        method='cv_geometry', config=None, frozen_inputs=None,
        out=overlay.ARTIFACT_PREFIX+'case', runtime=overlay.RUNTIME_PREFIX+'case',
        diagnostic_lock=overlay.SETUP_PREFIX+'lock.json', check=True)


def quality_fixture(*,ticks=3,terminal=False):
    restored=dict(restored=True,restoration_error=None,scope_exception_type=None,errors=[],
        restoration_checks=[{'restored':True}],sysprofile_installed=False,original_return_or_exception_rewritten=False)
    frames=ticks+(0 if terminal and ticks>0 else 1)
    trace=dict(restored,schema='r10_evaluation_existing_public_execution_trace_v1126',
        max_frames=201,max_plans=24,geometry_calls=1,frame_calls=frames,frame_returns=frames,frame_exceptions=0,
        frames=[{'public_DTO':'only_existing_frame'} for _ in range(frames)],plan_calls=1,plan_returns=0,plan_exceptions=1,
        plans=[{'original_exception':{'type':'RuntimeError'}}],unknown_reasons={'unobserved_public_field':1})
    for key in ('recording_errors','frame_dropped','plan_dropped','plan_value_truncations','extra_env_observe_calls',
        'extra_native_getter_or_condition_calls','extra_prediction_calls','extra_path_step_or_FK_calls'):trace[key]=0
    methods=('frontend.ready','collision_port.readiness','source_policy.candidates','generator.observed_reference')
    rows=[dict(method=method,call_sequence=index+1,finish_sequence=index+1,original_returned=index!=3,
        **({'original_exception':{'type':'ValueError','reason':'original missing public reference'}} if index==3 else
            {'original_return':False if index<2 else []})) for index,method in enumerate(methods)]
    counts={method:dict(calls=1,returns=int(index!=3),exceptions=int(index==3)) for index,method in enumerate(methods)}
    probe=dict(restored,schema='r10_evaluation_existing_public_readiness_probe_v1134',max_events=1000,
        events=rows,method_call_counts=counts,geometry_calls=1,geometry_returns=1,geometry_exceptions=0,
        recording_errors=0,dropped_events=0,extra_readiness_reference_forecast_or_native_calls=0,unknown_reasons={})
    records=[{'event':'original_task_success_call'}]
    for ordinal in (None,0,1,2,3):
        records.extend(({'event':'original_condition_met_call'},
            {'event':'original_condition_met_return','registered_leaf_ordinal':ordinal,'result':[False,False]}))
    records.append(dict(event='original_task_success_return',original_four_detector_registration=True,
        leaf_returns=[dict(ordinal=index,result=[False,False]) for index in range(4)]))
    profile=dict(schema='r10_fixed_TRAIN_ring_passive_original_success_tree_v1125',
        task='insert_onto_rotating_peg_high_speed',method='cv_geometry',restored=True,scope_exception_type=None,
        existing_profiler_never_overwritten=True,physical_methods_patched=False,events=records,
        original_success_calls_observed=1,original_success_returns_observed=1,
        original_condition_calls_observed=5,original_condition_returns_observed=5)
    for key in ('dropped_events','observer_errors','exception_or_nonstandard_returns','registration_truncations','inner_sequence_drops',
        'extra_success_calls','extra_condition_calls','extra_physical_getter_calls'):profile[key]=0
    return dict(schema=overlay.SCHEMA+'_observers',restored=True,factory_restored=True,writer_restored=True,
        parent_annotation_restored=True,observer_finalization_errors=[],sysprofile_installed=True,
        additional_native_getters_or_success_or_condition_calls=0,public_execution_trace=dict(restored=True,trace_receipt=trace),
        readiness_probe=dict(restored=True,readiness_probe=probe),passive_native_conditions=profile)


class PassiveQuality(unittest.TestCase):
    def test_readyFalse_empty_candidates_and_original_exceptions_are_retained_data(self):
        value=overlay.require_passive_quality(quality_fixture(),ticks=3)
        self.assertEqual(value['public_frames'],4);self.assertEqual(value['original_plan_exceptions'],1)
        self.assertEqual(value['original_method_counts']['generator.observed_reference']['exceptions'],1)
        self.assertTrue(value['readyFalse_empty_candidates_and_original_helper_exceptions_are_data'])
        self.assertEqual(value['trace_unknown_reasons'],{'unobserved_public_field':1})

    def test_terminal_frame_omission_and_initial_terminal_do_not_add_observation(self):
        for ticks,expected in ((3,3),(0,1)):
            value=overlay.require_passive_quality(quality_fixture(ticks=ticks,terminal=True),ticks=ticks,terminal=True)
            self.assertEqual(value['public_frames'],expected)
        with self.assertRaises(ValueError):overlay.require_passive_quality(quality_fixture(terminal=True),ticks=3)

    def test_original_methods_not_reached_and_no_plans_remain_unknown_not_collection_errors(self):
        events=quality_fixture(ticks=0,terminal=True)
        probe=events['readiness_probe']['readiness_probe'];probe['events']=[]
        probe['method_call_counts']={name:dict(calls=0,returns=0,exceptions=0) for name in probe['method_call_counts']}
        trace=events['public_execution_trace']['trace_receipt'];trace.update(plan_calls=0,plan_returns=0,plan_exceptions=0,plans=[])
        value=overlay.require_passive_quality(events,ticks=0,terminal=True)
        self.assertEqual(len(value['unobserved_original_methods']),4);self.assertEqual(value['original_plans'],0)

    def test_error_drop_extra_calls_missing_restore_or_leaf_cannot_pass(self):
        for stream,key,bad in (('trace','recording_errors',1),('trace','frame_dropped',1),
            ('trace','extra_env_observe_calls',1),('probe','dropped_events',1),('probe','recording_errors',1),
            ('profile','extra_condition_calls',1),('profile','observer_errors',1),('profile','restored',False)):
            events=quality_fixture()
            target={'trace':events['public_execution_trace']['trace_receipt'],'probe':events['readiness_probe']['readiness_probe'],
                'profile':events['passive_native_conditions']}[stream]
            target[key]=bad
            with self.subTest(stream=stream,key=key),self.assertRaises(ValueError):overlay.require_passive_quality(events,ticks=3)
        events=quality_fixture();events['passive_native_conditions']['events'][-1]['leaf_returns'].pop()
        with self.assertRaises(ValueError):overlay.require_passive_quality(events,ticks=3)


class DiagnosticSelection(unittest.TestCase):
    def test_actual_frozen1141_parent_and_high102_original_first4_routes(self):
        value,reference=overlay._parent_lock(ROOT)
        self.assertEqual(reference,dict(path=overlay.PARENT_LOCK,sha256=overlay.PARENT_LOCK_SHA))
        self.assertTrue(value['integration_complete'])
        for method in ('cv_geometry','context_geometry'):
            args=args_for();args.method=method;chosen=overlay.selection(ROOT,args)
            self.assertEqual(chosen['original_first_four'],['episode102','episode103','episode110','episode111'])
            self.assertEqual(chosen['fixed_parent_index'],0)
        for source,expected in overlay.PASSIVE_PINS.items():overlay.parent.raw(ROOT,source,expected)

    def test_exact_high102_two_methods_only_and_parent_original_route_preserved(self):
        chosen={'original_route':{'unchanged745':True}}
        with patch.object(overlay.parent,'selection',return_value=chosen) as parent:
            self.assertIs(overlay.selection(Path('/fixture'),args_for()),chosen)
            context=args_for();context.method='context_geometry'
            self.assertIs(overlay.selection(Path('/fixture'),context),chosen)
            for field,value in (('method','physmani'),
                                ('raw_episode_id','episode101'),('task','insert_onto_rotating_peg'),
                                ('task','beat_the_rotating_buzz')):
                args=args_for();setattr(args,field,value)
                with self.assertRaises(ValueError):overlay.selection(Path('/fixture'),args)
        self.assertEqual(parent.call_count,2)

    def test_prepare_new_identity_honest_actual_worker_nested_parent_and_namespace_restore(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);args=args_for();before=overlay.parent.ARTIFACT_PREFIX;seen=[]
            inherited=dict(identity={'sha256':'parent1141','config_sha256':'config'},
                execution_protocol={'schema':overlay.parent.SCHEMA,'policy':'1131'},
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
            self.assertEqual(result['execution_protocol']['inherited_1141_execution_protocol'],inherited['execution_protocol'])
            self.assertEqual(result['identity']['source_overlay_lock_sha256'],'newlock')
            self.assertEqual(result['inherited_1141_identity']['sha256'],'parent1141')
            self.assertNotEqual(result['identity']['sha256'],'parent1141')
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
    def test_existing_profiler_is_refused_without_native_body_or_profile_override(self):
        from our_mani.multitask.train_ring_success_tap_v1125 import PassiveRingSuccessTap
        runtime=NS(_geometry=object(),atomic_json=lambda *args:None);before=(runtime._geometry,runtime.atomic_json)
        prepared=dict(args=args_for(),out=Path('/new'),source_overlay_pointer={},diagnostic_overlay_pointer={},execution_protocol={})
        @contextmanager
        def scope(*args,**kwargs):
            if len(args)>1 and type(args[1]) is dict:args[1]['restored']=True
            yield
        def imported(name):
            if name.endswith('v1126'):return NS(public_execution_trace_scope=scope)
            if name.endswith('v1134'):return NS(readiness_probe_scope=scope)
            return NS(PassiveRingSuccessTap=PassiveRingSuccessTap)
        events={}
        with patch.object(overlay,'require_native_ready'),patch.object(overlay,'verify'), \
             patch.object(overlay.parent,'runtime_scope',side_effect=scope), \
             patch.object(overlay.importlib,'import_module',side_effect=imported), \
             patch('sys.getprofile',return_value=object()),patch('sys.setprofile',side_effect=AssertionError('preserve original profiler')):
            with self.assertRaisesRegex(RuntimeError,'existing profiler'):
                with overlay.runtime_scope(Path('/fixture'),prepared,runtime,events):raise AssertionError('Native body must not run')
        self.assertEqual((runtime._geometry,runtime.atomic_json),before)
        self.assertFalse(events['restored']);self.assertFalse(events['sysprofile_installed'])

    def test_parent_annotation_called_once_before_single_write_and_new_version_false_preserved(self):
        self.scoped_case('insert_onto_rotating_peg_high_speed',raise_source=False)

    def test_native_exception_propagates_and_all_owned_scopes_restore(self):
        self.scoped_case('insert_onto_rotating_peg_high_speed',raise_source=True)

    def test_missing_observer_restoration_cannot_mask_original_native_exception(self):
        self.scoped_case('insert_onto_rotating_peg_high_speed',raise_source=True,probe_restored=False)

    def test_observer_exit_exception_cannot_mask_original_native_exception(self):
        self.scoped_case('insert_onto_rotating_peg_high_speed',raise_source=True,exit_error=True)

    def scoped_case(self,task,*,raise_source,probe_restored=True,exit_error=False):
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
        @contextmanager
        def probe_scope(module,events,**limits):
            order.append('probe_enter')
            try:yield
            finally:
                events.update(restored=probe_restored,readiness_probe=dict(restored=probe_restored,dropped_events=0,recording_errors=0))
                order.append('probe_exit')
                if exit_error:raise ValueError('observer exit failed')
        class Tap:
            def __init__(self,*args,**kwargs):self._entered=False;self.restored=False
            def __enter__(self):self._entered=True;order.append('profile_enter');return self
            def __exit__(self,*args):self.restored=True;order.append('profile_exit');return False
            def receipt(self):return {'restored':self.restored}
        trace=NS(public_execution_trace_scope=trace_scope);probe=NS(readiness_probe_scope=probe_scope)
        def imported(name):
            return trace if name.endswith('v1126') else (probe if name.endswith('v1134') else NS(PassiveRingSuccessTap=Tap))
        events={};error=RuntimeError('original Native exception')
        with patch.object(overlay,'require_native_ready'),patch.object(overlay,'verify'), \
             patch.object(overlay.parent,'runtime_scope',side_effect=parent_scope), \
             patch.object(overlay.importlib,'import_module',side_effect=imported) as modules, \
             patch('sys.setprofile',side_effect=AssertionError('no profile installation')), \
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
                self.assertEqual(writes[0]['inherited_1141_diagnostic_parent_case_version'],overlay.parent.SCHEMA+'_case')
                self.assertEqual(writes[0]['source_overlay_lock'],prepared['diagnostic_overlay_pointer'])
            self.assertEqual(modules.call_count,3)
        self.assertIs(runtime._geometry,original_geometry);self.assertIs(runtime.atomic_json,original_writer)
        self.assertIs(overlay.parent.annotated_output,original_annotation)
        self.assertIs(events['restored'],probe_restored)
        self.assertEqual(len(events['observer_finalization_errors']),int(exit_error))
        self.assertTrue(events['sysprofile_installed'])
        self.assertTrue(events['passive_native_conditions']['restored'])
        self.assertEqual(order,['parent_enter','trace_enter','probe_enter','profile_enter','profile_exit','probe_exit','trace_exit','parent_exit'])


class DiagnosticWorker(unittest.TestCase):
    def test_CPU_check_never_installs_native_probe_trace_or_constructs_runtime(self):
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
                'scripts.run_TRAIN_author_release_full16_overlay_v1141_r10':parent_worker,
                'our_mani.multitask.frozen_input_edges_v463':edges,
                'scripts.evaluate_multitask_case_r10':evaluator}
            prepared=dict(identity={'sha256':'newdiagnostic'},inherited_1141_identity={'sha256':'old1141'},
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
                finally:events.update(restored=True,readiness_probe={'restored':True,'readiness_probe':{'restored':True}},
                                      public_execution_trace={'restored':True})
            def run(current):
                overlay.parent.write_json(root/args.out/'case.json',dict(version=overlay.SCHEMA+'_case',
                    execution_protocol=prepared['execution_protocol'],source_overlay_lock=source,
                    parent_source_overlay_lock=inherited,cleanup_completed=True,process_id=os.getpid(),
                    official_task_success=False,native_terminal_failure=False,counted_clock={'ticks':200}))
            validator=Mock()
            allowed={'our_mani.multitask.post_completed_TEST_resource_guard_v1106':guard,
                'scripts.run_TRAIN_author_release_full16_overlay_v1141_r10':NS(environment_proof=lambda root:{}),
                'our_mani.multitask.frozen_input_edges_v463':edges,
                'scripts.evaluate_multitask_case_r10':NS(),
                'scripts.run_task_executor_smoke_author_suite16_v744_r10':NS(run=Mock(side_effect=run)),
                'our_mani.multitask.evaluation':NS(validate_completed_case=validator)}
            with patch.object(worker,'ROOT',root),patch.object(overlay,'selection',return_value={}), \
                 patch.object(overlay,'read_diagnostic_lock',return_value=({'integration_complete':True},source)), \
                 patch.object(overlay,'prepare',return_value=prepared),patch.object(overlay,'verify'), \
                 patch.object(worker.importlib,'import_module',side_effect=lambda name:allowed[name]), \
                 patch.object(overlay,'runtime_scope',side_effect=native_scope), \
                 patch.object(overlay,'require_passive_quality',return_value={'complete':True}):
                result=worker.run_case(args,root=root)
            self.assertIs(result['execution']['native_success'],False)
            self.assertTrue(result['execution']['native_false_preserved'])
            self.assertEqual(result['schema'],overlay.SCHEMA)
            self.assertEqual(result['source_overlay_lock'],source)
            self.assertTrue((root/args.runtime/'native/worker_sidecar.json').exists())
            validator.assert_called_once()


if __name__=='__main__':unittest.main()
