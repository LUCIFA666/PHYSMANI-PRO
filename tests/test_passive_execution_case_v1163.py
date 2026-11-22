"""Pure CPU integration contracts; no Native app/model, live case, or real lock."""
from contextlib import contextmanager,redirect_stdout,redirect_stderr
from copy import deepcopy
import importlib.util
import io,json,sys,tempfile,hashlib
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from our_mani.multitask import passive_execution_case_v1163 as m
from our_mani.multitask.passive_execution_diagnostic_v1162 import ProfileSite,_PROJECT_SITES,_EXECUTION_SITES
from tests.test_train_ring_success_tap_v1125 import fixture

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('entry1163',ROOT/m.WORKER)
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


def args(**overrides):
    value=dict(task=m.SELECTIONS[0][0],raw_episode_id=m.SELECTIONS[0][1],method=m.SELECTIONS[0][2],
        split='train',check=True,execute=False,out=m.ARTIFACT_PREFIX+'fixture',runtime=m.RUNTIME_PREFIX+'fixture',
        diagnostic_lock=m.SETUP_PREFIX+'source_overlay_lock.json',config=None,frozen_inputs=None,training_exit=None)
    value.update(overrides);return NS(**value)


def native_receipt(samples=1):
    task,sensors=fixture();tap=m.PassiveRingSuccessTap(m.SELECTIONS[0][0],'context_geometry')
    # Exercise the exact1125 callback through the real single1162-owned profile.
    def existing():return True
    one=m.OneProfile((ProfileSite.from_callable('existing',existing,kind='other'),),m.SELECTIONS[0][0],'context_geometry')
    with one:
        for _ in range(samples):task.success()
        existing()
    return one.combined_receipt()['native_callback']


def quality_fixture(pose=False):
    labels=[x[0] for x in (*_PROJECT_SITES,*_EXECUTION_SITES)]+['path.step','path.motion']
    counts={s:dict(calls=0,return_events=0) for s in labels};events=[];seq=0
    def site(name,n=1,returned=True,parent=None):
        nonlocal seq
        counts[name]=dict(calls=n,return_events=n)
        for _ in range(n):
            seq+=1;events.extend([dict(site=name,event='original_python_call',call_id=seq,parent_call_id=parent),
                dict(site=name,event='original_python_return_event',call_id=seq,parent_call_id=parent,
                     original_return=returned,return_status='non_none_original_return')])
    ticks=2 if pose else 0
    site('frontend.observe',ticks+1);site('frontend.ready')
    planrows=[];ranges=[]
    if pose:
        site('terminal.execute',returned={'path_found':False,'error':'released_modified_path_too_short'})
        for name in ('terminal.command','actuator.command','actuator.apply'):site(name,ticks)
        site('path.filter');site('path.extract');planrows=[{'original_returned':True}]
        ranges=[dict(plan_start=0,plan_end=1)]
    owner=dict(schema='r10_passive_existing_execution_diagnostic_v1162',restored=True,restoration_error=None,scope_exception_type=None,diagnostic_available=True,
        dropped_events=0,value_truncations=0,recording_errors=0,active_frame_overflows=0,
        method_call_counts=counts,events=events)
    trace=dict(schema='r10_evaluation_existing_public_execution_trace_v1126',restored=True,errors=[],scope_exception_type=None,recording_errors=0,frame_dropped=0,plan_dropped=0,
        plan_value_truncations=0,extra_env_observe_calls=0,extra_native_getter_or_condition_calls=0,
        extra_prediction_calls=0,extra_path_step_or_FK_calls=0,geometry_calls=1,frame_calls=ticks+1,
        frame_returns=ticks+1,frame_exceptions=0,frames=[{}]*(ticks+1),plan_calls=len(planrows),
        plan_returns=len(planrows),plan_exceptions=0,plans=planrows)
    pc={k:dict(calls=int(k=='frontend.ready'),returns=int(k=='frontend.ready'),exceptions=0) for k in
        ('frontend.ready','collision_port.readiness','source_policy.candidates','generator.observed_reference')}
    probe=dict(schema='r10_evaluation_existing_public_readiness_probe_v1134',restored=True,errors=[],scope_exception_type=None,recording_errors=0,dropped_events=0,
        extra_readiness_reference_forecast_or_native_calls=0,geometry_calls=1,geometry_returns=1,
        method_call_counts=pc,events=[dict(method='frontend.ready')])
    result=dict(schema=m.SCHEMA+'_observers',restored=True,factory_restored=True,writer_restored=True,parent_annotation_restored=True,
        observer_finalization_errors=[],public_execution_trace={'trace_receipt':trace,'restored':True},
        readiness_probe={'readiness_probe':probe,'restored':True},one_profile=dict(schema=m.SCHEMA+'_one_profile',owner=owner,native_callback=native_receipt(),
            native_callback_only=True,native_callback_context_entered=False,native_callback_lifecycle_fields_authoritative=False,
            native_callback_cleared=True,native_pending_at_exit=dict(success=0,condition=0),callback_errors=0,
            simultaneous_profile_scopes=1,execution_plan_ranges=ranges))
    case=dict(task=m.SELECTIONS[0][0],method='context_geometry',counted_clock={'ticks':ticks},official_task_success=False,native_terminal_failure=False,
        native_outcome_sampling={'completed_actions':[dict(label='geometry_execute',ticks=ticks)] if pose else []})
    return result,case


class Tests(unittest.TestCase):
    def setUp(self):
        self.assertIsNone(sys.getprofile());self.addCleanup(sys.setprofile,None)

    def test_default_inert_and_exact_modes(self):
        with patch.object(entry,'run_case',side_effect=AssertionError('not inert')),redirect_stdout(io.StringIO()):
            self.assertFalse(entry.main([])['enabled'])
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):entry.parser().parse_args(['--check','--execute'])

    def test_refuses_split_method_and_replacement_before_source_read(self):
        for bad in (dict(split='validation'),dict(split='test'),dict(method='physmani'),dict(raw_episode_id='episode101'),
                    dict(task='remove_cups_from_rotating_frame'),dict(check=False,execute=False)):
            with self.subTest(bad=bad),patch.object(m.parent,'raw',side_effect=AssertionError('read forbidden')):
                with self.assertRaises(ValueError):m.selection(ROOT,args(**bad))
        self.assertEqual(len(m.SELECTIONS),3)

    def test_selected_parent_has_no_rejected_factory_or_formal_worker(self):
        self.assertEqual(m.parent.SOURCE,'our_mani/multitask/train_block_preclose_full16_overlay_v1154.py')
        self.assertIn('0562c817',m.PARENT_LOCK_SHA)
        self.assertEqual(m.parent.CORE_INTEGRATION_MODULE,'our_mani.multitask.author_ring_release_integration_v1140')
        self.assertEqual(m.parent.BLOCK_MODULE,'our_mani.multitask.retained_preclose_block_integration_v1153')
        self.assertIn(m.SOURCE,m.OVERLAY_SOURCES);self.assertIn(m.WORKER,m.OVERLAY_SOURCES)
        self.assertTrue(set(m.parent.OVERLAY_SOURCES)<=set(m.OVERLAY_SOURCES))
        self.assertIn('our_mani/multitask/passive_execution_diagnostic_v1162.py',m.PASSIVE_PINS)

    def test_single_profile_fanout_preserves_native_calls_and_raw_lifecycle(self):
        task,sensors=fixture();calls=[]
        def original():calls.append(1);return 7
        site=ProfileSite.from_callable('existing',original,kind='other')
        one=m.OneProfile((site,),m.SELECTIONS[0][0],'context_geometry')
        real=sys.setprofile
        with patch.object(sys,'setprofile',wraps=real) as setter:
            with one:
                self.assertEqual(original(),7);self.assertEqual(task.success(),(False,False))
            self.assertEqual(setter.call_count,2)
        receipt=one.combined_receipt();self.assertEqual(calls,[1]);self.assertEqual([x.calls for x in sensors],[1]*4)
        self.assertTrue(receipt['owner']['restored']);self.assertFalse(receipt['native_callback']['restored'])
        self.assertFalse(receipt['native_callback_context_entered']);self.assertFalse(receipt['native_callback_lifecycle_fields_authoritative'])
        self.assertTrue(receipt['native_callback_cleared']);self.assertEqual(receipt['native_pending_at_exit'],dict(success=0,condition=0))
        self.assertEqual(receipt['native_callback']['original_condition_returns_observed'],5)

    def test_original_exception_identity_and_none_unknown(self):
        error=RuntimeError('original')
        def original():raise error
        one=m.OneProfile((ProfileSite.from_callable('bad',original,kind='other'),),m.SELECTIONS[0][0],'context_geometry')
        try:
            with one:original()
        except RuntimeError as actual:self.assertIs(actual,error)
        self.assertIsNone(sys.getprofile());self.assertTrue(one.restored)
        row=one.events[-1];self.assertEqual(row['return_status'],'none_return_or_python_exception_unwind_unknown')

    def test_foreign_profile_preserved_and_preexisting_refused(self):
        def original():return True
        def foreign(*unused):pass
        sites=(ProfileSite.from_callable('f',original,kind='other'),)
        sys.setprofile(foreign)
        try:
            with self.assertRaises(RuntimeError):m.OneProfile(sites,m.SELECTIONS[0][0],'context_geometry').__enter__()
            self.assertIs(sys.getprofile(),foreign)
        finally:sys.setprofile(None)
        one=m.OneProfile(sites,m.SELECTIONS[0][0],'context_geometry')
        with one:sys.setprofile(foreign)
        self.assertIs(sys.getprofile(),foreign);self.assertFalse(one.restored)

    def test_quality_allows_real_shortcircuit_zero_optional_sites_and_NativeFalse(self):
        e,c=quality_fixture();q=m.require_passive_quality(e,c)
        self.assertTrue(q['complete']);self.assertFalse(q['tracking_sites_reached'])
        self.assertIn('not_reached',q['site_coverage']['completion.current593']['status'])
        self.assertTrue(q['NativeFalse_fatal_preserved'])

    def test_quality_keeps_original_short_modified_path_without_rml_step(self):
        e,c=quality_fixture(pose=True);q=m.require_passive_quality(e,c)
        self.assertTrue(q['complete']);self.assertEqual(q['original_dispatch_ticks'],2)
        self.assertEqual(q['site_coverage']['path.step']['calls'],0)

    def test_original_done_context_requires_its_otherwise_optional_helper_sites(self):
        e,c=quality_fixture();probe=e['readiness_probe']['readiness_probe']
        probe['method_call_counts']['generator.observed_reference']=dict(calls=1,returns=1,exceptions=0)
        probe['events'].append(dict(method='generator.observed_reference',
            context=dict(kind='done',source_file='physmani_ring_release_v590.py')))
        owner=e['one_profile']['owner'];owner['method_call_counts']['reference.observed']=dict(calls=1,return_events=1)
        owner['events'].extend([dict(site='reference.observed',event='original_python_call',call_id=70,parent_call_id=None),
            dict(site='reference.observed',event='original_python_return_event',call_id=70,parent_call_id=None)])
        with self.assertRaisesRegex(ValueError,'done-context helper'):m.require_passive_quality(e,c)

    def test_quality_rejects_zero_stream_drop_count_spoof_and_callback_lifecycle_forgery(self):
        for mutation in ('frames','drop','counts','native','lifecycle','truncation'):
            e,c=quality_fixture()
            if mutation=='frames':e['public_execution_trace']['trace_receipt']['frames']=[]
            elif mutation=='drop':e['one_profile']['owner']['dropped_events']=1
            elif mutation=='counts':e['one_profile']['owner']['method_call_counts']['frontend.ready']['calls']=2
            elif mutation=='native':e['one_profile']['native_callback']['original_success_returns_observed']=0
            elif mutation=='lifecycle':e['one_profile']['native_callback_context_entered']=True
            else:e['one_profile']['owner']['value_truncations']=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):m.require_passive_quality(e,c)

    def test_quality_rejects_wrong_schema_in_each_actual_stream(self):
        for label in ('envelope','trace','probe','one','execution','native'):
            e,c=quality_fixture()
            streams=dict(envelope=e,trace=e['public_execution_trace']['trace_receipt'],
                probe=e['readiness_probe']['readiness_probe'],one=e['one_profile'],
                execution=e['one_profile']['owner'],native=e['one_profile']['native_callback'])
            streams[label]['schema']='wrong_stream'
            with self.subTest(label=label),self.assertRaises(ValueError):m.require_passive_quality(e,c)

    def test_post_step_target_and_plan_counter_range(self):
        trace=NS(plan_calls=0);sent=[]
        class PathCPU:
            def step(self):self._joint_position_action=[3.]*7;return True
        path=PathCPU()
        def helper(*args,**kwargs):sent.append(list(args[1]))
        class Terminal:
            def execute(self,env,action):
                trace.plan_calls+=1;done=path.step();last_target=path._joint_position_action
                tick=0;ticks=2;executed=0;error=arm_error=None
                helper(env,last_target,opening=0.,phase='path')
                tick=1;helper(env,last_target,opening=1.,phase='grip')
                return {'path_found':True}
        obj=Terminal();sites=(ProfileSite.from_callable('terminal',obj.execute,kind='terminal'),
            ProfileSite.from_callable('command',helper,kind='command'))
        with m.OneProfile(sites,m.SELECTIONS[0][0],'context_geometry',trace=trace) as one:obj.execute(NS(),[0.]*8)
        rows=[x for x in one.events if x['site']=='command' and x['event']=='original_python_call']
        self.assertEqual(sent,[[3.]*7]*2)
        self.assertEqual(rows[0]['snapshot']['forwarded_joints'],[3.]*7)
        self.assertTrue(rows[1]['snapshot']['terminal_after_step_assignment_before_dispatch']['done'])
        self.assertEqual([(x['plan_start'],x['plan_end']) for x in one.ranges],[(0,1)])

    def test_new_prepare_keeps_real_parent_namespace_lifetime_and_honest_identity(self):
        a=args();parent_ref={'path':m.PARENT_LOCK,'sha256':m.PARENT_LOCK_SHA};diag={'files':{k:'x' for k in (m.SOURCE,m.WORKER,m.TEST_SOURCE)}}
        def prepare(root,forwarded):
            self.assertEqual(m.parent.ARTIFACT_PREFIX,m.ARTIFACT_PREFIX);self.assertEqual(forwarded.overlay_lock,m.PARENT_LOCK)
            return dict(identity={'sha256':'old'},execution_protocol={'schema':m.parent.SCHEMA},source_overlay_pointer=parent_ref)
        prior=m.parent.ARTIFACT_PREFIX
        with patch.object(m,'selection',return_value={}),patch.object(m,'read_diagnostic_lock',return_value=(diag,{'path':'diag','sha256':'d'})),\
             patch.object(m,'selection_contract',return_value={'sha256':'fixture'}),\
             patch.object(m.parent,'prepare',side_effect=prepare),patch.object(m.parent,'pointer',side_effect=lambda root,name:{'path':name,'sha256':'0'*64}):
            p=m.prepare(ROOT,a)
        self.assertEqual(m.parent.ARTIFACT_PREFIX,prior);self.assertIn('inherited_1154_identity',p)
        self.assertFalse(p['execution_protocol']['formal_graph_worker_approval'])
        self.assertEqual(p['execution_protocol']['actual_worker']['path'],m.WORKER)
        self.assertNotIn('graph_witness',p)

    def test_execute_needs_actual_current_training_exit_before_any_guard_or_source_selection(self):
        with patch.object(m,'require_training_complete',side_effect=ValueError('still running')) as gate,\
             patch.object(m,'selection',side_effect=AssertionError('source selected too early')):
            with self.assertRaises(ValueError):entry.run_case(args(check=False,execute=True))
        self.assertEqual(gate.call_count,1)
        with self.assertRaises(ValueError):m.require_training_complete(ROOT,'artifacts/framework/protocol/validation_manifest.json')

    def test_cpu_entry_never_enters_observers_or_runtime_and_writes_only_fresh_phase(self):
        @contextmanager
        def scope(*unused,**kwargs):yield object()
        guard=NS(reserved_resources_after_TEST=scope,require_permit=lambda *a:{'CPU_fixture':True})
        edges=NS(declared_edge_scope=scope,evaluator_scope=scope)
        environment=NS(environment_proof=lambda root:{'CPU_fixture':True})
        modules={'our_mani.multitask.post_completed_TEST_resource_guard_v1106':guard,
            'scripts.run_TRAIN_block_preclose_full16_overlay_v1154_r10':environment,
            'our_mani.multitask.frozen_input_edges_v463':edges,'scripts.evaluate_multitask_case_r10':NS()}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=args();p=dict(identity={},selection_identity={},inherited_1154_identity=None,component_domain=NS(receipt=lambda:{}),
                model_preflight={'CPU_fixture':True},execution_protocol={})
            lock={'integration_complete':False,'unpinned_sources':[]};ref={'path':'fixture','sha256':'0'*64}
            with patch.object(entry,'ROOT',root),patch.object(entry.importlib,'import_module',side_effect=lambda n:modules[n]),\
                 patch.object(m,'selection',return_value={}),patch.object(m,'read_diagnostic_lock',return_value=(lock,ref)),\
                 patch.object(m,'prepare_check',return_value=p),patch.object(m,'verify_check'),\
                 patch.object(m.parent,'pointer',side_effect=lambda root,name:{'path':name,'sha256':'0'*64}),\
                 patch.object(m,'runtime_scope',side_effect=AssertionError('observers installed in CPU')):
                result=entry.run_case(a,root=root)
                self.assertFalse(result['execution']['native_started']);self.assertFalse(result['execution']['GPU_started'])
                self.assertFalse((root/a.out).exists());self.assertFalse((root/a.runtime/'native').exists())
                self.assertTrue((root/a.runtime/'CPU_check/worker_sidecar.json').is_file())
                with self.assertRaises(ValueError):entry.run_case(a,root=root)

    def test_static_metadata_check_hashes_weights_without_inherited_model_constructor(self):
        import our_mani.multitask.rml_component_domain_v640 as domain
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=args();files={}
            def put(name,value,binary=False):
                file=root/name;file.parent.mkdir(parents=True,exist_ok=True)
                file.write_bytes(value if binary else json.dumps(value).encode())
                checksum=hashlib.sha256(file.read_bytes()).hexdigest();files[str(file)]=checksum
                return dict(path=name,sha256=checksum)
            model=put('artifacts/model.pt',b'CPU fixture, never torch.load',True)
            config=dict(context_checkpoint=model,learned_artifacts={a.task:{'geometry':model,'cv_value':None}})
            config_ref=put('artifacts/config.json',config)
            put(m.parent.TRAIN_INPUTS,{'files':dict(files)})
            put(m.parent.MANIFEST,{'split':'train','episodes':[dict(task=a.task,raw_episode_id=a.raw_episode_id,path='/CPU_fixture/data.h5')]})
            chosen=dict(original_route={'configuration':config_ref},original_route_sha256='fixture')
            lock={'files':{k:'0'*64 for k in (m.SOURCE,m.WORKER,m.TEST_SOURCE)}}
            original=m.parent.raw
            def raw(project,name,expected=None):
                # The metadata fixture is synthetic; model-byte expected SHA remains real.
                return original(project,name,None if name in (m.parent.TRAIN_INPUTS,m.parent.MANIFEST) else expected)
            with patch.object(m,'selection',return_value=chosen),patch.object(m,'read_diagnostic_lock',return_value=(lock,{'path':'diag','sha256':'f'*64})),\
                 patch.object(m.parent,'raw',side_effect=raw),patch.object(m.parent,'prepare',side_effect=AssertionError('model constructor path')) as inherited,\
                 patch.object(m.parent,'pointer',side_effect=lambda root,name:{'path':name,'sha256':'0'*64}),\
                 patch.object(domain,'for_train',return_value=NS(receipt=lambda:{'CPU_fixture':True})):
                p=m.prepare_check(root,a);m.verify_check(root,p)
                inherited.assert_not_called()
            self.assertIsNone(p['inherited_1154_identity'])
            self.assertFalse(p['model_preflight']['actual_model_loader_validation'])
            self.assertTrue(p['model_preflight']['model_loader_preflight_deferred_to_execute'])
            self.assertFalse(p['model_preflight']['model_constructed'])

    def test_native_pending_refused_before_runtime_construct(self):
        with self.assertRaises(ValueError):m.require_native_ready(dict(diagnostic_overlay={'integration_complete':False}))

    def test_nested_scope_firstwrite_and_original_exception_survive_cleanup_error(self):
        import our_mani.multitask.train_public_execution_trace_v1126 as trace
        import our_mani.multitask.train_public_readiness_probe_v1134 as probe
        writes=[];annotations=[];original_error=RuntimeError('original body')
        def geometry(*unused):return 'unchanged result'
        runtime=NS(_geometry=geometry,atomic_json=lambda path,value:writes.append(value))
        original_writer=runtime.atomic_json
        def original_annotation(p,path,value):
            annotations.append(1);return dict(value,version='actual_parent1154_case')
        @contextmanager
        def parent_scope(root,p,current):
            def writer(path,value):return original_writer(path,m.parent.annotated_output(p,path,value))
            with patch.object(current,'atomic_json',writer):
                try:yield
                finally:raise ValueError('parent cleanup fixture')
        @contextmanager
        def public_scope(current,events,**limits):
            try:yield NS(plan_calls=0)
            finally:events.update(restored=True,trace_receipt={})
        @contextmanager
        def probe_scope(current,events,**limits):
            try:yield
            finally:events.update(restored=True,readiness_probe={})
        def original():return True
        p=dict(training_exit_name='fixture',training_completion={'path':'done'},args=args(),
            execution_protocol={},diagnostic_overlay={'native_source_hashes':{'/CPU/'+m.NATIVE_SOURCE_NAMES[0]:'0'*64}},
            diagnostic_overlay_pointer={'path':'diag'},source_overlay_pointer={'path':'parent'})
        events={}
        with patch.object(m,'require_native_ready'),patch.object(m,'verify'),\
             patch.object(m,'require_training_complete',return_value={'path':'done'}),\
             patch.object(m.parent,'runtime_scope',parent_scope),patch.object(m.parent,'annotated_output',original_annotation),\
             patch.object(trace,'public_execution_trace_scope',public_scope),patch.object(probe,'readiness_probe_scope',probe_scope),\
             patch.object(m,'frozen_source_sites',return_value=(ProfileSite.from_callable('existing',original,kind='other'),)):
            try:
                with m.runtime_scope(ROOT,p,runtime,events):
                    self.assertTrue(original());runtime.atomic_json(Path('case.json'),{'official_task_success':False})
                    raise original_error
            except RuntimeError as actual:self.assertIs(actual,original_error)
            else:self.fail('original exception was masked')
        self.assertEqual(annotations,[1]);self.assertEqual(len(writes),1)
        self.assertEqual(writes[0]['version'],m.SCHEMA+'_case');self.assertIs(writes[0]['official_task_success'],False)
        self.assertIs(runtime._geometry,geometry);self.assertIs(runtime.atomic_json,original_writer)
        self.assertIsNone(sys.getprofile());self.assertTrue(events['restored'])
        self.assertEqual(events['observer_finalization_errors'][0]['type'],'ValueError')


if __name__=='__main__':unittest.main()
