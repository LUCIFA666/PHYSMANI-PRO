"""CPU-only synthetic checks; no original preflight, lease, GPU or Native run."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from our_mani.multitask import train_author_aligned_full16_overlay_v1122 as overlay
from scripts import run_TRAIN_author_aligned_full16_overlay_v1122_r10 as worker


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def args_for(task=None, method='cv_geometry'):
    return NS(task=task or overlay.TASKS[0], method=method, raw_episode_id='0', config=None,
        frozen_inputs=None, out=overlay.ARTIFACT_PREFIX+'case', runtime=overlay.RUNTIME_PREFIX+'case',
        overlay_lock=overlay.SETUP_PREFIX+'lock.json', check=True)


def route_fixture():
    rows = []; values = {}
    for task in overlay.TASKS:
        for method in overlay.METHODS:
            name = 'configs/'+task+'_'+method+'.json'
            value = {} if method == 'physmani' else dict(context_checkpoint=overlay.CHECKPOINT)
            values[name] = encoded(value)
            rows.append(dict(task=task, method=method, GPU_required=method == 'physmani',
                configuration=dict(path=name, sha256=hashlib.sha256(values[name]).hexdigest()),
                frozen_inputs=dict(path=overlay.TRAIN_INPUTS, sha256=overlay.TRAIN_INPUTS_SHA),
                **{field:dict(path=name, sha256=overlay.FROZEN_PINS[name]) for field,name in (
                    ('worker',overlay.LEGACY_WORKER), ('consumer',overlay.LEGACY_RUNTIME),
                    ('source_preflight','our_mani/multitask/rml_case_preflight_v742.py'))}))
    values[overlay.ROUTES] = encoded(dict(routes=rows))
    values[overlay.MANIFEST] = encoded(dict(split='train', episodes=[dict(task=t, raw_episode_id=str(i))
        for t in overlay.TASKS for i in range(5)]))
    values[overlay.TRAIN_INPUTS] = encoded(dict(files={}))
    return values, rows


class SelectionTests(unittest.TestCase):
    def test_all192_routes_use_exact_method_config_and_first_four(self):
        values, rows = route_fixture()
        with patch.object(overlay, 'raw', side_effect=lambda root,name,expected=None:values[name]):
            count = 0
            for route in rows:
                args = args_for(route['task'], route['method'])
                for index in range(4):
                    args.raw_episode_id = str(index)
                    selected = overlay.selection(Path('/fixture'), args)
                    self.assertEqual(selected['fixed_parent_index'], index)
                    self.assertEqual(selected['original_route'], route)
                    self.assertEqual(selected['original_first_four'], ['0','1','2','3'])
                    count += 1
            self.assertEqual(count, 192)

    def test_wrong_method_config_parent_and_duplicate_route_rejected(self):
        values, rows = route_fixture(); args = args_for(method='physmani')
        with patch.object(overlay, 'raw', side_effect=lambda root,name,expected=None:values[name]):
            args.config = rows[0]['configuration']['path']
            with self.assertRaisesRegex(ValueError, 'argv differs'):overlay.selection(Path('/fixture'),args)
            args.config = None; args.raw_episode_id = '4'
            with self.assertRaisesRegex(ValueError, 'first four'):overlay.selection(Path('/fixture'),args)
            args.raw_episode_id = '0'
            values[overlay.ROUTES] = encoded(dict(routes=rows[:-1]+[rows[0]]))
            with self.assertRaisesRegex(ValueError, 'unique original745'):overlay.selection(Path('/fixture'),args)

    def test_changed_gpu_or_original_dependency_rejected(self):
        values, rows = route_fixture(); args = args_for()
        with patch.object(overlay, 'raw', side_effect=lambda root,name,expected=None:values[name]):
            rows[0]['GPU_required'] = True; values[overlay.ROUTES] = encoded(dict(routes=rows))
            with self.assertRaisesRegex(ValueError, 'GPU allocation'):overlay.selection(Path('/fixture'),args)
            rows[0]['GPU_required'] = False; rows[0]['worker']['sha256'] = 'wrong'
            values[overlay.ROUTES] = encoded(dict(routes=rows))
            with self.assertRaisesRegex(ValueError, 'dependency changed'):overlay.selection(Path('/fixture'),args)

    def test_prepare_checks_original_worker_but_binds_new_identity_and_route(self):
        from our_mani.multitask import rml_case_preflight_v742 as substrate
        values, rows = route_fixture(); args = args_for(method='physmani')
        chosen = dict(original_route=rows[2], original_route_sha256=overlay.digest(rows[2]))
        base = dict(identity=dict(sha256='old',config_sha256='config'), config={})
        source = dict(files={s:'sha' for s in (overlay.SOURCE,overlay.WORKER,overlay.TEST_SOURCE)}, integration_complete=False)
        seen = []
        def prepare(root, forwarded, actual_worker):
            seen.append((forwarded, actual_worker));return deepcopy(base)
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            with patch.object(overlay,'selection',return_value=chosen),\
                 patch.object(overlay,'read_overlay_lock',return_value=(source,{'path':'lock','sha256':'overlay'})),\
                 patch.object(overlay,'pointer',side_effect=lambda root,name:dict(path=name,sha256='sha')),\
                 patch.object(substrate,'prepare',side_effect=prepare),\
                 patch.object(substrate,'outcome_protocol_binding',return_value={'same741':True}):
                actual=overlay.prepare(root,args)
        forwarded, checked=seen[0]
        self.assertEqual(checked,root/overlay.LEGACY_WORKER)
        self.assertEqual(forwarded.config,rows[2]['configuration']['path'])
        self.assertEqual((forwarded.split,forwarded.development_demo,forwarded.suite),('train',True,None))
        self.assertEqual(actual['source_substrate_identity']['sha256'],'old')
        self.assertNotEqual(actual['identity']['sha256'],'old')
        self.assertEqual(actual['execution_protocol']['execution_branch'],'unchanged744_physmani')
        self.assertEqual(actual['execution_protocol']['changed_factor'],'no_core_policy_variant_applied')
        variant=actual['execution_protocol']['core_factory']['matched_policy_variant']
        self.assertEqual(variant['changed_factor'],'first_acquire_close_uses_original_policy_qualification')
        self.assertEqual(variant['release_and_other_opening_dispatch'],'unchanged_parent_author_requested_opening')
        self.assertFalse(actual['execution_protocol']['formal_VAL_TEST_qualification'])


class LockTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        for name in ('source.py','worker.py','test.py'):(self.root/name).write_text(name)
        self.sources=('source.py','worker.py','test.py','pending1120.py')
        self.constants=patch.object(overlay,'OVERLAY_SOURCES',self.sources)
        self.constants.start();self.addCleanup(self.constants.stop)
        self.pins=patch.object(overlay,'_substrate_pins');self.pins.start();self.addCleanup(self.pins.stop)

    def test_default_pending_has_honest_missing_sources_and_no_native_or_overwrite(self):
        name=overlay.SETUP_PREFIX+'pending.json';overlay.write_overlay_lock(self.root,name)
        lock,_=overlay.read_overlay_lock(self.root,name)
        self.assertEqual(lock['unpinned_sources'],['pending1120.py'])
        self.assertNotIn('pending1120.py',lock['files'])
        with self.assertRaisesRegex(ValueError,'integration pending'):
            overlay.require_native_ready(dict(source_overlay=lock))
        with self.assertRaises(FileExistsError):overlay.write_overlay_lock(self.root,name)

    def test_activation_requires_all_sources_and_exact_complete_cpu_evidence(self):
        with self.assertRaisesRegex(ValueError,'all integration sources'):
            overlay.write_overlay_lock(self.root,overlay.SETUP_PREFIX+'bad.json',integration_complete=True)
        (self.root/'pending1120.py').write_text('ready')
        files={name:overlay.pointer(self.root,name)['sha256'] for name in self.sources}
        name=overlay.SETUP_PREFIX+'CPU.json'
        evidence=dict(schema=overlay.CPU_EVIDENCE_SCHEMA,complete=True,tests_exit_code=0,
            native_started=False,GPU_started=False,source_hashes=files)
        overlay.write_json(self.root/name,evidence);ref=overlay.pointer(self.root,name)
        overlay.write_overlay_lock(self.root,overlay.SETUP_PREFIX+'complete.json',integration_complete=True,integration_cpu_evidence=ref)
        value,_=overlay.read_overlay_lock(self.root,overlay.SETUP_PREFIX+'complete.json')
        overlay.require_native_ready(dict(source_overlay=value))
        (self.root/'worker.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'input SHA changed'):
            overlay.read_overlay_lock(self.root,overlay.SETUP_PREFIX+'complete.json')

    def test_boolean_exit_code_or_wrong_source_map_cannot_activate(self):
        (self.root/'pending1120.py').write_text('ready')
        files={name:overlay.pointer(self.root,name)['sha256'] for name in self.sources}
        for index,(exit_code,source_hashes) in enumerate(((False,files),(0,{}))):
            name=overlay.SETUP_PREFIX+str(index)+'CPU.json'
            overlay.write_json(self.root/name,dict(schema=overlay.CPU_EVIDENCE_SCHEMA,complete=True,
                tests_exit_code=exit_code,native_started=False,GPU_started=False,source_hashes=source_hashes))
            with self.assertRaisesRegex(ValueError,'CPU integration evidence'):
                overlay.write_overlay_lock(self.root,overlay.SETUP_PREFIX+str(index)+'lock.json',
                    integration_complete=True,integration_cpu_evidence=overlay.pointer(self.root,name))

    def test_core_factory_choice_is_explicit_and_change_invalidates_existing_lock(self):
        name=overlay.SETUP_PREFIX+'selected_core.json';overlay.write_overlay_lock(self.root,name)
        value,_=overlay.read_overlay_lock(self.root,name)
        self.assertEqual(value['core_factory']['module'],overlay.CORE_INTEGRATION_MODULE)
        self.assertEqual(value['core_factory']['source'],overlay.CORE_INTEGRATION_SOURCE)
        self.assertEqual(value['core_factory']['module'],'our_mani.multitask.retained_preclose_policy_integration_v1121')
        self.assertEqual(value['core_factory']['matched_policy_variant']['applies_to'],'core14_geometry_only')
        with patch.object(overlay,'CORE_INTEGRATION_MODULE','our_mani.multitask.future_matched_variant'):
            with self.assertRaisesRegex(ValueError,'core factory module/source binding changed'):
                overlay.read_overlay_lock(self.root,name)


class RuntimeTests(unittest.TestCase):
    def prepared(self,task=None,method='cv_geometry'):
        return dict(source_overlay=dict(integration_complete=True),args=args_for(task,method),
            config=dict(baseline_repo='/native_source'),component_domain=NS(receipt=lambda:{'purpose':'train_qualification'}),
            model_preflight=dict(actual_shared16_block_adapter={'original97':True}),
            source_overlay_pointer={'path':'lock','sha256':'new'},execution_protocol={'schema':overlay.SCHEMA},out=Path('/new_case'))

    def runtime(self,writes):
        return NS(_geometry=object(),_baseline=object(),atomic_json=lambda path,value:writes.append((path,value)))

    def test_core_scope_lasts_through_receipt_and_restores_after_error(self):
        from our_mani.multitask import rml_component_extensions_v708 as actor_factory
        events=[];writes=[];runtime=self.runtime(writes);original=runtime._geometry;baseline=runtime._baseline
        prepared=self.prepared();actor=object();frontend=object();policy=object();executor=object()
        @contextmanager
        def integrated(*args):
            events.append('enter')
            try:yield frontend,policy,executor
            finally:events.append('exit')
        with patch.object(overlay,'verify'),patch.object(actor_factory,'create_actor',return_value=(actor,{})),\
             patch.object(overlay.importlib,'import_module',return_value=NS(integrated_components=integrated)) as imported:
            with self.assertRaisesRegex(RuntimeError,'source error'):
                with overlay.runtime_scope(Path('/fixture'),prepared,runtime):
                    triple=runtime._geometry(prepared,object(),[])
                    self.assertEqual(triple[2:],(frontend,policy,executor));self.assertEqual(events,['enter'])
                    runtime.atomic_json(Path('/new_case/case.json'),dict(version='old744',official_task_success=False,
                        full16_frozen_execution={'old_policy':'source_only'}))
                    self.assertEqual(events,['enter']);raise RuntimeError('source error')
        self.assertEqual(events,['enter','exit']);self.assertIs(runtime._geometry,original);self.assertIs(runtime._baseline,baseline)
        imported.assert_called_once_with(overlay.CORE_INTEGRATION_MODULE)
        value=writes[0][1];self.assertEqual(value['version'],overlay.SCHEMA+'_case')
        self.assertIs(value['official_task_success'],False);self.assertNotIn('full16_frozen_execution',value)
        self.assertFalse(value['new_protocol_formal_VAL_TEST_qualification'])

    def test_blocks_use1120_not_actor_and_preserve_original_block_proof(self):
        from our_mani.multitask import rml_component_extensions_v708 as actor_factory
        events=[];runtime=self.runtime([]);prepared=self.prepared(overlay.BLOCKS[0]);triple=(object(),object(),object())
        @contextmanager
        def integrated(*args):
            events.append(args)
            try:yield triple
            finally:events.append('exit')
        block=NS(integrated_block_components=integrated,binding=Mock(return_value={'new1120':True}))
        with patch.object(overlay,'verify'),patch.object(overlay.importlib,'import_module',return_value=block) as imported,\
             patch.object(actor_factory,'create_actor') as create:
            with overlay.runtime_scope(Path('/fixture'),prepared,runtime):
                result=runtime._geometry(prepared,'existing_env',[])
                self.assertIsNone(result[0]);self.assertEqual(result[2:],triple)
                self.assertEqual(result[1]['actual_shared16_block_adapter'],{'original97':True})
                self.assertEqual(result[1]['new_block_integration'],{'new1120':True})
                self.assertEqual(events[0][-1],'existing_env');self.assertEqual(len(events),1)
        self.assertEqual(events[-1],'exit');create.assert_not_called();imported.assert_called_once_with(overlay.BLOCK_MODULE)

    def test_physmani_leaves_both_original_factories_untouched_and_imports_no_components(self):
        runtime=self.runtime([]);original_geometry=runtime._geometry;original_baseline=runtime._baseline
        with patch.object(overlay,'verify'),patch.object(overlay.importlib,'import_module') as imported:
            with overlay.runtime_scope(Path('/fixture'),self.prepared(method='physmani'),runtime):
                self.assertIs(runtime._geometry,original_geometry);self.assertIs(runtime._baseline,original_baseline)
            imported.assert_not_called()


class WorkerTests(unittest.TestCase):
    def test_pending_native_is_rejected_before_guard_environment_or_consumer_import(self):
        args=args_for();args.check=False
        with patch.object(overlay,'selection',return_value={}),\
             patch.object(overlay,'read_overlay_lock',return_value=({'integration_complete':False},{})),\
             patch.object(worker.importlib,'import_module') as imported:
            with self.assertRaisesRegex(ValueError,'integration pending'):worker.run_case(args)
            imported.assert_not_called()

    def test_real820_namespace_field_and_current_process_are_required(self):
        proof=dict(complete=True,observer_pid=os.getpid(),wrapper=worker.ENVIRONMENT['wrapper'],
            manifest=worker.ENVIRONMENT['manifest'],namespace_qualification=worker.ENVIRONMENT['qualification'])
        with patch.object(overlay,'raw'),patch.object(worker.importlib,'import_module',
            return_value=NS(capture_namespace_evidence=Mock(return_value=proof))):
            self.assertEqual(worker.environment_proof(worker.ROOT),proof)
            proof['observer_pid']=-1
            with self.assertRaisesRegex(ValueError,'same-process'):worker.environment_proof(worker.ROOT)

    def test_cpu_check_isolated_output_no_native_consumer_or_gpu(self):
        @contextmanager
        def reservation(*args,**kwargs):yield object()
        guard=NS(reserved_resources_after_TEST=reservation,require_permit=Mock(return_value={'synthetic':True}))
        modules={'our_mani.multitask.post_completed_TEST_resource_guard_v1106':guard,
            'our_mani.multitask.frozen_input_edges_v463':NS(declared_edge_scope=lambda:nullcontext(),evaluator_scope=lambda e:nullcontext()),
            'scripts.evaluate_multitask_case_r10':object()}
        prepared=dict(identity={'new':'identity'},source_substrate_identity={'old':'identity'},
            source_overlay=dict(integration_complete=False,unpinned_sources=['pending1120.py']),
            component_domain=NS(receipt=lambda:{'split':'train'}),model_preflight={'GPU_started':False},execution_protocol={'schema':overlay.SCHEMA})
        with TemporaryDirectory() as temporary:
            root=Path(temporary);args=args_for()
            with patch.object(worker,'ROOT',root),patch.object(overlay,'selection',return_value={}),\
                 patch.object(overlay,'read_overlay_lock',return_value=(prepared['source_overlay'],{'path':'lock','sha256':'new'})),\
                 patch.object(overlay,'prepare',return_value=prepared),patch.object(overlay,'verify'),\
                 patch.object(worker,'environment_proof',return_value={'synthetic820':True}),\
                 patch.object(worker.importlib,'import_module',side_effect=lambda name:modules[name]):
                result=worker.run_case(args,root=root)
            self.assertFalse(result['execution']['native_started']);self.assertFalse(result['execution']['GPU_started'])
            self.assertFalse((root/args.out).exists())
            self.assertTrue((root/args.runtime/'CPU_check/worker_sidecar.json').exists())


if __name__ == '__main__':unittest.main()
