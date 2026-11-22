"""CPU-only synthetic checks; no original preflight, lease, GPU or Native run."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import hashlib
import json
import os
import io
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from our_mani.multitask import train_native_release_full16_overlay_v1165 as overlay
from scripts import run_TRAIN_native_release_full16_overlay_v1165_r10 as worker


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def args_for(task=None, method='cv_geometry'):
    return NS(task=task or overlay.TASKS[0], method=method, raw_episode_id='0', config=None,
        frozen_inputs=None, out=overlay.ARTIFACT_PREFIX+'case', runtime=overlay.RUNTIME_PREFIX+'case',
        overlay_lock=overlay.SETUP_PREFIX+'lock.json', check=True,execute=False,split='train',prior_stop=None)


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
    def test_actual_release_rules_and_only_eight_routes_change_factor(self):
        root=Path(__file__).resolve().parents[1]
        from our_mani.multitask import author_native_release_core_v1164 as core
        changed=[]
        for task in overlay.TASKS:
            for method in overlay.METHODS:
                factor=overlay.applied_phase_factor(task,method)
                if factor!=overlay.comparison_parent.applied_phase_factor(task,method):changed.append((task,method))
        self.assertEqual(set(changed),{(t,m) for t in overlay.RELEASE_TASKS for m in ('cv_geometry','context_geometry')})
        for task in overlay.RELEASE_TASKS:
            config=json.loads((root/('artifacts/framework/multitask_r10/post_training_qualification_v680_setup/'+task+'_ours_configuration.json')).read_text())
            for method in ('cv_geometry','context_geometry'):
                actual=overlay.phase_component_binding(root,config,task,method)
                self.assertEqual(actual['integration']['schema'],core.SCHEMA)
                self.assertEqual(actual['task_rule']['changed_factor'],core.CHANGED_FACTOR)
                self.assertEqual(actual['task_rule']['inherited_author_release_binding']['release_distance_m' if task in overlay.BASKETS else 'author_terminal_release_distance_m'],.03)
                self.assertEqual(len(overlay.digest(actual)),64)

    def test_final1153_complete_source_graph_and_actual_block_bindings(self):
        from our_mani.multitask import retained_preclose_block_integration_v1153 as block
        root=Path(__file__).resolve().parents[1]
        self.assertEqual(overlay.pointer(root,block.SOURCE)['sha256'],
            'ef646686984cf2de785e00ae39ec460bb68ac2fae9b3b4c1b9ca8e5a303a1b92')
        self.assertTrue(set(block.SOURCES)<=set(overlay.OVERLAY_SOURCES))
        roster=json.loads(overlay.raw(root,overlay.MANIFEST,overlay.MANIFEST_SHA))
        for task in overlay.BLOCKS:
            first=next(row['raw_episode_id'] for row in roster['episodes'] if row['task']==task)
            for method in ('cv_geometry','context_geometry'):
                args=args_for(task,method);args.raw_episode_id=first
                route=overlay.selection(root,args)['original_route'];reference=route['configuration']
                config=json.loads(overlay.raw(root,reference['path'],reference['sha256']))
                value=overlay.phase_component_binding(root,config,task,method)
                self.assertEqual(set(value),{'block_integration'})
                binding=value['block_integration'];self.assertEqual(binding['schema'],block.SCHEMA)
                self.assertEqual(binding['changed_factor'],'retained_original_block_first_preclose_witness')
                self.assertEqual(binding['parent']['schema'],'r10_author_aligned_block_integration_v1120')
                self.assertFalse(binding['public_threshold_is_native_grasp_necessity_or_sufficiency_claim'])
                self.assertEqual(len(overlay.digest(value)),64)

    def test_actual_full48_source_bindings_are_json_bound_without_native_claim(self):
        root=Path(__file__).resolve().parents[1]
        routes=json.loads(overlay.raw(root,overlay.ROUTES,overlay.ROUTES_SHA))['routes']
        self.assertEqual(len(routes),48)
        seen=set()
        for route in routes:
            task,method=route['task'],route['method'];seen.add((task,method))
            config=json.loads(overlay.raw(root,route['configuration']['path'],route['configuration']['sha256']))
            value=overlay.phase_component_binding(root,config,task,method)
            if method=='physmani':
                self.assertIsNone(value)
                continue
            component=value['block_integration' if task in overlay.BLOCKS else 'integration']
            self.assertFalse(component['formal_VAL_TEST_qualification'])
            self.assertTrue(set(component['sources_sha256'])<=set(overlay.OVERLAY_SOURCES))
            if task in overlay.RELEASE_TASKS:
                self.assertEqual(value['task_rule']['controlled_phase'],'deposit' if task in overlay.BASKETS else 'release')
                self.assertFalse(value['task_rule']['Native_capture_or_success_guaranteed'])
                if task in overlay.BUZZ_TASKS:
                    self.assertEqual(value['unchanged_Buzz_approach_rule']['task'],task)
            self.assertEqual(len(overlay.digest(value)),64)
        self.assertEqual(seen,{(t,m) for t in overlay.TASKS for m in overlay.METHODS})

    def test_actual_original_routes_and_all192_first4_match_frozen1154_parent(self):
        root=Path(__file__).resolve().parents[1]
        manifest=json.loads(overlay.raw(root,overlay.MANIFEST,overlay.MANIFEST_SHA))
        count=0
        for task in overlay.TASKS:
            first=[row['raw_episode_id'] for row in manifest['episodes'] if row['task']==task][:4]
            for method in overlay.METHODS:
                for raw in first:
                    args=args_for(task,method);args.raw_episode_id=raw
                    self.assertEqual(overlay.selection(root,args),overlay.comparison_parent.selection(root,args));count+=1
        self.assertEqual(count,192)

    def test_actual_problem_family_rules_and_1140_source_graph_are_json_bound(self):
        from our_mani.multitask import author_native_release_core_v1164 as components
        root=Path(__file__).resolve().parents[1]
        self.assertTrue(set(components.SOURCES)<=set(overlay.OVERLAY_SOURCES))
        self.assertTrue(all((root/name).is_file() for name in components.SOURCES))
        for task in ('insert_onto_rotating_peg','insert_onto_rotating_peg_high_speed',
                     'beat_the_rotating_buzz','beat_the_rotating_buzz_high_speed'):
            config=json.loads((root/('artifacts/framework/multitask_r10/'
                'post_training_qualification_v680_setup/'+task+'_ours_configuration.json')).read_text())
            value=overlay.phase_component_binding(root,config,task,'cv_geometry')
            self.assertEqual(value['integration']['schema'],
                'r10_retained_author_native_release_core_integration_v1164')
            self.assertEqual(value['task_rule']['task'],task)
            if task.startswith('insert_onto_rotating_peg'):
                self.assertFalse(value['task_rule']['ring_seating_v1129_installed'])
                self.assertIn('inherited_author_release_binding',value['task_rule'])
            self.assertEqual(len(overlay.digest(value)),64)

    def test_new_source_graph_keeps_frozen_parent_and_declares_actual1140_candidate(self):
        self.assertEqual(overlay.CORE_INTEGRATION_MODULE,
                         'our_mani.multitask.author_native_release_core_v1164')
        for name in ('our_mani/multitask/committed_public_role_integration_v1137.py',
                     'tests/test_committed_public_role_integration_v1137.py',
                     'our_mani/multitask/author_ring_release_integration_v1140.py',
                     'tests/test_author_ring_release_integration_v1140.py',
                     'our_mani/multitask/public_task_phase_integration_v1131.py',
                     'tests/test_public_task_phase_integration_v1131.py',overlay.PARENT_OVERLAY_LOCK):
            self.assertIn(name,overlay.OVERLAY_SOURCES)
        self.assertNotIn('our_mani/multitask/public_ring_seating_policy_v1129.py',overlay.OVERLAY_SOURCES)
        self.assertEqual(overlay.PARENT_OVERLAY_LOCK_SHA,
            '0562c8170bac17f0178a2a43381777484c2dacd478514abaef0371f9d8f3f9a9')
        self.assertEqual(overlay.BLOCK_MODULE,'our_mani.multitask.retained_preclose_block_integration_v1153')
        self.assertTrue(set(overlay.comparison_parent.OVERLAY_SOURCES)<=set(overlay.OVERLAY_SOURCES))
        self.assertIn(overlay.BLOCK_SOURCE,overlay.OVERLAY_SOURCES)
        self.assertIn(overlay.BLOCK_TEST,overlay.OVERLAY_SOURCES)
        self.assertNotIn('blocks',overlay.CORE_POLICY_VARIANT)

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
        self.assertEqual(variant['changed_factor'],'retained_original_author_first_release_qualification')
        self.assertEqual(variant['other_core_tasks'],'exact_parent1140_delegate')
        self.assertFalse(actual['execution_protocol']['formal_VAL_TEST_qualification'])
        self.assertFalse(actual['execution_protocol']['actual_worker_is_comparison_parent1154'])
        self.assertFalse(actual['execution_protocol']['core_changed_relative_to1154'])
        self.assertFalse(actual['execution_protocol']['baseline_changed_relative_to1154'])
        self.assertEqual(actual['execution_protocol']['comparison_parent_source_overlay_lock'],
            dict(path=overlay.PARENT_OVERLAY_LOCK,sha256=overlay.PARENT_OVERLAY_LOCK_SHA))

    def test_block_prepare_declares1153_only_and_new_protocol_before_execution(self):
        from our_mani.multitask import rml_case_preflight_v742 as substrate
        values,rows=route_fixture();task=overlay.BLOCKS[0];args=args_for(task,'context_geometry')
        route=next(row for row in rows if (row['task'],row['method'])==(task,args.method))
        chosen=dict(original_route=route,original_route_sha256=overlay.digest(route))
        base=dict(identity=dict(sha256='old',config_sha256='config'),config={})
        source=dict(files={s:'sha' for s in (overlay.SOURCE,overlay.WORKER,overlay.TEST_SOURCE)},integration_complete=False)
        with TemporaryDirectory() as temporary,patch.object(overlay,'selection',return_value=chosen), \
             patch.object(overlay,'read_overlay_lock',return_value=(source,{'path':args.overlay_lock,'sha256':'new'})), \
             patch.object(substrate,'prepare',return_value=base), \
             patch.object(substrate,'outcome_protocol_binding',return_value={'original741':'CPU_fixture'}), \
             patch.object(overlay,'phase_component_binding',return_value={'block_integration':{'schema':'new1153'}}), \
             patch.object(overlay,'pointer',side_effect=lambda root,name:dict(path=name,sha256='sha')):
            prepared=overlay.prepare(Path(temporary),args)
        protocol=prepared['execution_protocol']
        self.assertEqual(protocol['execution_branch'],'block1153')
        self.assertEqual(protocol['changed_factor'],'retained_original_block_first_preclose_witness')
        self.assertEqual(protocol['block_factory']['module'],overlay.BLOCK_MODULE)
        self.assertEqual(protocol['public_phase_component_binding'],{'block_integration':{'schema':'new1153'}})
        self.assertFalse(protocol['core_changed_relative_to1154']);self.assertFalse(protocol['baseline_changed_relative_to1154'])

    def test_only_block_factor_changes_relative_to1154(self):
        for task in overlay.TASKS:
            self.assertEqual(overlay.applied_phase_factor(task,'physmani'),'no_core_policy_variant_applied')
        for task in overlay.BLOCKS:
            self.assertEqual(overlay.applied_phase_factor(task,'cv_geometry'),'retained_original_block_first_preclose_witness')
        for task in overlay.TASKS:
            if task not in overlay.BLOCKS and task not in overlay.RELEASE_TASKS:
                for method in overlay.METHODS:
                    self.assertEqual(overlay.applied_phase_factor(task,method),overlay.comparison_parent.applied_phase_factor(task,method))
        for task in ('insert_onto_rotating_peg','insert_onto_rotating_peg_high_speed'):
            self.assertEqual(overlay.applied_phase_factor(task,'context_geometry'),'author_wp4_release_with_confirmed_semantic_commitment')
        for task in ('beat_the_rotating_buzz','beat_the_rotating_buzz_high_speed'):
            self.assertEqual(overlay.applied_phase_factor(task,'cv_geometry'),'retained_original_author_first_release_qualification')
        self.assertEqual(overlay.applied_phase_factor('remove_cups_from_rotating_frame','cv_geometry'),'unchanged1121_public_policy')

    def test_phase_binding_is_cpu_only_block1153_is_distinct_and_baseline_imports_nothing(self):
        component=NS(binding=Mock(return_value={'source_graph':True}),
                     ring_rule_binding=Mock(return_value={'original_author_rule':True}))
        with patch.object(overlay.importlib,'import_module',return_value=component) as imported:
            actual=overlay.phase_component_binding(Path('/fixture'),{'baseline_repo':'frozen'},'insert_onto_rotating_peg','cv_geometry')
            self.assertEqual(actual,{'integration':{'source_graph':True},'task_rule':{'original_author_rule':True}})
            imported.assert_called_once_with(overlay.CORE_INTEGRATION_MODULE)
            component.binding.assert_called_once_with(Path('/fixture'),'frozen')
            component.ring_rule_binding.assert_called_once_with(Path('/fixture'),{'baseline_repo':'frozen'},'insert_onto_rotating_peg')
        with patch.object(overlay.importlib,'import_module',return_value=component) as imported:
            value=overlay.phase_component_binding(Path('/fixture'),{'baseline_repo':'frozen'},overlay.BLOCKS[0],'cv_geometry')
            self.assertEqual(value,{'block_integration':{'source_graph':True}})
            imported.assert_called_once_with(overlay.BLOCK_MODULE)
        with patch.object(overlay.importlib,'import_module') as imported:
            self.assertIsNone(overlay.phase_component_binding(None,{},'insert_onto_rotating_peg','physmani'))
            imported.assert_not_called()


class LockTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        for name in ('source.py','worker.py','test.py'):(self.root/name).write_text(name)
        self.sources=('source.py','worker.py','test.py','pending1140.py')
        self.constants=patch.object(overlay,'OVERLAY_SOURCES',self.sources)
        self.constants.start();self.addCleanup(self.constants.stop)
        self.pins=patch.object(overlay,'_substrate_pins');self.pins.start();self.addCleanup(self.pins.stop)

    def test_default_pending_has_honest_missing_sources_and_no_native_or_overwrite(self):
        name=overlay.SETUP_PREFIX+'pending.json';overlay.write_overlay_lock(self.root,name)
        lock,_=overlay.read_overlay_lock(self.root,name)
        self.assertEqual(lock['unpinned_sources'],['pending1140.py'])
        self.assertNotIn('pending1140.py',lock['files'])
        with self.assertRaisesRegex(ValueError,'integration pending'):
            overlay.require_native_ready(dict(source_overlay=lock))
        with self.assertRaises(FileExistsError):overlay.write_overlay_lock(self.root,name)

    def test_activation_requires_all_sources_and_exact_complete_cpu_evidence(self):
        with self.assertRaisesRegex(ValueError,'all integration sources'):
            overlay.write_overlay_lock(self.root,overlay.SETUP_PREFIX+'bad.json',integration_complete=True)
        (self.root/'pending1140.py').write_text('ready')
        files={name:overlay.pointer(self.root,name)['sha256'] for name in self.sources}
        name=overlay.SETUP_PREFIX+'CPU.json'
        evidence=dict(schema=overlay.CPU_EVIDENCE_SCHEMA,complete=True,tests_exit_code=0,
            native_started=False,GPU_started=False,source_hashes=files,
            parent_source_overlay_lock=dict(path=overlay.PARENT_OVERLAY_LOCK,sha256=overlay.PARENT_OVERLAY_LOCK_SHA))
        overlay.write_json(self.root/name,evidence);ref=overlay.pointer(self.root,name)
        overlay.write_overlay_lock(self.root,overlay.SETUP_PREFIX+'complete.json',integration_complete=True,integration_cpu_evidence=ref)
        value,_=overlay.read_overlay_lock(self.root,overlay.SETUP_PREFIX+'complete.json')
        overlay.require_native_ready(dict(source_overlay=value,prior_stop={'path':'fixture'}))
        (self.root/'worker.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'input SHA changed'):
            overlay.read_overlay_lock(self.root,overlay.SETUP_PREFIX+'complete.json')

    def test_boolean_exit_code_or_wrong_source_map_cannot_activate(self):
        (self.root/'pending1140.py').write_text('ready')
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
        self.assertEqual(value['core_factory']['module'],'our_mani.multitask.author_native_release_core_v1164')
        self.assertEqual(value['core_factory']['matched_policy_variant']['applies_to'],'basketball_deposit_and_Buzz_release_geometry_closed_to_requested_open_only')
        with patch.object(overlay,'CORE_INTEGRATION_MODULE','our_mani.multitask.future_matched_variant'):
            with self.assertRaisesRegex(ValueError,'core factory module/source binding changed'):
                overlay.read_overlay_lock(self.root,name)

    def test_parent_source_lock_is_required_in_cpu_evidence_and_locked_protocol(self):
        (self.root/'pending1140.py').write_text('ready')
        files={name:overlay.pointer(self.root,name)['sha256'] for name in self.sources}
        evidence=dict(schema=overlay.CPU_EVIDENCE_SCHEMA,complete=True,tests_exit_code=0,
            native_started=False,GPU_started=False,source_hashes=files)
        name=overlay.SETUP_PREFIX+'missing_parent_CPU.json'
        overlay.write_json(self.root/name,evidence)
        with self.assertRaisesRegex(ValueError,'CPU integration evidence'):
            overlay.write_overlay_lock(self.root,overlay.SETUP_PREFIX+'bad_parent.json',
                integration_complete=True,integration_cpu_evidence=overlay.pointer(self.root,name))
        locked=overlay.SETUP_PREFIX+'pending_parent.json'
        overlay.write_overlay_lock(self.root,locked)
        value=json.loads((self.root/locked).read_text())
        self.assertEqual(value['parent_source_overlay_lock'],
            dict(path=overlay.PARENT_OVERLAY_LOCK,sha256=overlay.PARENT_OVERLAY_LOCK_SHA))
        value['parent_source_overlay_lock']['sha256']='changed'
        (self.root/locked).write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'comparison-parent'):
            overlay.read_overlay_lock(self.root,locked)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        scoped=patch.object(overlay,'require_prior_stop',return_value={'path':'controlled'})
        scoped.start();self.addCleanup(scoped.stop)

    def prepared(self,task=None,method='cv_geometry'):
        return dict(source_overlay=dict(integration_complete=True),args=args_for(task,method),
            config=dict(baseline_repo='/native_source'),component_domain=NS(receipt=lambda:{'purpose':'train_qualification'}),
            model_preflight=dict(actual_shared16_block_adapter={'original97':True}),
            prior_stop={'path':'controlled'},prior_stop_name='fixture',source_overlay_pointer={'path':'lock','sha256':'new'},execution_protocol={'schema':overlay.SCHEMA},out=Path('/new_case'))

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

    def test_blocks_use1153_not_actor_and_preserve_original_block_proof(self):
        from our_mani.multitask import rml_component_extensions_v708 as actor_factory
        events=[];runtime=self.runtime([]);prepared=self.prepared(overlay.BLOCKS[0]);triple=(object(),object(),object())
        @contextmanager
        def integrated(*args):
            events.append(args)
            try:yield triple
            finally:events.append('exit')
        block=NS(integrated_block_components=integrated,binding=Mock(return_value={'new1153':True}))
        with patch.object(overlay,'verify'),patch.object(overlay.importlib,'import_module',return_value=block) as imported,\
             patch.object(actor_factory,'create_actor') as create:
            with overlay.runtime_scope(Path('/fixture'),prepared,runtime):
                result=runtime._geometry(prepared,'existing_env',[])
                self.assertIsNone(result[0]);self.assertEqual(result[2:],triple)
                self.assertEqual(result[1]['actual_shared16_block_adapter'],{'original97':True})
                self.assertEqual(result[1]['new_block_integration'],{'new1153':True})
                self.assertEqual(events[0][-1],'existing_env');self.assertEqual(len(events),1)
        self.assertEqual(events[-1],'exit');create.assert_not_called();imported.assert_called_once_with(overlay.BLOCK_MODULE)

    def test_physmani_leaves_both_original_factories_untouched_and_imports_no_components(self):
        runtime=self.runtime([]);original_geometry=runtime._geometry;original_baseline=runtime._baseline
        with patch.object(overlay,'verify'),patch.object(overlay.importlib,'import_module') as imported:
            with overlay.runtime_scope(Path('/fixture'),self.prepared(method='physmani'),runtime):
                self.assertIs(runtime._geometry,original_geometry);self.assertIs(runtime._baseline,original_baseline)
            imported.assert_not_called()


class WorkerTests(unittest.TestCase):
    def test_default_inert_and_invalid_split_rejected_before_source_read(self):
        with patch.object(worker,'run_case',side_effect=AssertionError('not inert')),redirect_stdout(io.StringIO()):
            self.assertFalse(worker.main([])['enabled'])
        for split in ('validation','test'):
            a=args_for();a.split=split
            with patch.object(overlay,'raw',side_effect=AssertionError('source read')):
                with self.assertRaises(ValueError):overlay.selection(worker.ROOT,a)
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            worker.parser().parse_args(['--check','--execute'])

    def test_execute_checks_controlled_exit_before_any_source_selection(self):
        a=args_for();a.check=False;a.execute=True;a.prior_stop='fixture'
        with patch.object(overlay,'require_prior_stop',side_effect=ValueError('not complete')) as gate,\
             patch.object(overlay,'selection',side_effect=AssertionError('selected before exit')):
            with self.assertRaisesRegex(ValueError,'not complete'):worker.run_case(a)
        self.assertEqual(gate.call_count,1)

    def test_pending_native_is_rejected_before_guard_environment_or_consumer_import(self):
        args=args_for();args.check=False;args.execute=True;args.prior_stop='fixture'
        with patch.object(overlay,'require_prior_stop',return_value={}),patch.object(overlay,'selection',return_value={}),\
             patch.object(overlay,'read_overlay_lock',return_value=({'integration_complete':False},{})),\
             patch.object(worker.importlib,'import_module') as imported:
            with self.assertRaisesRegex(ValueError,'pending integration'):worker.run_case(args)
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
            source_overlay=dict(integration_complete=False,unpinned_sources=['pending1140.py']),
            component_domain=NS(receipt=lambda:{'split':'train'}),model_preflight={'GPU_started':False},execution_protocol={'schema':overlay.SCHEMA})
        with TemporaryDirectory() as temporary:
            root=Path(temporary);args=args_for()
            with patch.object(worker,'ROOT',root),patch.object(overlay,'selection',return_value={}),\
                 patch.object(overlay,'read_overlay_lock',return_value=(prepared['source_overlay'],{'path':'lock','sha256':'new'})),\
                 patch.object(overlay,'prepare',return_value=prepared),patch.object(overlay,'verify'),\
                 patch.object(overlay,'pointer',side_effect=lambda root,name:dict(path=name,sha256='fixture')),\
                 patch.object(worker,'environment_proof',return_value={'synthetic820':True}),\
                 patch.object(worker.importlib,'import_module',side_effect=lambda name:modules[name]):
                result=worker.run_case(args,root=root)
            self.assertFalse(result['execution']['native_started']);self.assertFalse(result['execution']['GPU_started'])
            self.assertFalse((root/args.out).exists())
            self.assertTrue((root/args.runtime/'CPU_check/worker_sidecar.json').exists())


    def test_mock_native_false_or_fatal_is_preserved_and_numeric_bool_spoof_refused(self):
        @contextmanager
        def reservation(*args,**kwargs):yield object()
        guard=NS(reserved_resources_after_TEST=reservation,require_permit=Mock(return_value={'CPU_fixture':True}))
        for success,fatal in ((False,False),(False,True),(True,False),(0,False),(False,1)):
            with self.subTest(success=success,fatal=fatal),TemporaryDirectory() as temporary:
                root=Path(temporary);a=args_for();a.check=False;a.execute=True;a.prior_stop='CPU_fixture'
                stop={'path':'CPU_fixture','sha256':'CPU_fixture'}
                source={'path':a.overlay_lock,'sha256':'CPU_fixture'}
                prepared=dict(identity={'CPU_fixture':'identity'},execution_protocol={'schema':overlay.SCHEMA})
                case=dict(version=overlay.SCHEMA+'_case',execution_protocol=prepared['execution_protocol'],
                    source_overlay_lock=source,cleanup_completed=True,process_id=os.getpid(),
                    official_task_success=success,native_terminal_failure=fatal)
                def run(p):
                    overlay.write_json(root/a.out/'case.json',case)
                runtime=NS(run=run)
                completed=Mock()
                modules={'our_mani.multitask.post_completed_TEST_resource_guard_v1106':guard,
                    'our_mani.multitask.frozen_input_edges_v463':NS(declared_edge_scope=lambda:nullcontext(),evaluator_scope=lambda e:nullcontext()),
                    'scripts.evaluate_multitask_case_r10':object(),overlay.LEGACY_RUNTIME[:-3].replace('/','.'):runtime,
                    'our_mani.multitask.evaluation':NS(validate_completed_case=completed)}
                with patch.object(worker,'ROOT',root),patch.object(overlay,'selection',return_value={}), \
                     patch.object(overlay,'require_prior_stop',return_value=stop), \
                     patch.object(overlay,'read_overlay_lock',return_value=({'integration_complete':True},source)), \
                     patch.object(overlay,'prepare',return_value=prepared),patch.object(overlay,'verify'), \
                     patch.object(overlay,'runtime_scope',return_value=nullcontext()), \
                     patch.object(overlay,'pointer',side_effect=lambda root,name:dict(path=name,sha256='CPU_fixture')), \
                     patch.object(worker,'environment_proof',return_value={'CPU_fixture820':True}), \
                     patch.object(worker.importlib,'import_module',side_effect=lambda name:modules[name]):
                    if type(success) is bool and type(fatal) is bool:
                        result=worker.run_case(a,root=root)
                        self.assertIs(result['execution']['native_success'],success)
                        self.assertIs(result['execution']['native_failure'],fatal)
                        self.assertTrue(result['execution']['native_false_preserved'])
                        self.assertEqual(result['prior_controlled_stop'],stop)
                        completed.assert_called_once_with(case,prepared['identity'])
                        self.assertTrue((root/a.runtime/'native/worker_sidecar.json').exists())
                    else:
                        with self.assertRaisesRegex(ValueError,'clean new-protocol'):worker.run_case(a,root=root)
                        self.assertFalse((root/a.runtime/'native/worker_sidecar.json').exists())
                        failure=json.loads((root/a.runtime/'native/failure.json').read_text())
                        self.assertFalse(failure['counted_as_policy_failure'])
                        self.assertTrue(failure['original_exception_rethrown'])


class ControlledStopTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.base=overlay.PRIOR_OWNER;self.process=dict(pid=999999,start_ticks=7)
        original_stop_sha=overlay.PRIOR_STOP_SHA
        self.addCleanup(setattr,overlay,'PRIOR_STOP_SHA',original_stop_sha)
        self.jobs=[dict(task='CPU_fixture',raw_episode_id=str(i),method='cv_geometry') for i in range(192)]
        self.completed=[dict(j,attempt=1,native_success=False) for j in self.jobs[:3]]
        self.plan=dict(jobs=self.jobs,expected_cases=192)
        self.launch=dict(process=self.process)
        self.status=dict(stage='failed',active=None,attempted=3,completed=self.completed,
            failed=[dict(active=None,exception_type='ValueError',message='owner1149: STOP before next fixed source',
                native_failure_inferred=False,automatic_retry=False)])
        self.stop=dict(schema='r10_root_controlled_queue_stop_for_confirmed_repair_20261024',requested_by='root',
            reason='CPU controlled fixture',current_child_must_finish=True,task_goal_paused=False,
            Native_outcome_inferred=False,qualification_granted=False)
        self.plan_ref=self.put('plan.json',self.plan)
        p=patch.object(overlay,'PRIOR_PLAN_SHA',self.plan_ref['sha256']);p.start();self.addCleanup(p.stop)
        from our_mani.multitask import repaired_readiness_v1144 as readonly
        p=patch.object(readonly,'_not_live',return_value=True);self.dead=p.start();self.addCleanup(p.stop)

    def put(self,name,value):
        path=self.root/self.base/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return overlay.pointer(self.root,self.base+'/'+name)

    def write_complete(self,**changes):
        self.put('launch.json',self.launch);status=self.put('status.json',self.status)
        failure=self.put('failure.json',self.status);stop=self.put('STOP',self.stop)
        receipt=dict(schema=overlay.PRIOR_STOP_SCHEMA,actual_tool_exit_observed=True,controlled_queue_stop=True,
            controlled_stop_completed=True,task_goal_paused=False,natural_completion=False,cohort_complete=False,
            returncode=1,process=self.process,plan=self.plan_ref,status=status,failure=failure,stop=stop,
            completed=len(self.completed),expected_cases=192,qualification_granted=False,
            Native_failure_inferred=False,infrastructure_failure_inferred=False)
        receipt.update(changes);reference=self.put('controlled_stop_receipt.json',receipt)
        overlay.PRIOR_STOP_SHA=reference['sha256'];return reference

    def test_real_prefix_exit_one_is_accepted_but_is_never_readiness(self):
        ref=self.write_complete()
        self.assertEqual(overlay.require_prior_stop(self.root,ref['path']),ref)
        self.assertEqual(len(self.completed),3)

    def test_natural_or_boolean_exit_fake_complete_or_inferred_failure_refused(self):
        for change in (dict(returncode=True),dict(returncode=0),dict(natural_completion=True),
                       dict(cohort_complete=True),dict(qualification_granted=True),dict(Native_failure_inferred=True),
                       dict(infrastructure_failure_inferred=True),dict(completed=False)):
            ref=self.write_complete(**change)
            with self.subTest(change=change),self.assertRaises(ValueError):overlay.require_prior_stop(self.root,ref['path'])

    def test_active_child_infra_wrong_prefix_or_live_owner_cannot_be_controlled_stop(self):
        initial=deepcopy(self.status)
        for kind in ('active','infra','prefix','live','stamp'):
            self.status=deepcopy(initial);self.stop['current_child_must_finish']=True;self.dead.return_value=True
            if kind=='active':self.status['active']={'phase':'native'}
            elif kind=='infra':self.status['failed'][0]['message']='actual Native child error'
            elif kind=='prefix':self.status['completed'][0]['raw_episode_id']='different'
            elif kind=='live':self.dead.return_value=False
            else:self.stop['current_child_must_finish']=False
            ref=self.write_complete()
            with self.subTest(kind=kind),self.assertRaises(ValueError):overlay.require_prior_stop(self.root,ref['path'])

    def test_other_receipt_domain_refused_before_read(self):
        with patch.object(overlay,'raw',side_effect=AssertionError('source read')):
            with self.assertRaises(ValueError):overlay.require_prior_stop(self.root,'artifacts/framework/protocol/validation_manifest.json')


if __name__ == '__main__':unittest.main()
