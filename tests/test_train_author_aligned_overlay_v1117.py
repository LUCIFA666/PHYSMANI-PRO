"""Synthetic CPU protocol tests; no original dataset preflight/Native launch."""
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

from our_mani.multitask import train_author_aligned_overlay_v1117 as overlay
from scripts import run_TRAIN_author_aligned_overlay_v1117_r10 as worker


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


class SourceLockTests(unittest.TestCase):
    def setUp(self):
        self.temporary=TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        for name, value in (('manifest.json', b'TRAIN source'), ('inputs.json', b'TRAIN locked inputs'),
                            ('core.py', b'new core'), ('worker.py', b'new worker')):
            (self.root/name).write_bytes(value)
        changes=dict(MANIFEST='manifest.json', MANIFEST_SHA=hashlib.sha256(b'TRAIN source').hexdigest(),
            TRAIN_INPUTS='inputs.json', TRAIN_INPUTS_SHA=hashlib.sha256(b'TRAIN locked inputs').hexdigest(),
            FROZEN_PINS={}, OVERLAY_SOURCES=('core.py','worker.py','pending.py'))
        self.constants=patch.multiple(overlay, **changes); self.constants.start(); self.addCleanup(self.constants.stop)

    def test_pending_overlay_has_no_invented_sha_and_cannot_activate_native(self):
        name=overlay.SETUP_PREFIX+'pending_lock.json'
        reference=overlay.write_overlay_lock(self.root, name)
        value, actual=overlay.read_overlay_lock(self.root, name)
        self.assertEqual(reference, actual)
        self.assertEqual(value['unpinned_sources'], ['pending.py'])
        self.assertNotIn('pending.py', value['files'])
        with self.assertRaisesRegex(ValueError, 'integration pending'):
            overlay.require_native_ready(dict(source_overlay=value))
        with self.assertRaises(FileExistsError): overlay.write_overlay_lock(self.root, name)

    def test_changed_source_and_conflicting_source_roster_are_rejected(self):
        name=overlay.SETUP_PREFIX+'lock.json'; overlay.write_overlay_lock(self.root, name)
        (self.root/'core.py').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA changed'):
            overlay.read_overlay_lock(self.root, name)
        (self.root/'core.py').write_bytes(b'new core')
        lock=json.loads((self.root/name).read_text()); lock['required_sources'].append('unapproved.py')
        (self.root/name).write_text(json.dumps(lock))
        with self.assertRaisesRegex(ValueError, 'source roster'):
            overlay.read_overlay_lock(self.root, name)

    def test_native_activation_needs_complete_cpu_evidence_for_exact_source_hash_map(self):
        (self.root/'pending.py').write_bytes(b'ready')
        files={name:overlay.pointer(self.root,name)['sha256'] for name in overlay.OVERLAY_SOURCES}
        evidence=dict(schema=overlay.CPU_EVIDENCE_SCHEMA, complete=True, tests_exit_code=0,
                      native_started=False, GPU_started=False, source_hashes=files)
        evidence_name=overlay.SETUP_PREFIX+'CPU.json'
        overlay.write_json(self.root/evidence_name, evidence)
        reference=overlay.pointer(self.root,evidence_name)
        name=overlay.SETUP_PREFIX+'complete_lock.json'
        overlay.write_overlay_lock(self.root,name,integration_complete=True,integration_cpu_evidence=reference)
        value,_=overlay.read_overlay_lock(self.root,name)
        overlay.require_native_ready(dict(source_overlay=value))
        evidence['source_hashes']={}; (self.root/evidence_name).write_text(json.dumps(evidence))
        with self.assertRaisesRegex(ValueError, 'SHA changed'): overlay.read_overlay_lock(self.root,name)


class SelectionAndPrepareTests(unittest.TestCase):
    def test_fixed_first_four_and_original_config_checkpoint_are_enforced(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary); task='insert_onto_rotating_peg'
            config_name='artifacts/framework/multitask_r10/post_training_qualification_v680_setup/'+task+'_ours_configuration.json'
            config=dict(context_checkpoint=overlay.CHECKPOINT)
            config_bytes=encoded(config); path=root/config_name; path.parent.mkdir(parents=True); path.write_bytes(config_bytes)
            manifest_bytes=encoded(dict(split='train',episodes=[dict(task=task,raw_episode_id=str(i)) for i in range(5)]))
            lock_bytes=encoded(dict(files={str(root/config_name):hashlib.sha256(config_bytes).hexdigest()}))
            (root/'manifest.json').write_bytes(manifest_bytes); (root/'inputs.json').write_bytes(lock_bytes)
            with patch.multiple(overlay,MANIFEST='manifest.json',MANIFEST_SHA=hashlib.sha256(manifest_bytes).hexdigest(),
                TRAIN_INPUTS='inputs.json',TRAIN_INPUTS_SHA=hashlib.sha256(lock_bytes).hexdigest()):
                args=NS(config=config_name,task=task,raw_episode_id='3',method='cv_geometry')
                self.assertEqual(overlay.selection(root,args)['fixed_parent_index'],3)
                args.raw_episode_id='4'
                with self.assertRaisesRegex(ValueError,'first four'): overlay.selection(root,args)
                args.raw_episode_id='0'; args.method='physmani'
                with self.assertRaisesRegex(ValueError,'two geometry'): overlay.selection(root,args)

    def test_original_prepare_checks_original743_but_new_identity_binds_distinct_overlay(self):
        from our_mani.multitask import rml_case_preflight_v742 as substrate
        from our_mani.multitask import author_aligned_geometry_integration_v1114b as actuator
        with TemporaryDirectory() as temporary:
            root=Path(temporary); args=NS(config='unchanged680.json',task='insert_onto_rotating_peg',raw_episode_id='101',
                method='cv_geometry',out=overlay.ARTIFACT_PREFIX+'case',overlay_lock='lock.json',check=True)
            old=dict(sha256='old_identity',config_sha256='old_config',split='train')
            base=dict(identity=old,config=dict(baseline_repo='original_native_repo'))
            source=dict(files={overlay.SOURCE:'source_sha',overlay.WORKER:'worker_sha'},integration_complete=False)
            seen=[]
            def prepare(current,forwarded,checked_worker):
                seen.append((current,forwarded,checked_worker)); return deepcopy(base)
            with patch.object(overlay,'selection',return_value={'fixed':True}),\
                 patch.object(overlay,'read_overlay_lock',return_value=(source,dict(path='lock.json',sha256='new_overlay'))),\
                 patch.object(overlay,'pointer',side_effect=lambda r,n:dict(path=n,sha256='sha')),\
                 patch.object(substrate,'prepare',side_effect=prepare),\
                 patch.object(substrate,'outcome_protocol_binding',return_value={'same741':True}),\
                 patch.object(actuator,'binding',return_value={'new1114b':True}):
                actual=overlay.prepare(root,args)
            self.assertEqual(seen[0][2],root/overlay.LEGACY_WORKER)
            forwarded=seen[0][1]
            self.assertEqual((forwarded.split,forwarded.replica,forwarded.exploration_seed),('train',0,None))
            self.assertTrue(forwarded.development_demo); self.assertFalse(forwarded.record_demo)
            self.assertEqual(actual['source_substrate_identity'],old)
            self.assertEqual(actual['identity']['config_sha256'],'old_config')
            self.assertEqual(actual['identity']['source_overlay_lock_sha256'],'new_overlay')
            self.assertNotEqual(actual['identity']['sha256'],old['sha256'])
            self.assertFalse(actual['execution_protocol']['original_745_execution_acceptance_granted'])


class RuntimeScopeTests(unittest.TestCase):
    def test_component_lifetime_output_annotation_and_restoration_on_exception(self):
        from our_mani.multitask import rml_component_extensions_v708 as factory
        events=[]; writes=[]; actor=object(); frontend=object(); policy=object(); executor=object()
        @contextmanager
        def components(root,config,task,owned):
            self.assertIs(owned,actor); events.append('enter')
            try: yield frontend,policy,executor
            finally: events.append('exit')
        runtime=NS(_geometry=object(),atomic_json=lambda path,value:writes.append((path,value)))
        original_geometry=runtime._geometry; original_write=runtime.atomic_json
        prepared=dict(source_overlay=dict(integration_complete=True),config={},
            args=NS(method='cv_geometry',task='remove_cups_from_rotating_frame'),component_domain=object(),
            source_overlay_pointer={'path':'newlock','sha256':'newsha'},execution_protocol={'schema':overlay.SCHEMA},
            out=Path('/new_case'))
        with patch.object(overlay,'verify'),patch.object(overlay.importlib,'import_module',return_value=NS(integrated_components=components)),\
             patch.object(factory,'create_actor',return_value=(actor,{'original_model':True})) as create:
            with self.assertRaisesRegex(RuntimeError,'original failure'):
                with overlay.runtime_scope(Path('/root_fixture'),prepared,runtime):
                    result=runtime._geometry(prepared,object(),['public_description'])
                    self.assertEqual(result[2:],(frontend,policy,executor)); self.assertEqual(events,['enter'])
                    runtime.atomic_json(Path('/new_case/case.json'),dict(version='old744',official_task_success=False,
                        full16_frozen_execution={'source_only':True}))
                    self.assertEqual(events,['enter'])
                    raise RuntimeError('original failure')
            create.assert_called_once()
        self.assertEqual(events,['enter','exit'])
        self.assertIs(runtime._geometry,original_geometry); self.assertIs(runtime.atomic_json,original_write)
        saved=writes[0][1]
        self.assertEqual(saved['version'],overlay.SCHEMA+'_case')
        self.assertEqual(saved['inherited_consumer_record_version'],'old744')
        self.assertIs(saved['official_task_success'],False)
        self.assertNotIn('full16_frozen_execution',saved)
        self.assertFalse(saved['original_745_execution_acceptance_granted'])


class WorkerTests(unittest.TestCase):
    def args(self):
        return NS(config='original.json',task='insert_onto_rotating_peg',raw_episode_id='101',method='cv_geometry',
            out=overlay.ARTIFACT_PREFIX+'testcase',runtime=overlay.RUNTIME_PREFIX+'testcase',
            overlay_lock=overlay.SETUP_PREFIX+'lock.json',check=True)

    def test_namespace_proof_uses_actual820_namespace_qualification_field(self):
        proof=dict(complete=True,observer_pid=os.getpid(),wrapper=worker.ENVIRONMENT['wrapper'],
            manifest=worker.ENVIRONMENT['manifest'],namespace_qualification=worker.ENVIRONMENT['qualification'])
        capture=Mock(return_value=proof)
        with patch.object(overlay,'raw'),patch.object(worker.importlib,'import_module',
                return_value=NS(capture_namespace_evidence=capture)):
            self.assertEqual(worker.environment_proof(worker.ROOT),proof)
            proof['observer_pid']=-1
            with self.assertRaisesRegex(ValueError,'same-process'): worker.environment_proof(worker.ROOT)

    def test_pending_check_does_not_call_native_consumer_or_create_case_output(self):
        @contextmanager
        def reservation(*a,**kw):yield object()
        guard=NS(reserved_resources_after_TEST=reservation,require_permit=Mock(return_value={'owned':True}))
        edges=NS(declared_edge_scope=lambda:nullcontext(),evaluator_scope=lambda e:nullcontext())
        modules={'our_mani.multitask.post_completed_TEST_resource_guard_v1106':guard,
            'our_mani.multitask.frozen_input_edges_v463':edges,'scripts.evaluate_multitask_case_r10':object()}
        prepared=dict(identity={'new_protocol':'bound'},source_substrate_identity={'old_inputs':'bound'},
            source_overlay=dict(integration_complete=False,unpinned_sources=['pending_integration.py']),
            component_domain=NS(receipt=lambda:{'split':'train'}),model_preflight={'GPU_started':False},
            execution_protocol={'schema':overlay.SCHEMA})
        lock_reference={'path':'newlock','sha256':'newsha'}
        with TemporaryDirectory() as temporary:
            root=Path(temporary); args=self.args()
            with patch.object(worker,'ROOT',root),patch.object(overlay,'selection',return_value={'fixed_parent':True}),\
                 patch.object(overlay,'read_overlay_lock',return_value=(prepared['source_overlay'],lock_reference)),\
                 patch.object(overlay,'prepare',return_value=prepared),patch.object(overlay,'verify'),\
                 patch.object(worker,'environment_proof',return_value={'namespace820':True}),\
                 patch.object(worker.importlib,'import_module',side_effect=lambda n:modules[n]):
                result=worker.run_case(args,root=root)
            self.assertFalse(result['execution']['native_started']); self.assertFalse(result['execution']['GPU_started'])
            self.assertFalse(result['execution']['native_execution_ready'])
            self.assertFalse((root/args.out).exists())
            self.assertTrue((root/args.runtime/'CPU_check/worker_sidecar.json').exists())
            self.assertEqual(guard.require_permit.call_count,2)

    def test_pending_native_is_rejected_before_importing_guard_or_runtime(self):
        args=self.args(); args.check=False
        with patch.object(overlay,'selection',return_value={}),\
             patch.object(overlay,'read_overlay_lock',return_value=({'integration_complete':False},{})),\
             patch.object(worker.importlib,'import_module') as imported:
            with self.assertRaisesRegex(ValueError,'integration pending'): worker.run_case(args)
            imported.assert_not_called()


if __name__=='__main__':unittest.main()
