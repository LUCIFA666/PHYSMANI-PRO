"""CPU-only owned-scope/witness integration checks; no Native construction."""
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from our_mani.multitask import public_task_phase_integration_v1131 as integration


class Dispatch:
    def __init__(self, source):
        self.source_policy=source
    def qualify_opening(self, requested, current):
        return requested, {'parent1121':True}, None
    def confirm_opening(self, *args, **kwargs):
        return self.source_policy.confirm_opening(*args, **kwargs)
    def receipt(self):
        return dict(parent1121=True, source=self.source_policy.receipt())


class IntegrationTests(unittest.TestCase):
    def fixture(self):
        order=[]
        source=NS(receipt=lambda:{'source_original':True},confirm_opening=Mock(return_value='confirmed'),
                  _done=lambda goal:'original_done',candidates=lambda:'original_candidates')
        dispatched=Dispatch(source);executor=NS(skill_policy=dispatched);frontend=object();actor=object()
        @contextmanager
        def parent(root,config,task,current_actor):
            self.assertIs(current_actor,actor);order.append('parent_enter')
            try:yield frontend,dispatched,executor
            finally:order.append('parent_exit')
        return NS(source=source,dispatched=dispatched,executor=executor,frontend=frontend,actor=actor,
                  parent=parent,order=order)

    def test_static_binding_records_new_and_complete_parent_source_graph_no_certificate(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            for name in integration.NEW_SOURCES:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name)
            inputs={'frozen_TRAIN.json':sha256(b'unchanged').hexdigest()}
            (root/'frozen_TRAIN.json').write_bytes(b'unchanged')
            with patch.object(integration,'SUBSTRATE_INPUTS',inputs), \
                 patch.object(integration.parent,'binding',return_value={'full1121':{'full1118':'source graph'}}) as parent:
                proof=integration.binding(root,'/original_native')
                self.assertEqual(set(proof['sources_sha256']),set(integration.NEW_SOURCES))
                self.assertEqual(proof['parent'],{'full1121':{'full1118':'source graph'}})
                self.assertEqual(proof['inherited_source_graph_locks'],inputs)
                self.assertTrue(proof['estimated_public_task_phase_candidate'])
                self.assertFalse(proof['native_or_physical_seating_or_collision_certificate'])
                self.assertFalse(proof['initial_RML_path_done_used_for_control'])
                self.assertFalse(proof['final_RML_path_done_observed_or_inferred'])
                parent.assert_called_once_with(root,'/original_native')
                (root/'frozen_TRAIN.json').write_text('changed')
                with self.assertRaises(ValueError):integration.binding(root,'/original_native')

    def test_nonadapted_core_and_button_yield_exact_parent_objects_and_methods(self):
        for task in ('remove_cups_from_rotating_frame','remove_cups_from_rotating_frame_high_speed',
                     'push_moving_button','push_moving_button_high_speed','put_cube_in_drawer'):
            with self.subTest(task=task):
                f=self.fixture();receipt=f.dispatched.receipt;done=f.source._done
                with patch.object(integration.parent,'integrated_components',side_effect=f.parent), \
                     patch.object(integration,'binding',side_effect=AssertionError('no adaptation binding')), \
                     patch.object(integration.importlib,'import_module',side_effect=AssertionError('no task import')):
                    with integration.integrated_components('/root',{},task,f.actor) as triple:
                        self.assertEqual(triple,(f.frontend,f.dispatched,f.executor))
                        self.assertEqual(f.dispatched.receipt,receipt)
                        self.assertIs(f.source._done,done)
                        self.assertEqual(f.order,['parent_enter'])
                self.assertEqual(f.order,['parent_enter','parent_exit'])

    def test_ring_returns_final_guard_and_executor_uses_same_guard_before_native_dispatch(self):
        f=self.fixture();original_done=f.source._done;guard_holder=[]
        class Guard(Dispatch):
            def __init__(self,parent):super().__init__(parent.source_policy);self.parent=parent
            def qualify_opening(self,requested,current):
                qualified,proof,proposed=self.parent.qualify_opening(requested,current)
                return (0. if qualified>=.5 and current<.5 else qualified),proof,proposed
            def confirm_opening(self,*args,**kwargs):return self.parent.confirm_opening(*args,**kwargs)
            def receipt(self):return dict(self.parent.receipt(),ring_guard=True)
        @contextmanager
        def ring(root,config,task,frontend,dispatched,executor):
            self.assertEqual((frontend,dispatched,executor),(f.frontend,f.dispatched,f.executor))
            guard=Guard(dispatched);guard_holder.append(guard);f.order.append('ring_enter')
            with patch.object(f.source,'_done',lambda goal:'estimated_seating'),patch.object(executor,'skill_policy',guard):
                try:yield frontend,guard,executor
                finally:f.order.append('ring_exit')
        with patch.object(integration.parent,'integrated_components',side_effect=f.parent), \
             patch.object(integration,'binding',return_value={'full_source_graph':True}), \
             patch.object(integration.importlib,'import_module',return_value=NS(integrated_ring_seating_scope=ring)) as imported:
            with integration.integrated_components('/root',{'baseline_repo':'/native'},integration.RING_TASKS[0],f.actor) as triple:
                frontend,guard,executor=triple
                self.assertIs(guard,guard_holder[0]);self.assertIs(executor.skill_policy,guard)
                self.assertEqual(guard.qualify_opening(1.,0.)[0],0.)
                self.assertEqual(executor.skill_policy.qualify_opening(1.,0.)[0],0.)
                self.assertEqual(f.dispatched.qualify_opening(1.,0.)[0],1.)
                self.assertEqual(guard.confirm_opening(0.,{},None,confirmed_after=1.),'confirmed')
                f.source.confirm_opening.assert_called_once()
                receipt=guard.receipt()
                self.assertTrue(receipt['ring_guard']);self.assertTrue(receipt['parent1121'])
                self.assertIn('public_task_phase_integration_v1131',receipt)
                self.assertEqual(f.source._done(None),'estimated_seating')
                self.assertEqual(f.order,['parent_enter','ring_enter'])
            self.assertIs(f.executor.skill_policy,f.dispatched)
            self.assertIs(f.source._done,original_done)
            self.assertNotIn('receipt',vars(guard_holder[0]))
        imported.assert_called_once_with(integration.RING_MODULE)
        self.assertEqual(f.order,['parent_enter','ring_enter','ring_exit','parent_exit'])

    def test_ring_scope_cannot_return_guard_without_installing_final_executor_witness(self):
        f=self.fixture();guard=Dispatch(f.source)
        @contextmanager
        def ring(*args):yield f.frontend,guard,f.executor
        with patch.object(integration.parent,'integrated_components',side_effect=f.parent), \
             patch.object(integration,'binding',return_value={}), \
             patch.object(integration.importlib,'import_module',return_value=NS(integrated_ring_seating_scope=ring)):
            with self.assertRaisesRegex(RuntimeError,'final-dispatch guard'):
                with integration.integrated_components('/root',{'baseline_repo':'/native'},integration.RING_TASKS[1],f.actor):
                    raise AssertionError('invalid final witness must not run')
        self.assertEqual(f.order,['parent_enter','parent_exit'])

    def test_buzz_scope_patches_same_source_but_returns_original_parent_wrapper_executor(self):
        f=self.fixture();old_done=f.source._done;old_candidates=f.source.candidates;old_receipt=f.dispatched.receipt
        rule={'frozen_TRAIN_21knots':True};derived=Mock(return_value=rule)
        @contextmanager
        def buzz(source,current_rule):
            self.assertIs(source,f.source);self.assertIs(current_rule,rule);f.order.append('buzz_enter')
            with patch.object(source,'candidates',lambda:'public_cursor'),patch.object(source,'_done',lambda goal:'cursor_done'):
                try:yield source
                finally:f.order.append('buzz_exit')
        with patch.object(integration.parent,'integrated_components',side_effect=f.parent), \
             patch.object(integration,'binding',return_value={}), \
             patch.object(integration.importlib,'import_module',return_value=NS(
                 derive_buzz_approach_rule=derived,owned_buzz_approach_scope=buzz)) as imported:
            with integration.integrated_components('/root',{'baseline_repo':'/native'},integration.BUZZ_TASKS[0],f.actor) as triple:
                self.assertEqual(triple,(f.frontend,f.dispatched,f.executor))
                self.assertIs(f.executor.skill_policy,f.dispatched)
                self.assertEqual(f.source.candidates(),'public_cursor')
                self.assertEqual(f.source._done(None),'cursor_done')
                self.assertIn('public_task_phase_integration_v1131',f.dispatched.receipt())
            self.assertIs(f.source._done,old_done);self.assertIs(f.source.candidates,old_candidates)
            self.assertEqual(f.dispatched.receipt,old_receipt)
        derived.assert_called_once_with('/root',{'baseline_repo':'/native'},integration.BUZZ_TASKS[0],f.source)
        imported.assert_called_once_with(integration.BUZZ_MODULE)
        self.assertEqual(f.order,['parent_enter','buzz_enter','buzz_exit','parent_exit'])

    def test_original_execution_exception_is_preserved_through_all_scope_restoration(self):
        f=self.fixture();old_done=f.source._done;error=RuntimeError('original actuator exception')
        @contextmanager
        def buzz(source,rule):
            with patch.object(source,'_done',lambda goal:False):
                try:yield source
                finally:f.order.append('buzz_exit')
        with patch.object(integration.parent,'integrated_components',side_effect=f.parent), \
             patch.object(integration,'binding',return_value={}), \
             patch.object(integration.importlib,'import_module',return_value=NS(
                 derive_buzz_approach_rule=lambda *args:{},owned_buzz_approach_scope=buzz)):
            with self.assertRaises(RuntimeError) as raised:
                with integration.integrated_components('/root',{'baseline_repo':'/native'},integration.BUZZ_TASKS[1],f.actor):
                    raise error
        self.assertIs(raised.exception,error);self.assertIs(f.source._done,old_done)
        self.assertNotIn('receipt',vars(f.dispatched))
        self.assertEqual(f.order,['parent_enter','buzz_exit','parent_exit'])

    def test_adapted_family_requires_owned_source_witness_without_duck_native_getters(self):
        f=self.fixture();f.dispatched=object()
        @contextmanager
        def parent(*args):yield f.frontend,f.dispatched,f.executor
        with patch.object(integration.parent,'integrated_components',side_effect=parent), \
             patch.object(integration,'binding',side_effect=AssertionError('no family construction')):
            with self.assertRaises(TypeError):
                with integration.integrated_components('/root',{'baseline_repo':'/native'},integration.BUZZ_TASKS[0],f.actor):pass

    def test_real1130_scope_and1121_wrapper_preserve_live_curve_trace_in_final_receipt(self):
        from tests.test_public_buzz_approach_curve_v1130 import ApproachCurveTests
        from our_mani.multitask.retained_preclose_policy_integration_v1121 import RetainedPrecloseWitness
        fixture=ApproachCurveTests();ApproachCurveTests.setUpClass()
        source,_,template=fixture.fixture()
        dispatched=RetainedPrecloseWitness(source)
        executor=NS(skill_policy=dispatched);frontend=source.frontend
        before=(source.candidates,source._done,source.receipt)
        @contextmanager
        def parent(*args):yield frontend,dispatched,executor
        root=Path(__file__).resolve().parents[1];config=fixture.configs[source.task]
        with patch.object(integration.parent,'integrated_components',side_effect=parent), \
             patch.object(integration,'binding',return_value={'source_graph':'actual1130_CPU'}):
            with integration.integrated_components(root,config,source.task,frontend.actor) as actual:
                self.assertEqual(actual,(frontend,dispatched,executor))
                row=source.candidates('approach')[0]
                self.assertLess(row['curve_index'],20)
                self.assertEqual(len(template.progress),21)
                receipt=dispatched.receipt()
                self.assertIn('public_buzz_approach_curve_v1130',receipt)
                self.assertIn('retained_preclose_dispatch_v1121',receipt)
                self.assertIn('public_task_phase_integration_v1131',receipt)
        self.assertEqual(before,(source.candidates,source._done,source.receipt))


if __name__=='__main__':unittest.main()
