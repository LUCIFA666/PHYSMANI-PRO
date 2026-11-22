"""CPU integration only: explicit factories, action scopes and actual worker argv."""
from contextlib import contextmanager,redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import importlib.util,io,json,unittest
from unittest.mock import Mock,patch
from scripts import run_task_executor_smoke_author_outcome_v743_r10 as worker
from scripts import run_task_executor_smoke_author_suite16_v744_r10 as runtime
from scripts import collect_multitask_recovery_r10 as recovery
from scripts import run_task_executor_smoke_joint16_v174_r10 as shared_source
from our_mani.multitask import simulation
from our_mani.multitask.physmani_geometry_executor_v517 import GeometryTargetRejected
from our_mani.multitask.native_outcome_sampling_v741 import AuthorActionOutcomeSamplingMixin
from our_mani.multitask.protocol import MultitaskProtocolError

_spec=importlib.util.spec_from_file_location('_sampling_fixture741',Path(__file__).with_name('test_native_outcome_sampling_v741.py'))
sampling_fixture=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(sampling_fixture)


class FactoryTests(unittest.TestCase):
    def test_owned_development_hook_enters_before_subclass_and_restores_in_reverse(self):
        class Original:
            def reset_development(self):return 'unchanged development reset'
        class Captured(Original):pass
        events=[]
        @contextmanager
        def old_scope(plan,captures):
            self.assertIs(simulation.AlignedDevelopmentSimulation,Original)
            self.assertEqual(plan,{'frozen':'original'});events.append('original_enter')
            with patch.object(simulation,'AlignedDevelopmentSimulation',Captured):
                yield
                self.assertIs(simulation.AlignedDevelopmentSimulation,Captured)
                events.append('original_exit')
        for split in ('train','validation'):
            with self.subTest(split=split),patch.object(simulation,'AlignedDevelopmentSimulation',Original),\
                 patch.object(recovery,'scoped_owned_hooks',side_effect=old_scope),\
                 patch.object(shared_source,'hook_plan_for',return_value={'frozen':'original'}):
                p={'args':SimpleNamespace(task='beat_the_rotating_buzz',split=split)};captures={}
                with runtime.outcome_environment_scope(p,captures):
                    installed=simulation.AlignedDevelopmentSimulation
                    self.assertEqual(installed.__bases__,(AuthorActionOutcomeSamplingMixin,Captured))
                    self.assertEqual(installed().reset_development(),'unchanged development reset')
                    self.assertTrue(captures['outcome_factories'][0]['original_owned_reset_hook_entered_first'])
                self.assertIs(simulation.AlignedDevelopmentSimulation,Original)
        self.assertEqual(events,['original_enter','original_exit']*2)

    def test_development_factory_restores_after_body_exception(self):
        original=simulation.AlignedDevelopmentSimulation
        @contextmanager
        def old_scope(plan,captures):yield
        with patch.object(recovery,'scoped_owned_hooks',side_effect=old_scope),patch.object(shared_source,'hook_plan_for',return_value={}):
            with self.assertRaisesRegex(RuntimeError,'body failure'):
                with runtime.outcome_environment_scope({'args':SimpleNamespace(task='beat_the_rotating_buzz',split='validation')},{}):
                    self.assertIsNot(simulation.AlignedDevelopmentSimulation,original)
                    raise RuntimeError('body failure')
        self.assertIs(simulation.AlignedDevelopmentSimulation,original)

    def test_official_factory_inherits_exact_reset_without_development_hooks(self):
        class OriginalOfficial:
            def reset_official(self,row):return ('original reset',row)
        p={'args':SimpleNamespace(task='beat_the_rotating_buzz',split='test')};captures={}
        # Synthetic class/row only; no actual TEST manifest or recording exists.
        with patch.object(simulation,'AlignedOfficialSimulation',OriginalOfficial),\
             patch.object(recovery,'scoped_owned_hooks',side_effect=AssertionError('development hook in official scope')):
            with runtime.outcome_environment_scope(p,captures):
                installed=simulation.AlignedOfficialSimulation
                self.assertEqual(installed.__bases__,(AuthorActionOutcomeSamplingMixin,OriginalOfficial))
                self.assertEqual(installed().reset_official({'fixture':True}),('original reset',{'fixture':True}))
            self.assertIs(simulation.AlignedOfficialSimulation,OriginalOfficial)
        self.assertFalse(captures['outcome_factories'][0]['development_hooks_applied'])

    def test_block_factory_keeps_original_classes_and_avoids_core_hooks(self):
        original=simulation.AlignedDevelopmentSimulation
        with patch.object(recovery,'scoped_owned_hooks',side_effect=AssertionError('block used core hook')):
            with runtime.outcome_environment_scope({'args':SimpleNamespace(task=next(iter(runtime.BLOCKS)),split='train')},{}):
                self.assertIs(simulation.AlignedDevelopmentSimulation,original)


class ExplicitBoundaryTests(unittest.TestCase):
    def test_single_hold_then_pose_consumes_fatal_at_actual_action_end(self):
        env=sampling_fixture.core(lambda tick:(False,tick==11));actions=[]
        with runtime.action_scope(env,label='public_hold',max_ticks=1,block=False,block_actions=actions):env._tick('hold')
        with runtime.action_scope(env,label='geometry_execute',max_ticks=10,block=False,block_actions=actions):
            for _ in range(10):env._tick('arm')
        receipt=runtime.sampling_receipt(env,block=False,block_actions=actions)
        self.assertEqual([(r['start_tick'],r['end_tick']) for r in receipt['completed_actions']],[(0,1),(1,11)])
        self.assertTrue(env._native_failure_latched);self.assertEqual(env.scene_steps,11)

    def test_transient_fatal_does_not_interrupt_original_ten_tick_execute(self):
        env=sampling_fixture.core(lambda tick:(False,tick==3))
        with runtime.action_scope(env,label='baseline_execute',max_ticks=10,block=False,block_actions=[]):
            for _ in range(10):env._tick('arm')
        self.assertFalse(env._native_terminal_latched);self.assertEqual(env.scene_steps,10)
        self.assertEqual(env.native_action_sampling_receipt()['intermediate_failure_samples'],1)

    def test_public_geometry_rejection_is_not_a_clean_native_failure(self):
        env=sampling_fixture.core()
        with self.assertRaises(GeometryTargetRejected):
            with runtime.action_scope(env,label='geometry_execute',max_ticks=10,block=False,block_actions=[]):
                env._tick('arm');raise GeometryTargetRejected('same public safety rejection')
        receipt=env.native_action_sampling_receipt()
        self.assertTrue(receipt['aborted']);self.assertEqual(receipt['completed_actions'],[])
        self.assertFalse(env._native_failure_latched);self.assertEqual(env.scene_steps,1)

    def test_budget_partial_and_internal_success_preserve_actual_ticks(self):
        env=sampling_fixture.core();actions=[]
        for count in [10]*19+[8]:
            with runtime.action_scope(env,label='geometry_execute',max_ticks=count,block=False,block_actions=actions):
                for _ in range(count):env._tick('arm')
        with runtime.action_scope(env,label='baseline_execute',max_ticks=10,block=False,block_actions=actions):
            env._tick('arm');env._tick('arm')
        self.assertEqual(env.scene_steps,200)
        self.assertEqual(env.native_action_sampling_receipt()['completed_actions'][-1]['ticks'],2)
        successful=sampling_fixture.core(lambda tick:(tick==2,tick==2))
        with runtime.action_scope(successful,label='geometry_execute',max_ticks=10,block=False,block_actions=[]):
            successful._tick('arm');successful._tick('arm')
        self.assertEqual(successful.scene_steps,2);self.assertTrue(successful._native_success_latched)

    def test_blocks_keep_no_fatal_conditions_and_zero_extra_task_success_calls(self):
        success=Mock(side_effect=AssertionError('extra task.success call'))
        env=SimpleNamespace(task=SimpleNamespace(_fail_conditions=[],success=success),ledger=SimpleNamespace(ticks=0),
            _before_action=Mock(),_native_success_latched=False);actions=[]
        with runtime.action_scope(env,label='geometry_execute',max_ticks=10,block=True,block_actions=actions):env.ledger.ticks=10
        with runtime.action_scope(env,label='public_hold',max_ticks=1,block=True,block_actions=actions):
            env.ledger.ticks=11;env._native_success_latched=True
        receipt=runtime.sampling_receipt(env,block=True,block_actions=actions)
        success.assert_not_called();self.assertEqual(receipt['extra_task_success_calls'],0)
        self.assertEqual(receipt['last_tick'],11);self.assertTrue(receipt['completed_actions'][-1]['sampled_outcome']['success'])
        env.task._fail_conditions.append(object())
        with self.assertRaisesRegex(ValueError,'empty block'):
            with runtime.action_scope(env,label='invalid',max_ticks=1,block=True,block_actions=actions):pass

    def test_block_exception_is_preserved_without_completed_boundary(self):
        env=SimpleNamespace(task=SimpleNamespace(_fail_conditions=[]),ledger=SimpleNamespace(ticks=0),_before_action=Mock())
        actions=[]
        with self.assertRaisesRegex(RuntimeError,'observer failure'):
            with runtime.action_scope(env,label='baseline_execute',max_ticks=10,block=True,block_actions=actions):
                env.ledger.ticks=1;raise RuntimeError('observer failure')
        self.assertEqual(actions,[])


class ActualWorkerCheckTests(unittest.TestCase):
    def test_actual_parser_uses_real743_and_CPU_proof_declares_new_consumer_and_protocol(self):
        argv=['v743','--config','fixture.json','--manifest','TRAIN_fixture.json','--frozen-inputs','lock.json',
            '--task','beat_the_rotating_buzz','--raw-episode-id','fixed-original-0','--method','cv_geometry',
            '--split','train','--out','fixture_unused','--development-demo','--check']
        domain=SimpleNamespace(receipt=lambda:{'purpose':'train_qualification','split':'train'})
        prepared=dict(identity={'fixture':True},component_domain=domain,model_preflight={'CPU':True},source_lock_sha256=None)
        output=io.StringIO()
        def prepare(root,args,actual_worker):
            self.assertEqual(actual_worker,Path(worker.__file__));self.assertTrue(args.check);return prepared
        with patch.object(worker.sys,'argv',argv),patch.object(worker.contract,'prepare',side_effect=prepare),\
             patch.object(worker.runtime,'run',side_effect=AssertionError('native/GPU started')),redirect_stdout(output):worker.main()
        proof=json.loads(output.getvalue())
        self.assertIs(proof['native_started'],False);self.assertIs(proof['GPU_started'],False)
        self.assertEqual(proof['native_runtime']['path'],'scripts/run_task_executor_smoke_author_suite16_v744_r10.py')
        self.assertEqual(proof['original_native_runtime']['path'],'scripts/run_task_executor_smoke_rml_suite16_v664_r10.py')
        self.assertEqual(proof['author_outcome_protocol'],worker.contract.outcome_protocol_binding(worker.ROOT))
        self.assertNotIn('unchanged_native_runtime',proof)

    def test_new_preflight_keeps_every_old_source_and_binds_all_new_physical_sources(self):
        from our_mani.multitask.rml_case_preflight_v710 import SOURCES as original
        self.assertTrue(set(original)<=set(worker.contract.SOURCES))
        self.assertTrue({'our_mani/multitask/native_outcome_sampling_v741.py','our_mani/multitask/rml_case_preflight_v742.py',
            'scripts/run_task_executor_smoke_author_outcome_v743_r10.py','scripts/run_task_executor_smoke_author_suite16_v744_r10.py'}<=set(worker.contract.SOURCES))


if __name__=='__main__':unittest.main()
