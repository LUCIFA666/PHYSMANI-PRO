"""Fresh child protocol boundaries with CPU mocks; never instantiate native sim."""
from argparse import Namespace
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import evaluate_multitask_case_r10 as child
from scripts.run_multitask_fresh_queue_r10 import build_jobs
from our_mani.multitask.candidate_success import ContinuationAttempts, FEATURE_NAMES
from our_mani.multitask.evaluation import case_identity, digest, file_sha, validate_completed_case
from our_mani.multitask.protocol import MultitaskProtocolError


class MockEnv:
    def __init__(self, terminal_tick=3, close_error=False):
        self.ledger=SimpleNamespace(ticks=0,dt=.05,remaining=10.)
        self.time=0.;self.terminal_tick=terminal_tick;self.close_error=close_error
        self._native_terminal_latched=self._native_success_latched=self._native_failure_latched=False
        self.observe_calls=0;self.close_calls=0;self.closed=False

    def observe(self):
        self.observe_calls+=1
        obs={'timestamp':self.time,'observation/images':{
            name:np.full((256,256,3),self.ledger.ticks,np.uint8) for name in child.CAMERAS}}
        return obs,SimpleNamespace(timestamp=self.time,joints=np.zeros(7))

    def tick(self):
        self.ledger.ticks+=1;self.time=self.ledger.ticks*.05
        self.ledger.remaining=10.-self.time
        if self.ledger.ticks == self.terminal_tick:
            self._native_success_latched=self._native_terminal_latched=True

    def alignment_receipt(self):
        return dict(clock=dict(ticks=self.ledger.ticks,dt=.05,budget_seconds=10.,simulation_seconds=self.time),
            actual_physics_seconds=self.time,actual_task_seconds=self.time,
            native_success_latched=self._native_success_latched,
            native_terminal_failure_latched=self._native_failure_latched)

    def close(self):
        self.close_calls+=1
        if self.close_error: raise RuntimeError('cleanup backend failed')
        self.closed=True


class MockActor:
    def __init__(self):
        self.predictor='cv';self.first_selection_policy='geometry';self.last_diagnostics={}
        self.controller=SimpleNamespace(continuation_sha256='controller',library=SimpleNamespace(sha256='library'),
            attempts=ContinuationAttempts(task_name='fixture',predictor='cv',
                continuation_sha256='controller',library_sha256='library'))
        self.calls=[]

    def act(self,observation,robot,remaining):
        # Only public observation/proprioception/budget enter this policy boundary.
        self.assert_public=set(observation)=={'timestamp','observation/images'}
        self.calls.append((robot.timestamp,remaining))
        if not self.controller.attempts.records:
            self.controller.attempts.begin('acquire','candidate0',robot.timestamp,
                np.zeros(len(FEATURE_NAMES)))
        return dict(command='hold',public_command_kind='hold')

    def metrics_report(self):
        return dict(native_state_is_policy_input=False,continuation_attempts=self.controller.attempts.to_report())


class MockAdapter:
    compute_seconds=.7
    def predict(self,observation,robot): return np.array([0,0,0,0,0,0,1,1.])
    def execute(self,env,action,observer=None):
        before=env.ledger.ticks
        for _ in range(10):
            env.tick();obs,robot=env.observe()
            if observer: observer(obs,robot)
            if env._native_terminal_latched: break
        return dict(ticks=env.ledger.ticks-before)


class MockBackend:
    def __init__(self,env=None):
        self.env=env or MockEnv();self.actor_instance=MockActor();self.fail_dispatch=False
    def environment(self,p): return self.env,['public task description']
    def initial_conditions(self,env,obs,robot):
        return dict(metric_only=True,policy_use_allowed=False)
    def actor(self,p,env,descriptions): return self.actor_instance,dict(sha256='controller')
    def baseline(self,p,env,obs,robot): return MockAdapter(),object()
    def dispatch(self,env,command):
        if self.fail_dispatch: raise RuntimeError('physics backend failed')
        env.tick();return dict(ticks=1)


class FreshCaseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def write(self,path,value):
        path=self.root/path;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return path
    def prepared(self,method='cv_geometry',record_demo=False):
        source=dict(task='fixture',raw_episode_id='episode4',path='/read-only/source')
        config=self.write('config.json',dict(seed=3,baseline_checkpoint=str(self.root/'weights')))
        manifest=self.write('manifest.json',dict(entries=[source]));weights=self.root/'weights';weights.write_bytes(b'weights')
        files={str(config):file_sha(config),str(manifest):file_sha(manifest),str(weights):file_sha(weights)}
        lock=self.write('lock.json',dict(files=files))
        identity=case_identity(split='test',task='fixture',raw_episode_id='episode4',method=method,
            source_row=source,config_sha256=file_sha(config),inputs_sha256=file_sha(lock))
        return dict(args=Namespace(method=method,split='test',raw_episode_id='episode4',replica=0,
                exploration_seed=None,demo_only=False),config=json.loads(config.read_text()),config_path=config,
            manifest_path=manifest,manifest_sha256=file_sha(manifest),row=source,original_row=source,
            out=self.root/'case',identity=identity,files=files,lock_path=lock,lock_sha256=file_sha(lock),
            r09_release={},r10_validation_release={},record_demo=record_demo)

    def test_complete_only_after_cleanup_and_actual_native_receipt(self):
        p=self.prepared();backend=MockBackend()
        record=child.run_case(p,backend)
        self.assertTrue(backend.env.closed)
        self.assertEqual(record['counted_clock']['ticks'],3)
        self.assertTrue(validate_completed_case(record,p['identity'])['passed'])
        self.assertEqual(len(record['continuation_attempts']),1)
        self.assertTrue(record['continuation_attempts'][0]['official_task_success'])
        self.assertTrue(backend.actor_instance.assert_public)
        self.assertTrue((p['out']/'case.json').is_file())
        self.assertFalse((p['out']/'infrastructure_error.json').exists())

    def test_backend_and_cleanup_errors_are_censored_not_policy_negatives(self):
        for kind in ('backend','cleanup'):
            with self.subTest(kind=kind):
                p=self.prepared();p['out']=self.root/kind
                backend=MockBackend(MockEnv(close_error=kind=='cleanup'))
                backend.fail_dispatch=kind=='backend'
                with self.assertRaises(RuntimeError): child.run_case(p,backend)
                self.assertFalse((p['out']/'case.json').exists())
                report=json.loads((p['out']/'infrastructure_error.json').read_text())
                self.assertFalse(report['counted_as_policy_failure'])
                self.assertFalse(report['formal_metrics_included'])
                self.assertIsNone(report['continuation_attempts'][0]['official_task_success'])
                self.assertTrue(report['continuation_attempts'][0]['censored'])

    def test_baseline_partial_terminal_records_existing_tick_frames(self):
        p=self.prepared('physmani',record_demo=True);backend=MockBackend()
        copied=[]
        def encode(path,frames):
            self.assertTrue(backend.env.closed)
            copied.extend(int(frame[0,0,0]) for frame in frames)
            return dict(path='case.mp4',sha256='a'*64,decoded_frame_count=len(frames),fps=20.,resolution=[512,512])
        with patch.object(child,'save_mp4',side_effect=encode):
            record=child.run_case(p,backend)
        self.assertEqual(copied,[0,1,2,3])
        # Initial + 3 mandatory tick observations + normal next decision observation.
        self.assertEqual(backend.env.observe_calls,5)
        self.assertEqual(record['counted_clock']['ticks'],3)
        self.assertTrue(record['formal_test_used'])
        self.assertEqual(record['video']['decoded_frame_count'],4)

    def test_input_changed_during_execution_prevents_complete_record(self):
        p=self.prepared();backend=MockBackend()
        original=backend.dispatch
        def altered(env,command):
            result=original(env,command)
            p['config_path'].write_text('{}')
            return result
        backend.dispatch=altered
        with self.assertRaises(MultitaskProtocolError): child.run_case(p,backend)
        self.assertTrue(backend.env.closed)
        self.assertFalse((p['out']/'case.json').exists())

    def test_failed_native_reset_closes_local_environment(self):
        class ResetFailure(MockEnv):
            def __init__(self,*args,**kwargs): super().__init__();instances.append(self)
            def reset_official(self,row): raise RuntimeError('reset failed')
        instances=[]
        p=self.prepared();p['args'].task='fixture'
        with patch('our_mani.multitask.simulation.AlignedOfficialSimulation',ResetFailure):
            with self.assertRaisesRegex(RuntimeError,'reset failed'):
                child.NativeBackend().environment(p)
        self.assertEqual(len(instances),1)
        self.assertTrue(instances[0].closed)

    def test_explicit_malformed_outputs_are_policy_failures_not_infra(self):
        for method in ('cv_geometry','physmani'):
            with self.subTest(method=method):
                p=self.prepared(method);p['out']=self.root/method
                backend=MockBackend()
                if method=='cv_geometry':
                    backend.actor_instance.act=lambda *args:dict(command='joint_trajectory',joints=[0.],opening=1.)
                else:
                    adapter=MockAdapter();adapter.predict=lambda *args:np.full(8,np.nan)
                    backend.baseline=lambda *args:(adapter,object())
                record=child.run_case(p,backend)
                self.assertFalse(record['official_task_success'])
                self.assertTrue(record['policy_failure'])
                self.assertTrue(backend.env.closed)
                self.assertTrue((p['out']/'case.json').exists())
                self.assertFalse((p['out']/'infrastructure_error.json').exists())

    def test_reset_and_cleanup_errors_both_remain_traceable(self):
        class ResetAndCloseFailure(MockEnv):
            def __init__(self,*args,**kwargs): super().__init__(close_error=True)
            def reset_official(self,row): raise RuntimeError('original reset failed')
        p=self.prepared();p['args'].task='fixture'
        with patch('our_mani.multitask.simulation.AlignedOfficialSimulation',ResetAndCloseFailure):
            with self.assertRaisesRegex(Exception,'original reset failed.*cleanup backend failed'):
                child.NativeBackend().environment(p)

    def test_child_identity_matches_parent_file_lock_digest(self):
        task=next(name for name in child.TASK_SPECS if name.startswith('beat_'))
        sourcepath=self.root/'train_package_compressed'/task/'episode7'/'data.h5'
        sourcepath.parent.mkdir(parents=True);sourcepath.write_bytes(b'synthetic schema-only fixture')
        row=dict(task=task,raw_episode_id='episode7',path=str(sourcepath),split='train')
        geometry=self.write('geometry.json',{});interaction=self.write('interaction.json',{})
        config=self.write('protocol.json',dict(seed=3,dt=.05,simulation_seconds=10.,gripper_velocity=.2,
            current_instance_masks=True,information_setting='CPU mock only',learned_artifacts={task:dict(
                geometry=dict(path=str(geometry),sha256=file_sha(geometry)),
                interaction=dict(path=str(interaction),sha256=file_sha(interaction)))}))
        manifest=self.write('sources.json',dict(entries=[row]))
        files={str(path):file_sha(path) for path in (config,manifest,geometry,interaction)}
        lock=self.write('sources.lock.json',dict(files=files))
        args=Namespace(config=str(config),manifest=str(manifest),frozen_inputs=str(lock),split='train',
            task=task,raw_episode_id='episode7',method='cv_geometry',out=str(self.root/'artifacts'/'newcase'),
            replica=2,exploration_seed=17,record_demo=False,demo_only=False)
        with patch.object(child,'ROOT',self.root),patch.object(child,'BASE',self.root/'artifacts'),\
                patch.object(child,'required_local_sources',return_value=[]),\
                patch.object(child,'r09_release',return_value={}):
            p=child.preflight(args)
        parent=case_identity(split='train',task=task,raw_episode_id='episode7__policy_aug_r2',
            method='cv_geometry',source_row=row,config_sha256=file_sha(config),inputs_sha256=file_sha(lock),
            parent_raw_episode_id='episode7',replica=2,exploration_seed=17)
        self.assertEqual(p['identity'],parent)

    def test_test_release_requires_original_val_lock_superset_and_full_coverage(self):
        from our_mani.multitask.evaluation import comparison_protocol_sha256
        dependency=self.write('fixed.json',dict(fixed=True))
        evidence=self.write('evidence.json',{})
        profile=self.write('profile.json',dict(TRAIN_evidence=dict(path=str(evidence),sha256=file_sha(evidence))))
        validated=dict(policy='frozen',formal_comparison_ready=False,
            initial_condition_tolerance_profile=str(profile))
        val_config=self.write('val.config.json',validated)
        val_lock=self.write('val.lock.json',dict(files={str(path):file_sha(path)
            for path in (dependency,evidence,profile,val_config)}))
        coverage=self.write('coverage.json',dict(CPU_fixture_only=True))
        release=self.write('release.json',dict(complete=True,protocol_acceptance_passed=True,
            split='validation',TEST_used=False,formal_test_permitted=True,expected_source_cases=275,
            expected_method_cases=1375,completed_method_cases=1375,config_sha256=file_sha(val_config),
            validated_config=dict(path=str(val_config),sha256=file_sha(val_config)),
            comparison_protocol_sha256=comparison_protocol_sha256(validated),
            model_coverage_certificate=dict(path=str(coverage),sha256=file_sha(coverage)),
            validated_input_lock=dict(path=str(val_lock),sha256=file_sha(val_lock))))
        config=dict(validated,formal_comparison_ready=True,r10_validation_release=str(release))
        files={str(path):file_sha(path) for path in (dependency,val_lock,evidence,profile,val_config,coverage,release)}
        with patch.object(child,'profile_problems',return_value=[]),\
                patch('our_mani.multitask.release_preflight.require_frozen_coverage_certificate',
                    return_value=dict(passed=True,CPU_boundary_fixture_only=True)):
            report=child.test_release(config,files,'c'*64)
            self.assertTrue(report['performance_screen_is_separate_from_protocol'])
            with self.assertRaisesRegex(MultitaskProtocolError,'changed a policy/model/external condition'):
                child.test_release(dict(config,policy='changed'),files,'c'*64)
            omitted=dict(files);omitted.pop(str(dependency))
            with self.assertRaises(MultitaskProtocolError): child.test_release(config,omitted,'c'*64)
            bad=json.loads(release.read_text());bad['completed_method_cases']=1374;release.write_text(json.dumps(bad))
            files[str(release)]=file_sha(release)
            with self.assertRaises(MultitaskProtocolError): child.test_release(config,files,'c'*64)

    def test_cpu_video_encodes_and_decodes_every_copied_frame(self):
        if not child.FFMPEG.is_file(): self.skipTest('local FFmpeg absent; production gate will reject recording')
        frames=[np.full((512,512,3),value,np.uint8) for value in (0,40,80,120)]
        with patch.object(child,'ROOT',self.root):
            receipt=child.save_mp4(self.root/'recording.mp4',frames)
        self.assertEqual(receipt['decoded_frame_count'],4)
        self.assertEqual(receipt['recorded_frame_count'],4)
        self.assertEqual(receipt['resolution'],[512,512])
        self.assertEqual(receipt['path'],'recording.mp4')
        self.assertEqual(receipt['sha256'],file_sha(self.root/'recording.mp4'))


if __name__=='__main__': unittest.main()
