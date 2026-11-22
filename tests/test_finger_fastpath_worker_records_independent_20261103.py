"""Independent fastpath routing/archive guards; no Native, fit or TEST reads."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from our_mani.local_mpc_20261102 import records
import our_mani.local_mpc_control_20261103.public_grasp_candidates_v5 as v5
import our_mani.local_mpc_control_20261103.public_candidate_goal_maintenance_v56 as v56
import our_mani.local_mpc_control_20261103.finger_fastpath_integration_20261103 as integration
from tests import test_public_candidate_v56_worker_wiring_independent_20261103 as worker_fixture
from tests import test_local_mpc_future_payload_archive_independent_20261103 as payload_fixture

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT/'config/framework/P4_student_closedloop_devVAL_v56_v1/00_val_static.json'
FAST_SOURCES = {'our_mani/local_mpc_control_20261103/'+name for name in (
    'finger_response_fastpath_20261103.py','finger_fastpath_integration_20261103.py')}


class CoordinatorCaptured(RuntimeError): pass


class FakePublicObserver:
    def __init__(self, *, maintenance=False):
        self.samples=1; self.sample_calls=0
        self.public_hold_metric=SimpleNamespace(config=object())
        self.bridge=SimpleNamespace(goal_sphere_filter=False,postconfirmed_maintenance=maintenance,active=object())
        self.assembler=self.recovery=self.pilot_inference=self.candidate_cache=object()
    def sample(self):
        self.sample_calls += 1
        raise AssertionError('invalid config must fail before sensor sampling')


class FingerFastpathWorkerRecordsTests(unittest.TestCase):
    def worker_harness(self):
        return worker_fixture.V56WorkerWiringTests('test_default_and_explicit_false_construct_and_dispatch_only_original_v5')

    def config(self, **overrides):
        result=self.worker_harness().config(response_factory='coupled_joint_20261103_v1',response_controller='coupled')
        result.update(overrides); return result

    def test_worker_default_false_and_true_preserve_public_backend_route(self):
        helper=self.worker_harness()
        for flag in ('missing',False,True):
            for maintenance in (False,True):
                cfg=self.config(postconfirmed_position_maintenance_v56=maintenance)
                if flag != 'missing':cfg['finger_response_fastpath']=flag
                report=helper.mocked_worker(cfg)
                expected='v56' if maintenance else 'v5'
                self.assertEqual(report['rc'],0)
                self.assertEqual([r[0]for r in report['constructors']],[expected])
                self.assertEqual([r[0]for r in report['runners']],[expected])
                self.assertEqual(report['native_calls'],1)
                self.assertEqual(report['old42_calls'],0)
                self.assertEqual(report['closes'],1)

    def test_worker_rejects_bad_types_or_routes_before_backend(self):
        helper=self.worker_harness(); cases=[]
        for invalid in (0,1,None,'true',[],{}):cases.append((self.config(finger_response_fastpath=invalid),'p1'))
        for key,value in [('public_grasp_candidate_transformer_v5',False),('public_grasp_candidate_transformer_v5',1),
                          ('response_factory','other'),('response_controller','learned')]:
            cfg=self.config(finger_response_fastpath=True);cfg[key]=value;cases.append((cfg,'p1'))
        cases.append((self.config(finger_response_fastpath=True),'p0'))
        cases.append((self.config(finger_response_fastpath=True,public_grasp_active_visual_v42=True),'p1'))
        for cfg,mode in cases:
            with self.subTest(cfg=cfg,mode=mode):
                report=helper.mocked_worker(cfg,mode)
                self.assertEqual(report['rc'],1);self.assertEqual(report['native_calls'],0)
                self.assertEqual(report['constructors'],[]);self.assertEqual(report['runners'],[])
                self.assertEqual(report['failure']['type'],'ValueError')

    def test_real_inventory_true_adds_exact_two_and_keeps_four_payload_pins(self):
        cfg=json.loads(BASE_CONFIG.read_text())
        with TemporaryDirectory(prefix='finger_fast_inventory_CPU_',dir=ROOT/'config/framework')as directory:
            path=Path(directory)/'config.json';path.write_text(json.dumps(cfg));default=records.freeze_sources(path,'p1')
            cfg['finger_response_fastpath']=False;path.write_text(json.dumps(cfg));disabled=records.freeze_sources(path,'p1')
            own=str(path.relative_to(ROOT));self.assertEqual(set(default),set(disabled));self.assertEqual(len(disabled),124)
            self.assertEqual({k:v for k,v in default.items()if k!=own},{k:v for k,v in disabled.items()if k!=own})
            cfg['finger_response_fastpath']=True;path.write_text(json.dumps(cfg));enabled=records.freeze_sources(path,'p1')
            self.assertEqual(set(enabled)-set(disabled),FAST_SOURCES);self.assertEqual(len(enabled),126)
            self.assertEqual(set(disabled)-set(enabled),set())
            for name in FAST_SOURCES:self.assertEqual(enabled[name],records.sha(ROOT/name))
            for name,digest in payload_fixture.PAYLOADS.items():
                self.assertEqual(disabled[name],digest);self.assertEqual(enabled[name],digest)

    def test_freeze_guard_rejects_invalid_modes_and_missing_fast_sources(self):
        cfg=json.loads(BASE_CONFIG.read_text());real_sha=records.sha
        with TemporaryDirectory(prefix='finger_fast_guards_CPU_',dir=ROOT/'config/framework')as directory:
            path=Path(directory)/'config.json'
            for invalid in (0,1,None,'true',[],{}):
                bad=deepcopy(cfg);bad['finger_response_fastpath']=invalid;path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):records.freeze_sources(path,'p1')
            enabled=dict(cfg,finger_response_fastpath=True);path.write_text(json.dumps(enabled))
            for stage in (None,'p0','fk','audit','gripper'):
                with self.assertRaisesRegex(ValueError,'coupled public candidate p1'):records.freeze_sources(path,stage)
            for key,value in [('public_grasp_candidate_transformer_v5',False),('response_factory','other'),('response_controller','learned')]:
                bad=deepcopy(enabled);bad[key]=value;path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):records.freeze_sources(path,'p1')
            for missing in FAST_SOURCES:
                target=(ROOT/missing).resolve()
                def digest(p):
                    if Path(p).resolve()==target:raise FileNotFoundError('explicit_CPU_missing_source '+missing)
                    return real_sha(p)
                path.write_text(json.dumps(enabled))
                with patch.object(records,'sha',side_effect=digest),self.assertRaises(FileNotFoundError):
                    records.freeze_sources(path,'p1')
                # Missing an unused optional module does not break originalFalse.
                path.write_text(json.dumps(dict(cfg,finger_response_fastpath=False)))
                with patch.object(records,'sha',side_effect=digest):disabled=records.freeze_sources(path,'p1')
                self.assertFalse(FAST_SOURCES&set(disabled))

    def test_both_real_runner_prefixes_enable_once_between_factory_and_coordinator(self):
        for module,function,backend_name,coordinator_name,maintenance in (
            (v5,v5.run_public_grasp_candidates_v5,'ObservedPublicCandidateV5Backend','CandidateGraspV5Coordinator',False),
            (v56,v56.run_public_candidate_goal_maintenance_v56,'ObservedPublicCandidateV56Backend','CandidateGraspV56Coordinator',True)):
            for flag in ('missing',False,True):
                events=[];backend=FakePublicObserver(maintenance=maintenance)
                executor=SimpleNamespace(backend=backend,limits=object(),config=SimpleNamespace(reference_frequency=6.))
                execution=object();core=SimpleNamespace(execution_model=execution)
                def build(*args,**kwargs):events.append('factory');return core,object(),{'CPU_fixture':True}
                def enable(model,provenance):
                    self.assertIs(model,execution);events.append('enable');return dict(provenance,finger_enabled_CPU=True)
                def coordinate(*args,**kwargs):events.append('coordinator');raise CoordinatorCaptured
                cfg=self.config(postconfirmed_position_maintenance_v56=maintenance)
                if flag!='missing':cfg['finger_response_fastpath']=flag
                with patch.object(module,backend_name,FakePublicObserver),patch.object(module,'build_response_planner',side_effect=build),\
                     patch.object(module,coordinator_name,side_effect=coordinate),patch.object(integration,'enable_finger_fastpath',side_effect=enable):
                    with self.assertRaises(CoordinatorCaptured):function(backend,executor,'.',cfg)
                self.assertEqual(events,['factory','enable','coordinator'] if flag is True else ['factory','coordinator'])
                self.assertEqual(backend.sample_calls,0)

    def test_runner_boolean_guard_is_before_any_sensor_sample(self):
        for module,function,backend_name,maintenance in (
            (v5,v5.run_public_grasp_candidates_v5,'ObservedPublicCandidateV5Backend',False),
            (v56,v56.run_public_candidate_goal_maintenance_v56,'ObservedPublicCandidateV56Backend',True)):
            backend=FakePublicObserver(maintenance=maintenance);backend.samples=0
            executor=SimpleNamespace(backend=backend)
            with patch.object(module,backend_name,FakePublicObserver),self.assertRaisesRegex(ValueError,'boolean'):
                function(backend,executor,'.',self.config(finger_response_fastpath=1,postconfirmed_position_maintenance_v56=maintenance))
            self.assertEqual(backend.sample_calls,0)


if __name__=='__main__':unittest.main()
