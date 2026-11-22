"""Execute the legacy case loop past200 ticks, with strict native-clock receipts."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from our_mani.aligned.protocol import TickLedger
from our_mani.task_time_limits_20261111_v1 import TaskBudget
from our_mani.task_time_limits_20261111_v1.legacy import install_legacy_budget
from our_mani.task_time_limits_20261111_v1.feedback_backend import ExtendedTimeFeedbackBackend,PhysManiFeedbackBackendV1
from our_mani.multitask_execution_pipeline_20261104_v1.registry import TASKS
from our_mani.task_time_limits_20261111_v1.evaluation import digest,validate_completed_case
from scripts import evaluate_multitask_case_60s_20261111_v1 as child
from tests import test_evaluate_multitask_case_r10 as old


class MockExtendedEnv(old.MockEnv):
    def __init__(self,terminal_tick=None):
        super().__init__(terminal_tick=terminal_tick)
        self.ledger=TickLedger(dt=.05,budget=10.);self.protocol={'simulation_seconds':10.}
    def observe(self):
        self.observe_calls+=1
        return {'timestamp':self.time,'observation/images':{}},SimpleNamespace(timestamp=self.time,joints=old.np.zeros(7))
    def tick(self):
        self.ledger.before();self.ledger.after(.05,.05,'unit_test');self.time=self.ledger.time
        if self.ledger.ticks==self.terminal_tick:self._native_success_latched=self._native_terminal_latched=True
    def alignment_receipt(self):
        r=super().alignment_receipt();r['clock']['budget_seconds']=self.ledger.budget;return r


class LegacyExtendedLoopTests(unittest.TestCase):
    def setUp(self):
        self.fixture=old.FreshCaseTests();self.fixture.setUp()
    def tearDown(self):self.fixture.tearDown()
    def prepared(self):
        p=self.fixture.prepared();p['config']['simulation_seconds']=10.
        p['task_budget']=TaskBudget();p['runtime_override_sources']={str(child.Path(child.__file__)):child.file_sha(child.Path(child.__file__))}
        ident=deepcopy(p['identity']);ident.pop('sha256');ident['task_time_limit_seconds']=60.;ident['budget_ticks']=1200;ident['sha256']=digest(ident);p['identity']=ident
        return p
    def test_native_success_after_old200_step_limit_stops_on_exact_tick(self):
        p=self.prepared();env=MockExtendedEnv(205);install_legacy_budget(env,p['task_budget'])
        backend=old.MockBackend(env);record=child.run_case(p,backend)
        self.assertEqual(record['counted_clock']['ticks'],205)
        self.assertTrue(record['official_task_success']);self.assertTrue(env.closed)
        self.assertFalse(record['formal_test_used']);self.assertFalse(record['original_10s_formal_comparable'])
        self.assertTrue(validate_completed_case(record,p['identity'])['passed'])
    def test_timeout_runs_all1200_ticks_and_is_not_fabricated_success(self):
        p=self.prepared();env=MockExtendedEnv();install_legacy_budget(env,p['task_budget'])
        record=child.run_case(p,old.MockBackend(env))
        self.assertEqual(record['counted_clock']['ticks'],1200);self.assertFalse(record['official_task_success'])
        self.assertEqual(record['alignment_receipt']['clock']['budget_seconds'],60.)
    def test_legacy_baseline_multi_tick_dispatch_can_finish_after_tick200(self):
        p=self.prepared();p['args'].method='physmani';p['identity']['method']='physmani'
        ident=dict(p['identity']);ident.pop('sha256');p['identity']['sha256']=digest(ident)
        env=MockExtendedEnv(205);install_legacy_budget(env,p['task_budget'])
        record=child.run_case(p,old.MockBackend(env))
        self.assertEqual(record['counted_clock']['ticks'],205)
        self.assertTrue(record['official_task_success']);self.assertEqual(env.close_calls,1)
    def test_all16_legacy_environments_share_budget_before_first_tick_only(self):
        for task in TASKS:
            env=MockExtendedEnv();env.task_name=task;source=env.protocol
            install_legacy_budget(env,TaskBudget())
            self.assertEqual(env.ledger.budget,60.);self.assertEqual(source['simulation_seconds'],10.)
            env.tick()
            with self.assertRaises(ValueError):install_legacy_budget(env,TaskBudget(80.))
    def test_generic_all16_feedback_uses1200_and_retains_original_tick_method(self):
        def initialize(backend,*args,budget_ticks,**kwargs):
            self.assertLessEqual(budget_ticks,200)
            backend._ticks=0;backend.budget_ticks=budget_ticks;backend.provenance={}
        with patch.object(PhysManiFeedbackBackendV1,'__init__',initialize):
            for task in TASKS:
                backend=ExtendedTimeFeedbackBackend(task,'unused',seed=1,variation=0)
                self.assertEqual(backend.budget_ticks,1200)
                backend._ticks=200;self.assertEqual(backend.remaining_ticks(),1000)
        self.assertIs(ExtendedTimeFeedbackBackend.tick,PhysManiFeedbackBackendV1.tick)


if __name__=='__main__':unittest.main()
