from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from dataclasses import replace
import json
import unittest
import numpy as np
import test_cup_current_tracking_v377 as prior
from our_mani.multitask.cup_reserve_executor_native import CupReserveBackend,binding,TASKS
from our_mani.multitask.task_executor_registry import SharedTaskBackend
from our_mani.multitask.reference_constraint_authority import install as install_authority
ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def fixture(self):
        actor,model,robot=prior.Tests().fixture();c,m=actor.controller,actor.continuous_reference_manager
        saved=(c.goal,m.plan,m.plan_kind,m.plan_candidate_id)
        c.goal=m.plan=None
        install_authority(actor)
        task='place_cups_on_rotating_frame'
        config=json.loads((ROOT/'artifacts/framework/multitask_r10/registered_tasks_v408_setup'/
            (task+'_configuration.json')).read_text());config['cup_reference_reserve']=binding(ROOT,config,task)
        with patch.object(SharedTaskBackend,'actor',return_value=(actor,{})):
            CupReserveBackend(task).actor(dict(config=config,args=NS(task=task)),None,None)
        c.goal,m.plan,m.plan_kind,m.plan_candidate_id=saved
        return actor,model,robot

    def test_actual_cup_renewal_survives_reference_service_upgrade_and_keeps_collision_proofs(self):
        actor,model,robot=self.fixture();m=actor.continuous_reference_manager;old=actor.controller.goal
        proof=m.check_stream(robot,.05)
        self.assertTrue(proof.feasible,proof)
        self.assertIs(actor.controller.goal,old)
        self.assertTrue(actor.metrics_report()['cup_current_tracking_v377']['events'][-1]['committed'])
        self.assertEqual(m.plan_kind,'current_acquire_rolling')
        np.testing.assert_allclose(m.native_gate.model.acceleration_limits,6.)
        np.testing.assert_allclose(m.reference_reserve_gate.model.acceleration_limits,4.)
        np.testing.assert_allclose(m.approach_transit.reference_gate.model.acceleration_limits,4.)
        self.assertTrue(model.collision_calls)

    def test_fresh_collision_still_blocks_reinstalled_cup_renewal(self):
        actor,model,robot=self.fixture();model.dynamic_block=True
        self.assertFalse(actor.continuous_reference_manager.check_stream(robot,.05).feasible)
        self.assertIsNone(actor.continuous_reference_manager.pending)

    def test_stale_forecast_and_original_phase_deadline_cannot_be_bypassed(self):
        for stale in (True,False):
            actor,_,robot=self.fixture();m=actor.continuous_reference_manager
            if stale:m.current_forecast.metadata['observed_until']=0.
            else:actor.controller.goal=replace(actor.controller.goal,phase_budget=.49)
            self.assertFalse(m.check_stream(robot,.05).feasible)


if __name__=='__main__':unittest.main()
