import json
from pathlib import Path
from types import SimpleNamespace as NS,MethodType
from unittest.mock import patch,Mock
import unittest
import numpy as np
import test_ring_operational_gauge_v60 as prior
from our_mani.contracts import Pose
from our_mani.multitask.perception import MultiBodySE3Tracker
from our_mani.multitask.interaction import InteractionLibrary
from our_mani.multitask.ring_operational_gauge_v60 import install,binding
from our_mani.multitask.rotation_equivalence import CurrentRotationEquivalence
from our_mani.multitask.physmani_ring_frontend_v570 import PublicRingFrontend,OriginalFrontend

ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def test_actual_tracker_gets_public_TCP_before_first_certified_gauge_without_old_act(self):
        template,symmetry,pose,group,center,fit=prior.Tests().setup_case()
        c=json.loads((ROOT/'artifacts/framework/multitask_r10/task_executor_smoke_v58_setup/configuration.json').read_text())
        task='insert_onto_rotating_peg';c['ring_operational_gauge_v60']={task:binding(ROOT,c,task)}
        library=InteractionLibrary.load(ROOT/c['learned_artifacts'][task]['interaction']['path'])
        tracker=MultiBodySE3Tracker([template])
        actor=NS(predictor='context',ranking='geometry',tracker=tracker,controller=NS(library=library))
        actor.act=MethodType(lambda *a:None,actor);actor.metrics_report=lambda:{}
        install(actor,c,ROOT,task)
        actor.act=Mock(side_effect=AssertionError('old controller must not execute'))
        frontend=PublicRingFrontend(actor)
        local=library.for_phase('approach')[0].at(1.)[0].rotation
        robot=NS(pose=Pose(np.zeros(3),pose.rotation@group[4]@local))
        def public_observe(instance,observation,robot):
            return instance.actor.tracker.observe(observation,modes={'ring':'external'})
        with patch.object(tracker,'_registration',return_value=fit),patch.object(OriginalFrontend,'observe',public_observe):
            states,meta=frontend.observe(prior.render([template],{'ring':pose},0.),robot)
        evidence=meta['bodies']['ring']['rotation_equivalence_evidence'];body=states[0]
        self.assertTrue(evidence.matches(body_id='ring',timestamp=0.,pose=body.pose))
        self.assertFalse(meta['bodies']['ring']['orientation_observable'])
        self.assertEqual(evidence.evidence['root_hypothesis_count'],len(fit['retained_root_candidate_poses']))
        self.assertGreater(np.linalg.norm(body.pose.rotation-pose.rotation),1.)
        actor.act.assert_not_called()
        self.assertTrue(frontend.gauge_events[0]['public_TCP_supplied_before_tracking'])
        from our_mani.multitask import perception
        self.assertIs(perception.CurrentRotationEquivalence,CurrentRotationEquivalence)


if __name__=='__main__':unittest.main()
