from pathlib import Path
from types import SimpleNamespace as NS
from copy import deepcopy
import json
import unittest
from our_mani.multitask.bounded_transit_v180 import BoundedApproach,SCHEMA
from our_mani.multitask.cup_transit_horizon_v268 import binding,install

ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def setUp(self):
        self.c=json.loads((ROOT/'artifacts/framework/multitask_r10/cup_author_guard_v260_setup/configuration.json').read_text())
        self.task='place_cups_on_rotating_frame';self.c['cup_transit_horizon_v268']={self.task:binding(ROOT,self.c,self.task)}
        raw=deepcopy(self.c['policy_overlay']['configurations']['persistent_reference']['approach_transit'])
        raw.update(version=SCHEMA,maximum_TCP_step_m=.2)
        self.m=NS(confirmations=[],transport_phase_clock=object())
        self.m.approach_transit=BoundedApproach(self.m,raw)
        self.actor=NS(controller=NS(goal=None,feedback_step_calls=0),continuous_reference_manager=self.m,metrics_report=lambda:{})

    def test_only_prefix_horizon_and_proposal_extent_change_on_actual_class(self):
        old=deepcopy(self.m.approach_transit.normalized);clock=self.m.transport_phase_clock
        install(self.actor,self.c,ROOT,self.task)
        actual=self.m.approach_transit.normalized
        self.assertEqual(actual['segment_seconds'],.8);self.assertAlmostEqual(actual['maximum_TCP_step_m'],.32)
        self.assertEqual({k:v for k,v in actual.items() if k not in ('segment_seconds','maximum_TCP_step_m')},
            {k:v for k,v in old.items() if k not in ('segment_seconds','maximum_TCP_step_m')})
        self.assertIs(self.m.transport_phase_clock,clock)
        self.assertIsInstance(self.m.approach_transit,BoundedApproach)

    def test_missing_binding_or_already_used_controller_rejected(self):
        protocol=deepcopy(self.c);protocol.pop('cup_transit_horizon_v268')
        with self.assertRaises(ValueError):install(self.actor,protocol,ROOT,self.task)
        self.m.confirmations.append({'actual_tick':1})
        with self.assertRaises(ValueError):install(self.actor,self.c,ROOT,self.task)


if __name__=='__main__':unittest.main()
