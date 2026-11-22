from copy import deepcopy
import unittest
from scripts.evaluate_public_position_closed_cases_with_alignment_20261103 import alignment_from_records

class FirstCloseAlignmentTests(unittest.TestCase):
 def fixture(self):
  before=dict(time=.1,requested_opening=1.,object_position=[.1,.2,.3],object_quaternion=[0,0,0,1],tcp_position=[.13,.2,.3],held=False)
  after=dict(before,time=.15,requested_opening=0.,tcp_position=[.12,.2,.3],held=False)
  decision=dict(time=.1,decision={'action':{'candidate_id':'slot'},'candidate_pool':{'candidates':[{'id':'slot','object_tcp_position':[.01,0,0],'deadline':.2}]}})
  return before,after,decision
 def test_primary_is_first_completed_tick_not_dispatch_or_firstheld(self):
  a,b,d=self.fixture();r=alignment_from_records(a,[b],[d]);self.assertTrue(r['available']);self.assertAlmostEqual(r['primary_distance_m'],.01)
  self.assertAlmostEqual(r['comparisons'][0]['distance_m'],.02);self.assertFalse(r['Native_held_at_first_observation'])
 def test_failed_or_unmatched_close_is_retained_as_unavailable(self):
  a,b,d=self.fixture();d['decision']['action']=None;r=alignment_from_records(a,[b],[d]);self.assertFalse(r['available']);self.assertEqual(r['reason'],'no_selected_action_at_close')
  b['requested_opening']=1.;self.assertEqual(alignment_from_records(a,[b],[d])['reason'],'no_recorded_open_to_close_edge')
 def test_current_object_rotation_transforms_same_localpoint(self):
  a,b,d=self.fixture();b['object_quaternion']=[0,0,1,0];b['tcp_position']=[.09,.2,.3]
  self.assertAlmostEqual(alignment_from_records(a,[b],[d])['primary_distance_m'],0.,places=14)
if __name__=='__main__':unittest.main()
