from types import SimpleNamespace as NS
import unittest
import numpy as np
import test_cup_reference_reserve as actual
from our_mani.multitask.block_next_sample_v456 import ActualFirstGate

class Tests(unittest.TestCase):
 def test_full_path_passes_but_same_next_issued_reference_fails(self):
  actor,model,robot=actual.Tests().fixture();m=actor.continuous_reference_manager
  q=robot.joints.copy();q[0]+=.04
  b=NS(reference_q=q,reference_v=np.zeros(7),env=NS(ledger=NS(dt=.05)),next_sample_events=[])
  gate=ActualFirstGate(m.native_gate,b)
  self.assertTrue(gate.original.check_joints(robot,q,np.zeros(7),1.).feasible)
  self.assertFalse(gate.check_joints(robot,q,np.zeros(7),1.).feasible)
  self.assertFalse(b.next_sample_events[-1]['passed'])
  b.reference_q=robot.joints.copy()
  self.assertTrue(gate.check_joints(robot,q,np.zeros(7),1.).feasible)
  model.dynamic_block=True
  self.assertFalse(gate.check_joints(robot,q,np.zeros(7),1.).feasible)
if __name__=='__main__':unittest.main()
