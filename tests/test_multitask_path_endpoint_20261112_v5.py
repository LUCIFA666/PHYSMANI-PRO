import unittest
import numpy as np
from our_mani.multitask_executor_cycle_20261112_v5.sequencing import interpolate
from our_mani.multitask_executor_cycle_20261112_v5.common import ROOT

class EndpointRegression(unittest.TestCase):
    def test_RML_scalar_overshoot_returns_exact_original_endpoint(self):
        path=np.zeros((2,7));path[-1,-1]=2.8973000049591064
        lengths=np.array([0.,path[-1,-1]])
        for s in (lengths[-1],lengths[-1]+1e-14,lengths[-1]+.01):
            self.assertTrue(np.array_equal(interpolate(path,lengths,s),path[-1]))
        self.assertTrue(np.array_equal(interpolate(path,lengths,-1e-14),path[0]))
    def test_intermediate_positions_stay_on_original_polyline(self):
        path=np.zeros((3,7));path[1,0]=1.;path[2,0]=1.;path[2,1]=1.
        lengths=np.array([0.,1.,2.])
        self.assertTrue(np.allclose(interpolate(path,lengths,1.5),[1.,.5,0,0,0,0,0]))
    def test_owner_launcher_and_worker_use_current_bounded_sampler(self):
        owner=(ROOT/'our_mani/multitask_executor_cycle_20261112_v5/owner.py').read_text()
        self.assertIn('run_multitask_executor_cycle_native_20261112_v5.sh',owner)
        launcher=(ROOT/'scripts/run_multitask_executor_cycle_native_20261112_v5.sh').read_text()
        self.assertIn('multitask_executor_cycle_20261112_v5.native_worker',launcher)
        controller=(ROOT/'our_mani/multitask_executor_cycle_20261112_v5/controller.py').read_text()
        self.assertIn('from .sequencing import',controller)

if __name__=='__main__':unittest.main()
