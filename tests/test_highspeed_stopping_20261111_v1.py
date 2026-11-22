"""A moving reference can always brake before native joint limits."""
import unittest
import numpy as np
from our_mani.local_mpc_20261102.executor import JointLimits
from our_mani.highspeed_control_20261111_v1.tracking_v4 import reference_window


class StoppingContracts(unittest.TestCase):
    def test_accelerating_towards_limit_leaves_a_feasible_braking_sequence(self):
        limits=JointLimits([-2.9]*7,[2.9]*7,[2.]*7)
        for sign in (-1.,1.):
            r=np.full(7,sign*2.4);v=np.zeros(7);a=np.zeros(7)
            for _ in range(30):
                lo,hi=reference_window(r,v,a,limits)
                self.assertTrue(np.all(lo<=hi+1e-9))
                after=lo if sign<0 else hi
                speed=(after-r)/.05;accel=(speed-v)/.05
                self.assertTrue(np.all(np.abs(speed)<=2.+1e-8))
                self.assertTrue(np.all(np.abs(accel)<=12.+1e-8))
                r,v,a=after,speed,accel
            for _ in range(10):
                lo,hi=reference_window(r,v,a,limits)
                self.assertTrue(np.all(lo<=hi+1e-9))
                after=np.minimum(hi,np.maximum(lo,r));speed=(after-r)/.05;accel=(speed-v)/.05
                self.assertTrue(np.all(np.abs(accel)<=12.+1e-8))
                self.assertTrue(np.all(np.abs(after)<2.9))
                r,v,a=after,speed,accel
            self.assertLess(np.max(np.abs(v)),1e-8)


if __name__=='__main__':unittest.main()
