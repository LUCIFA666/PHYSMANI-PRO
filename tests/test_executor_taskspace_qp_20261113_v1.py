import unittest
import numpy as np
from our_mani.executor_grasp_frame_20261113_v1.reference import step
from our_mani.executor_contract_repair_20261112_v2.reference import ReferenceState
from our_mani.local_mpc_20261102.executor import JointLimits

class TaskspaceQPTests(unittest.TestCase):
    def test_position_progress_is_preserved_when_orientation_axis_is_near_singular(self):
        limits=JointLimits([-3.]*7,[3.]*7,[2.]*7,acceleration=12.,jerk=1000.)
        state=ReferenceState(np.zeros(7),np.zeros(7),np.zeros(7))
        jacobian=np.zeros((6,7));jacobian[0,0]=1.;jacobian[3,6]=.001
        desired=np.zeros(7);desired[0]=1.;desired[6]=10.
        result,receipt=step(state,desired,limits,actual_q=np.zeros(7),task_jacobian=jacobian,
            task_velocity=[1.,0,0,10.,0,0],null_velocity=np.zeros(7))
        self.assertGreater(result.velocity[0],.55)
        self.assertLess(abs(result.velocity[6]),.1)
        self.assertLessEqual(np.linalg.norm(result.acceleration),12.+1e-7)
        self.assertTrue(receipt['task_space_objective'])

if __name__=='__main__':unittest.main()
