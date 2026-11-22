"""Reflection correction learns from past positions and predicts unseen ticks."""
import unittest
import numpy as np
from our_mani.contracts import Forecast
from our_mani.highspeed_control_20261111_v1.public_motion import PublicReflectionObserver


def forecast(time,p,v):
    t=np.arange(17)*.05
    return Forecast(('target',),time+t,p[None,None]+t[:,None,None]*v[None,None],
        np.tile(np.eye(3),(17,1,1,1)),np.tile(v,(17,1,1)),np.zeros((17,1,3)))


class PublicMotionContracts(unittest.TestCase):
    def test_repeated_public_turns_predict_next_reflections_without_task_bounds(self):
        p=np.array([.2,0.,.8]);v=np.array([.525,0.,0.]);rows=[]
        for i in range(80):
            rows.append((p.copy(),v.copy()))
            p=p+v*.05
            if p[0]>=.35 or p[0]<=.05:v[0]=-v[0]
        observer=PublicReflectionObserver()
        for i in range(45):observer.observe(i*.05,rows[i][0])
        original=forecast(44*.05,rows[44][0],rows[44][1]);before=original.positions.copy()
        fused,evidence=observer.fuse(original)
        self.assertTrue(evidence)
        np.testing.assert_array_equal(original.positions,before)
        np.testing.assert_allclose(fused.positions[:5,0,0],[rows[i][0][0] for i in range(44,49)],atol=1e-12)
        self.assertFalse(fused.metadata['Native_inputs_used'])
        self.assertFalse(fused.metadata['future_observations_used'])

    def test_unconfirmed_motion_retains_gru_and_missing_frame_resets_history(self):
        observer=PublicReflectionObserver();p=np.array([.2,0.,.8]);v=np.array([.1,0.,0.])
        for i in range(6):observer.observe(i*.05,p+v*i*.05)
        f=forecast(.25,p+v*.25,v);out,e=observer.fuse(f)
        self.assertFalse(e);np.testing.assert_array_equal(out.positions,f.positions)
        observer.observe(.4,p+v*.4)
        self.assertEqual(len(observer.history),1)


if __name__=='__main__':unittest.main()
