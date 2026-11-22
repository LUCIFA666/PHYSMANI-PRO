from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
import numpy as np
from our_mani.multitask.physmani_rml_start_v522 import ContinuousStartPath,PhysManiGeometryExecutor,PreviousExecutor


class Tests(unittest.TestCase):
    def test_author_filtered_points_are_preserved_after_measured_start(self):
        path=NS(_path_points=np.zeros(21),_rml_handle=123)
        events=[];wrapped=ContinuousStartPath(path,np.zeros(7),events)
        selected=np.arange(14,dtype=float).reshape(2,7)/100.
        wrapped._path_points=selected.reshape(-1)
        result=path._path_points.reshape(-1,7)
        np.testing.assert_array_equal(result[0],np.zeros(7))
        np.testing.assert_array_equal(result[1:],selected)
        self.assertEqual(len(events),1)
        wrapped._rml_handle=None;self.assertIsNone(path._rml_handle)

    def test_empty_paths_or_existing_start_are_not_turned_into_artificial_motion(self):
        path=NS(_path_points=np.zeros(21));events=[]
        wrapped=ContinuousStartPath(path,np.zeros(7),events)
        wrapped._path_points=np.zeros(0);self.assertEqual(len(path._path_points),0)
        wrapped._path_points=np.r_[np.zeros(7),np.ones(7)]
        self.assertEqual(len(path._path_points),14);self.assertEqual(events,[])

    def test_reuses_existing_executor_and_restores_path_scope_on_failure(self):
        actual=np.arange(7,dtype=float)/100
        path=NS(_path_points=np.zeros(21))
        original=lambda *a,**kw:path
        env=NS(arm=NS(get_joint_positions=lambda:actual.copy()),map_service=NS(plan_path=original))
        adapter=PhysManiGeometryExecutor(dict(dt=.05,baseline_decision_ticks=10),
            observe_public_frame=lambda *a:None,admit_pose_action=lambda a:(True,{}),
            admit_joint_target=lambda q:(True,{}))
        def execute(instance,owned,action,observer=None):
            candidate=owned.map_service.plan_path()
            candidate._path_points=np.ones(14)
            np.testing.assert_array_equal(path._path_points[:7],actual)
            raise RuntimeError('test native error')
        with patch.object(PreviousExecutor,'execute',execute):
            with self.assertRaisesRegex(RuntimeError,'test native error'):
                adapter.execute(env,np.zeros(8))
        self.assertIs(env.map_service.plan_path,original)


if __name__=='__main__':unittest.main()
