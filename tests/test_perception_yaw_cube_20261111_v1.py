"""Public geometry recovers rotated cubes and rejects unsupported substitutes."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.contracts import Pose
from our_mani.benchmark_perception import box_surface
from our_mani.perception_repair_20261111_v1.box_fit import fit_current_cube


def component(yaw=.3,size=.05,center=(.2,.2,.775)):
    points=box_surface(Pose(np.array(center),Rotation.from_rotvec([0,0,yaw]).as_matrix()),(size,)*3,600)
    return dict(points=points,views=np.tile(np.array(['front','8','16','36']),150),colors=np.tile([240.,0.,0.],(600,1)))


class YawCubeContracts(unittest.TestCase):
    def test_rotated_cube_current_centroid_is_recovered(self):
        c=component();fit,e=fit_current_cube(c,np.array([.2,0.,1.]))
        self.assertIsNotNone(fit,e);np.testing.assert_allclose(fit['center'],[.2,.2,.775],atol=.001)
        self.assertFalse(e['Native_inputs_used']);self.assertFalse(e['old_NN_certificate_reused'])
    def test_too_few_views_wrong_size_and_near_goal_are_rejected(self):
        c=component();c['views']=np.full(600,'front')
        self.assertIsNone(fit_current_cube(c,[.2,0,1.])[0])
        self.assertIsNone(fit_current_cube(component(size=.08),[.2,0,1.])[0])
        self.assertIsNone(fit_current_cube(component(),[.2,.2,.8])[0])


if __name__=='__main__':unittest.main()
