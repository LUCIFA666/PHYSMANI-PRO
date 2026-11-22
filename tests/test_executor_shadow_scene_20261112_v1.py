import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.executor_timed_tasks_20261112_v1.shadow import Scene

class Object:
    def __init__(self,handle,pose,parent=None):self.handle=handle;self.matrix=pose.copy();self.parent=parent
    def get_handle(self):return self.handle
    def get_matrix(self):return self.matrix.copy()
    def get_parent(self):return self.parent
    def set_matrix(self,*args):raise AssertionError('Native setter called')

class ShadowTests(unittest.TestCase):
    def scene(self):return Scene(Object,lambda e:Rotation.from_euler('XYZ',e).as_matrix(),lambda m:Rotation.from_matrix(m[:3,:3]).as_euler('XYZ'))
    def test_late_child_snapshot_follows_virtual_parent_without_native_writes(self):
        base=np.eye(4);base[:3,3]=[.2,.3,.8];child=base.copy();child[0,3]+=.2
        native_parent=Object(1,base);native_child=Object(2,child,native_parent);scene=self.scene()
        parent=scene.wrap(native_parent);parent.rotate([0,0,np.pi/2])
        node=scene.wrap(native_child)
        np.testing.assert_allclose(node.get_position(),[.2,.5,.8],atol=1e-12)
        receipt=scene.verify();self.assertEqual(receipt['Native_setter_calls'],0)
        self.assertEqual(receipt['maximum_matrix_error'],0.)
    def test_reparenting_and_relative_waypoint_changes_remain_virtual(self):
        one=np.eye(4);two=np.eye(4);two[0,3]=1.
        a=Object(1,one);b=Object(2,two);scene=self.scene();node=scene.wrap(a);parent=scene.wrap(b)
        node.set_parent(parent);np.testing.assert_allclose(node.get_position(),[0,0,0])
        node.set_position([.2,.1,.3],relative_to=parent)
        np.testing.assert_allclose(node.get_position(),[1.2,.1,.3])
        self.assertEqual(scene.verify()['maximum_matrix_error'],0.)

if __name__=='__main__':unittest.main()
