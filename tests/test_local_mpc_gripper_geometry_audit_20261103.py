"""Getter-only robot calibration contracts, with forbidden task access traps."""
from enum import Enum
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from our_mani.local_mpc_control_20261103.gripper_geometry_audit import audit_gripper_geometry,GeometryClockMutation


class ObjectKind(Enum):
    SHAPE=0;JOINT=1;DUMMY=4


class JointKind(Enum):
    PRISMATIC=11


class Node:
    def __init__(self,name,handle,kind=ObjectKind.DUMMY,parent=None,matrix=None):
        self.name,self.handle,self.kind,self.parent=name,handle,kind,parent
        self.matrix=np.eye(4) if matrix is None else matrix.copy();self.tree=[self]
    def get_name(self):return self.name
    def get_handle(self):return self.handle
    def get_type(self):return self.kind
    def get_parent(self):return self.parent
    def get_matrix(self):return self.matrix.copy()
    def get_objects_in_tree(self,*,exclude_base,first_generation_only):
        assert not exclude_base and not first_generation_only
        return self.tree[:]
    def __getattr__(self,name):
        if name.startswith(('set_','step','grasp','get_grasped')):raise AssertionError('forbidden mutation/task query '+name)
        raise AttributeError(name)


class Finger(Node):
    def __init__(self,*args,q=.02,**kwargs):super().__init__(*args,kind=ObjectKind.JOINT,**kwargs);self.q=q
    def get_joint_type(self):return JointKind.PRISMATIC
    def get_joint_interval(self):return False,[0.,.04]
    def get_joint_position(self):return self.q


class Shape(Node):
    def __init__(self,*args,**kwargs):super().__init__(*args,kind=ObjectKind.SHAPE,**kwargs);self.mesh_hook=None
    def get_bounding_box(self):return np.array([-.01,.01,-.02,.02,-.03,.03])
    def get_mesh_data(self):
        if self.mesh_hook is not None:self.mesh_hook()
        return np.array([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.]]),np.array([[0,1,2]]),np.array([[0.,0.,1.]])


class ForbiddenTarget:
    def get_name(self):return 'target_secret_GT'
    def __getattr__(self,name):raise AssertionError('task descendant must not be queried: '+name)


class Backend:
    def __init__(self):
        self.time=0.;self.physics=.15;self.raw=SimpleNamespace(ledger=SimpleNamespace(ticks=0))
        base=Node('Panda_base',1)
        tip_matrix=np.eye(4);tip_matrix[:3,3]=[.3,0.,.8]
        tip=Node('Panda_tip',2,parent=base,matrix=tip_matrix)
        self.arm=SimpleNamespace(get_tip=lambda:tip)
        self.gripper=Node('Panda_gripper',3,parent=base)
        second=np.diag([1.,-1.,-1.,1.]);second[:3,3]=[.01,0.,0.]
        first=Finger('Panda_gripper_joint1',4,parent=self.gripper,q=.01)
        second=Finger('Panda_gripper_joint2',5,parent=self.gripper,matrix=second,q=.03)
        self.gripper.joints=[first,second]
        self.gripper.get_open_amount=lambda:[first.q/.04,second.q/.04]
        self.pad1=Shape('Panda_leftfinger_pad',6,parent=first)
        self.pad2=Shape('Panda_rightfinger_pad',7,parent=second)
        body=Shape('Panda_gripper_visual',8,parent=self.gripper)
        secret=ForbiddenTarget()
        first.tree=[first,self.pad1];second.tree=[second,self.pad2,secret]
        self.gripper.tree=[self.gripper,first,second,self.pad1,self.pad2,body,secret]
        self.staging_receipt=dict(controlled_ticks=0,excluded_from_control_budget=True,raw_physics_seconds=.15,
            reset_home_joints=[0.]*7,scenario={'Object_GT':'MUST_NOT_COPY'})
    def clock(self):return self.time
    def _physics_clock(self):return self.physics
    @property
    def task(self):raise AssertionError('task must never be accessed')
    def read_state(self):raise AssertionError('contains privileged object/held')
    def observe_truth(self):raise AssertionError('contains privileged object/held')
    def step(self):raise AssertionError('Native startup/step forbidden')


class GeometryAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.output=Path(self.temp.name);self.backend=Backend()
    def test_two_finger_axes_intervals_meshes_and_arrays_export_without_ticks(self):
        report=audit_gripper_geometry(self.backend,self.output)
        self.assertEqual(report['clocks_before'],report['clocks_after'])
        self.assertEqual(report['controlled_physics_ticks_added'],0)
        self.assertEqual(len(report['shapes']),3)
        self.assertEqual([j['joint_type']['name'] for j in report['finger_joints']],['PRISMATIC']*2)
        self.assertEqual([j['maximum'] for j in report['finger_joints']],[.04,.04])
        self.assertEqual([j['normalized_open_amount'] for j in report['finger_joints']],[.25,.75])
        np.testing.assert_array_equal(report['finger_joints'][0]['positive_intrinsic_Z_axis_TCP'],[0,0,1])
        np.testing.assert_array_equal(report['finger_joints'][1]['positive_intrinsic_Z_axis_TCP'],[0,0,-1])
        saved=json.loads((self.output/'robot_gripper_geometry.json').read_text())
        self.assertEqual(saved,report);self.assertEqual(saved['shapes'][0]['triangle_indices'],[[0,1,2]])
        self.assertFalse(report['inner_pad_planes_calibrated'])
    def test_task_descendant_pose_held_and_staging_scenario_never_read_or_copied(self):
        report=audit_gripper_geometry(self.backend,self.output)
        self.assertFalse(report['Object_GT_read']);self.assertFalse(report['held_or_grasp_membership_read'])
        text=(self.output/'robot_gripper_geometry.json').read_text()
        self.assertNotIn('target_secret_GT',text);self.assertNotIn('MUST_NOT_COPY',text)
        self.assertEqual(report['excluded_nonhardware_descendant_occurrences'],2)
        self.assertFalse(report['tree_coverage_complete_under_declared_hardware_allowlist'])
    def test_hidden_physics_step_in_mesh_getter_is_detected_not_silenced(self):
        self.backend.pad1.mesh_hook=lambda:setattr(self.backend,'physics',.20)
        with self.assertRaises(GeometryClockMutation):audit_gripper_geometry(self.backend,self.output)
        self.assertFalse((self.output/'robot_gripper_geometry.json').exists())
    def test_unreadable_mesh_is_explicitly_incomplete_not_a_fake_pad_plane(self):
        def unavailable():raise RuntimeError('mesh unsupported')
        self.backend.pad1.mesh_hook=unavailable
        result=audit_gripper_geometry(self.backend,self.output)
        self.assertFalse(result['all_included_meshes_readable'])
        failed=next(s for s in result['shapes'] if s['name']=='Panda_leftfinger_pad')
        self.assertFalse(failed['mesh_readable']);self.assertNotIn('vertices_local',failed)
    def test_old_audit_file_cannot_be_overwritten(self):
        audit_gripper_geometry(self.backend,self.output)
        path=self.output/'robot_gripper_geometry.json';original=path.read_bytes()
        with self.assertRaises(FileExistsError):audit_gripper_geometry(self.backend,self.output)
        self.assertEqual(path.read_bytes(),original)
    def test_controlled_physics_before_calibration_is_rejected(self):
        self.backend.time=.05;self.backend.raw.ledger.ticks=1
        with self.assertRaisesRegex(ValueError,'precede'):audit_gripper_geometry(self.backend,self.output)
        self.assertFalse((self.output/'robot_gripper_geometry.json').exists())


if __name__=='__main__':unittest.main()
