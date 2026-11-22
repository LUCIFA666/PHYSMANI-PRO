"""CPU-only canonical planner frame, raw-label preservation and heldout guards."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast,inspect
from textwrap import dedent
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.cube_privileged_contact_teacher_20261104_v1 import teacher
from our_mani.cube_privileged_contact_teacher_20261104_v1.worker import check
from our_mani.cube_privileged_contact_teacher_20261104_v1.contracts import validate_plan
from our_mani.local_mpc_20261102.candidates import GeometricCandidates
from our_mani.local_mpc_20261102.worker import run_p1

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'config/framework/cube_privileged_contact_teacher_TRAIN12_20261104_v1/plan.json'


class TeacherContracts(unittest.TestCase):
    def test_canonical_only_policy_copy_actual_labels_unchanged(self):
        original=dict(time=0.,held=False,object_position=np.array([.2,.1,.8]),
            object_quaternion=np.array([0.,1.,0.,0.]),object_angular_velocity=np.array([.1,.2,.3]),
            object_linear_velocity=np.array([.3,0,0]))
        before=deepcopy(original)
        raw=SimpleNamespace(read_state=lambda:original,
            _training_label_snapshot=lambda:dict(target_quaternion_world_xyzw=original['object_quaternion'].copy()))
        view=teacher.CanonicalCubeTeacherPolicyView(raw);state=view.read_state()
        np.testing.assert_array_equal(state['object_quaternion'],[0,0,0,1])
        np.testing.assert_array_equal(state['object_angular_velocity'],[0,0,0])
        for key in ('object_position','object_quaternion','object_angular_velocity','object_linear_velocity'):
            np.testing.assert_array_equal(original[key],before[key])
        np.testing.assert_array_equal(raw._training_label_snapshot()['target_quaternion_world_xyzw'],[0,1,0,0])
        state['object_position'][0]=99.;self.assertEqual(original['object_position'][0],.2)
        self.assertTrue(state['teacher_privileged_current_Cube_centre_velocity_held_goal_used'])

    def test_original_geometry_downward_R_and_upward_approach_in_canonical_frame(self):
        raw=dict(time=0.,held=False,object_quaternion=np.array([0,1,0,0]))
        prior=GeometricCandidates().update(raw)[0]
        R=Rotation.from_quat(raw['object_quaternion'])
        np.testing.assert_allclose(R.apply(prior['approach_direction']),[0,0,-1],atol=1e-12)
        # Original P1 already corrected the orientation: the faulty part is
        # the local approach direction/contact offset in a flipped body frame.
        np.testing.assert_allclose((R*Rotation.from_quat(prior['object_tcp_quaternion'])).as_matrix(),
                                   Rotation.from_quat([0,1,0,0]).as_matrix(),atol=1e-12)
        state=teacher.CanonicalCubeTeacherPolicyView(SimpleNamespace(read_state=lambda:raw)).read_state()
        candidate=GeometricCandidates().update(state)[0]
        np.testing.assert_array_equal(candidate['approach_direction'],[0,0,1])
        np.testing.assert_allclose(Rotation.from_quat(candidate['object_tcp_quaternion']).as_matrix(),
                                   Rotation.from_quat([0,1,0,0]).as_matrix(),atol=1e-12)
        self.assertEqual(candidate['deadline'],.5)
        self.assertEqual(candidate['allowed_relative_speed'],.14)

    def test_original_P1_object_and_raw_recorder_binding(self):
        self.assertIs(teacher.run_p1,run_p1)
        from our_mani.cube_privileged_contact_teacher_20261104_v1 import worker
        text=inspect.getsource(worker.collect)
        self.assertIn('TRAINLabelRecorder(raw,labels)',text)
        self.assertNotIn('TRAINLabelRecorder(view',text)
        node=ast.parse(dedent(inspect.getsource(teacher.CanonicalCubeTeacherPolicyView.read_state)))
        self.assertFalse(any(isinstance(n,ast.Attribute)and n.attr.startswith('set_')for n in ast.walk(node)))

    def test_exact12_wholeparents_heldout_rejected_before_Native(self):
        plan=validate_plan(PLAN)
        self.assertEqual(len(plan['parents']),12)
        self.assertEqual(sum(p['role']=='fit'for p in plan['parents']),8)
        self.assertEqual(sum(p['role']=='selection'for p in plan['parents']),4)
        self.assertTrue(all(p['episode_id'].endswith('_teacherpass_v1')for p in plan['parents']))
        with self.assertRaises(ValueError):check(PLAN,'Cube_native_generated_ordinary_seed2611042006')
        self.assertFalse(check(PLAN,plan['parents'][0]['parent_id'])['Native_started'])


if __name__=='__main__':unittest.main()
