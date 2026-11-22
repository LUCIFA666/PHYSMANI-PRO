"""Causality and irrevocable contact/action boundary for derived hold histories."""
from pathlib import Path
import json,unittest
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.multitask.evaluation import file_sha
from our_mani.multitask.two_center_pose import load_two_center_pose_prior
from our_mani.multitask.buzz_hold_observations_v77 import derive_hold_observations
ROOT=Path(__file__).resolve().parents[1]


class PublicFieldsOnly(dict):
    def __getitem__(self,key):
        if key.startswith(('target_','native_')):raise AssertionError('native/future labels entered observation channel')
        return super().__getitem__(key)


class Tests(unittest.TestCase):
    def fixture(self):
        p=ROOT/'artifacts/framework/multitask_r10/buzz_two_center_TRAIN_models_and_teacher_scope_v3/models/beat_the_rotating_buzz/model.json'
        m=json.loads(p.read_text());prior=load_two_center_pose_prior(ROOT,dict(path=str(p.relative_to(ROOT)),sha256=file_sha(p)),canonical_asset_pointer=m['canonical_basis']['TRAIN_CAD'])
        axis=prior.world_axis;radial=np.eye(3)[int(np.argmin(abs(axis)))];radial-=axis*np.dot(radial,axis);radial/=np.linalg.norm(radial)
        initial=np.column_stack((radial,np.cross(axis,radial),axis))@prior.local_basis.T
        times=np.arange(6)*.05;rotations=np.asarray([Rotation.from_rotvec(axis*.4*t).as_matrix()@initial for t in times])
        positions=np.zeros((6,2,3));positions[:,0]=[.2,.1,.8]
        positions[:,1]=positions[:,0]+np.einsum('nij,j->ni',rotations,prior.local_frame_to_wand_offset)
        z=PublicFieldsOnly(body_ids=np.array(['frame','wand']),times=times,positions=positions,valid=np.ones((6,2),bool),
            modes=np.zeros((6,2),int),mode_valid=np.ones((6,2),bool),robot_gripper_open=np.ones(6,bool),robot_contact=np.zeros(6,bool),
            public_command_kind=np.array(['hold']*5+['budget_stop']))
        return prior,z,rotations

    def test_current_centres_only_recover_rotation_and_twist_without_reading_native_or_future_labels(self):
        prior,z,expected=self.fixture();derived,proof=derive_hold_observations(prior,z)
        self.assertTrue(derived['rotation_observable'].all())
        np.testing.assert_allclose(derived['rotations'][:,0],expected,atol=1e-10)
        np.testing.assert_allclose(derived['angular_velocities'][1:,0],np.tile(prior.world_axis*.4,(5,1)),atol=1e-9)
        self.assertFalse(proof['post_action_pair_channel_allowed'])

    def test_contact_close_or_arm_action_never_reinitializes_pair_channel_later(self):
        for field,value in (('robot_contact',True),('robot_gripper_open',False),('public_command_kind','joint_trajectory')):
            prior,z,_=self.fixture()
            if field=='public_command_kind':z[field]=z[field].astype('U32')
            z[field][2]=value;derived,proof=derive_hold_observations(prior,z)
            self.assertTrue(derived['rotation_observable'][:2].all())
            self.assertFalse(derived['rotation_observable'][2:].any())


if __name__=='__main__':unittest.main()
