import unittest
from types import SimpleNamespace as NS
from our_mani.contracts import Pose
from our_mani.multitask.remove_runtime_v48 import nearest_roles


class Tests(unittest.TestCase):
    def state(self):
        bodies={name:NS(pose=Pose(p),visible=True,timestamp=1.) for name,p in dict(cup0=[.3,0,1],cup1=[.1,0,1],spoke0=[.29,0,1],spoke1=[.11,0,1]).items()}
        return NS(robot=NS(timestamp=1.,pose=Pose([0,0,1])),bodies=bodies,
            perception_metadata={'bodies':{k:dict(directly_visible=True,registration_valid=True) for k in bodies}})

    def test_current_public_nearest_cup_and_spoke_without_native_roles(self):
        s=self.state();r=nearest_roles(s,{}, {},.1)
        self.assertEqual(r['selected_cup'],('cup1',));self.assertEqual(r['selected_spoke'],('spoke1',))

    def test_committed_identity_is_retained_and_missing_current_spokes_are_not_invented(self):
        s=self.state();r=nearest_roles(s,{}, {'selected_cup':('cup0',)},.1)
        self.assertEqual(r['selected_cup'],('cup0',));self.assertEqual(r['selected_spoke'],('spoke0',))
        for name in ('spoke0','spoke1'):s.bodies[name].visible=False
        r=nearest_roles(s,{}, {'selected_cup':('cup0',)},.1);self.assertNotIn('selected_spoke',r)


if __name__=='__main__':unittest.main()
