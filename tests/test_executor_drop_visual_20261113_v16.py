import unittest
from types import SimpleNamespace
import numpy as np
from our_mani.executor_drop_visual_20261113_v16.visuals import capture_public_RGB

class PublicCameraDiagnosticTest(unittest.TestCase):
    def test_public_pixels_preserve_physics_clock(self):
        calls=[];camera=SimpleNamespace(handle_explicitly=lambda:calls.append('render'),capture_rgb=lambda:np.ones((2,2,3))*.5)
        backend=SimpleNamespace(cameras={'front':camera},_clock_tuple=lambda:(1.,1.))
        frames=capture_public_RGB(backend)
        self.assertEqual(calls,['render']);self.assertEqual(frames['front'].dtype,np.uint8)
        self.assertEqual(int(frames['front'][0,0,0]),127)

    def test_a_render_helper_that_advances_physics_is_rejected(self):
        time=[1.]
        camera=SimpleNamespace(handle_explicitly=lambda:time.__setitem__(0,time[0]+.05),capture_rgb=lambda:np.zeros((2,2,3)))
        backend=SimpleNamespace(cameras={'front':camera},_clock_tuple=lambda:tuple(time))
        with self.assertRaisesRegex(RuntimeError,'advanced physics'):capture_public_RGB(backend)

if __name__=='__main__':unittest.main()
