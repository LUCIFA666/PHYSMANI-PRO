from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import json,unittest
import numpy as np
import test_remove_continuous_executor as actual
from our_mani.contracts import Pose
from our_mani.multitask.interaction import InteractionLibrary,_world_target
from our_mani.multitask.canonical_terminal_curve import terminal_curve
from our_mani.multitask.remove_terminal_executor_v470 import RemoveTerminalBackend,PHASES,TASKS
from our_mani.multitask.remove_uncertain_translation_executor import RemoveUncertainTranslationBackend
ROOT=Path(__file__).resolve().parents[1]
class Tests(unittest.TestCase):
 def test_both_real_TRAIN_libraries_stop_locally_but_retain_world_frame_motion(self):
  for task in TASKS:
   c=json.loads((ROOT/'artifacts/framework/multitask_r10/remove_uncertain_translation_v445_setup'/(task+'_configuration.json')).read_text())
   library=InteractionLibrary.load(ROOT/c['learned_artifacts'][task]['interaction']['path'])
   for phase in PHASES:
    source=library.for_phase(phase)[0];control=terminal_curve(source);endpoint=Pose.from_xyzw(source.completion_geometry_xyzw)
    for progress in (1.,1.5):
     pose,v,w,_=control.at(progress);np.testing.assert_allclose(pose.xyzw(),endpoint.xyzw(),atol=1e-10)
     np.testing.assert_allclose(v,0.,atol=1e-10);np.testing.assert_allclose(w,0.,atol=1e-10)
    ref=Pose([.1,.2,.8]);linear=np.array([.1,.2,0.]);angular=np.array([0.,0.,.3])
    pose,v,w,_=_world_target(ref,linear,angular,control,1.)
    np.testing.assert_allclose(v,linear+np.cross(angular,pose.position-ref.position),atol=1e-10)
    np.testing.assert_allclose(w,angular,atol=1e-10)
    self.assertEqual(control.duration_seconds,source.duration_seconds)
    self.assertEqual(control.position_tolerance_m,source.position_tolerance_m)
 def test_actual_remove_backend_installs_conversion_before_selection_and_keeps_wp1_guard(self):
  for task in TASKS:
   actor,model,robot=actual.Tests().fixture(task);c=actor.controller;c.step=lambda *a,**kw:'unchanged selection'
   with patch.object(RemoveUncertainTranslationBackend,'actor',return_value=(actor,{})):
    RemoveTerminalBackend(task).actor({},None,None)
   self.assertEqual(c.step(NS(robot=robot),None,9.),'unchanged selection')
   for phase in PHASES:
    t=c.library.for_phase(phase)[0];np.testing.assert_allclose(t.at(1.)[1],0.,atol=1e-10)
   t=c.library.for_phase('approach')[0];c.goal=NS(phase='approach',template=t,selected_at=.5,phase_budget=2.)
   self.assertTrue(c._phase_done(NS(robot=robot),Pose(robot.pose.position+[.049,0,0]),public_geometry=True))
   self.assertFalse(c._phase_done(NS(robot=robot),Pose(robot.pose.position+[.051,0,0]),public_geometry=True))
   np.testing.assert_allclose(model.acceleration_limits,6.)
if __name__=='__main__':unittest.main()
