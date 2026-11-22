from pathlib import Path
import json
import unittest
import numpy as np
from our_mani.multitask.button_standoff import aligned, contact_radius


class Tests(unittest.TestCase):
    def test_real_TRAIN_plate_mesh_defines_contact_envelope(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root/'artifacts/framework/multitask_r10/button_reference_v389_setup/configuration.json').read_text())
        asset = json.loads((root/config['learned_artifacts']['push_moving_button']['geometry']['path']).read_text())
        radius = contact_radius(asset, .005)
        self.assertGreater(radius, .02)
        self.assertLess(radius, .024)
        self.assertFalse(aligned(np.array([.08, 0.]), np.zeros(2), radius, .2))
        self.assertTrue(aligned(np.array([.005, 0.]), np.array([.02, 0.]), radius, .2))

    def test_fast_relative_motion_cannot_authorize_a_press_under_current_alignment(self):
        self.assertFalse(aligned(np.zeros(2), np.array([.5, 0.]), .023, .2))
        self.assertTrue(aligned(np.zeros(2), np.zeros(2), .023, .2))

    def test_geometry_too_small_for_existing_fit_error_is_rejected(self):
        asset = {'bodies': {'plate': {'collision_vertices_body':
            [[-.001,-.001,0.],[-.001,.001,0.],[.001,-.001,0.],[.001,.001,0.]]}}}
        with self.assertRaisesRegex(ValueError, 'tolerance'):
            contact_radius(asset, .005)

    def test_actual_button_factory_tracks_above_then_selects_downward_offset(self):
        from types import MethodType
        from our_mani.dynamics import ConstantVelocityDynamics
        from our_mani.multitask.moving_reference import install_moving_reference
        from our_mani.multitask.button_standoff import install
        import test_button_author_v359 as actual
        helper = actual.Tests()
        helper.setUp()
        try:
            actor = helper.actor(); c = actor.controller
            install_moving_reference(actor, phases=('press',))
            root = Path(__file__).resolve().parents[1]
            config = json.loads((root/'artifacts/framework/multitask_r10/button_reference_v389_setup/configuration.json').read_text())
            asset = json.loads((root/config['learned_artifacts']['push_moving_button']['geometry']['path']).read_text())
            c.step = MethodType(lambda self,*args: {'command':'hold'}, c)
            events = install(actor, asset)
            c.phase_index = 1
            state, _ = helper.state(actor, .08, phase='press')
            c.goal = None
            forecast = ConstantVelocityDynamics().predict(list(state.bodies.values()), 1.1, .05)
            forecast.metadata['observed_until'] = state.robot.timestamp
            c.step(state, forecast, 9.)
            self.assertFalse(events[-1]['descent_permitted'])
            self.assertAlmostEqual(c.library.for_phase('press')[0].positions[-1,2],
                                   c.library.for_phase('approach')[0].positions[-1,2])
            state, _ = helper.state(actor, .001, phase='press')
            c.goal = None
            c.step(state, forecast, 9.)
            self.assertTrue(events[-1]['descent_permitted'])
            self.assertLess(c.library.for_phase('press')[0].positions[-1,2], 0.)
        finally:
            helper.doCleanups()


if __name__ == '__main__':
    unittest.main()
