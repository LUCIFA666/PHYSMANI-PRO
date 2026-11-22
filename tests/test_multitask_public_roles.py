"""CPU public-language tests: observed colours can change body identity."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from our_mani.contracts import ObjectState, Pose, RobotState
from our_mani.multitask.interaction import PolicyState
from our_mani.multitask.public_roles import (ColourFitConfig, PublicRoleResolver,
    PublicRoleResolutionError, TRAINColourSample, fit_colour_artifact)
from our_mani.multitask.specs import get_task_spec


class TestPublicColourRoles(unittest.TestCase):
    def fixture(self, family='insert_onto_rotating_peg'):
        colours = {'red':(255,0,0), 'green':(0,255,0), 'blue':(0,0,255)}
        roles = ['peg0','peg1','peg2'] if family == 'insert_onto_rotating_peg' else ['button']
        samples = [TRAINColourSample(family, 'train:'+family+':'+str(source), role,
            ('paint',), name, np.tile(rgb,(8,1)))
            for source in range(2) for role in roles for name,rgb in colours.items()]
        artifact = fit_colour_artifact(samples, ColourFitConfig(2,.9,.05,.01,4,.1), synthetic=True)
        resolver = PublicRoleResolver(get_task_spec(family), artifact,
            shared_semantic_permission=True, allow_synthetic=True)
        bodies = {role:ObjectState(role,0.,Pose(np.array([index*.1,0.,0.])),np.zeros((3,3)),
                                  np.zeros(3),np.zeros(3))
                  for index,role in enumerate(roles)}
        robot = RobotState(0.,Pose(np.zeros(3)),np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7))
        state = PolicyState(robot,bodies,{'bodies':{role:dict(directly_visible=True,
            registration_valid=True) for role in roles}},{role:(role,) for role in roles})
        templates = [SimpleNamespace(role_id=role, components=(SimpleNamespace(name='paint',
                     instance_ids=(index+1,)),)) for index,role in enumerate(roles)]
        rgb = np.concatenate([np.tile(colours[name],(2,2,1)) for name in list(colours)[:len(roles)]],axis=1)
        mask = np.concatenate([np.full((2,2),index+1) for index in range(len(roles))],axis=1)
        observation = {'timestamp':0.,'observation/images':{'front':rgb},
                       'observation/instance_masks':{'front':mask}}
        return resolver,state,observation,templates

    def test_current_RGBD_colour_moves_target_without_any_variation_input(self):
        resolver,state,observation,templates = self.fixture()
        binding,report = resolver.resolve(['put the ring on the red spoke'],state,observation,templates)
        self.assertEqual(binding['selected_peg'],('peg0',))
        observation['observation/images']['front'] = observation['observation/images']['front'][:,::-1].copy()
        binding,report = resolver.resolve(['slide the ring onto the red colored spoke'],state,observation,templates)
        self.assertEqual(binding['selected_peg'],('peg2',))
        self.assertTrue(report['semantic_instance_bindings_are_extra_shared_prior'])
        self.assertFalse(report['native_target_or_variation_read'])

    def test_missing_or_ambiguous_public_target_fails_closed(self):
        resolver,state,observation,templates = self.fixture()
        for descriptions in (['insert the ring'],['put ring on red spoke','put ring on blue spoke']):
            with self.assertRaises(PublicRoleResolutionError):
                resolver.resolve(descriptions,state,observation,templates)
        observation['observation/images']['front'][:] = [255,0,0]
        with self.assertRaises(PublicRoleResolutionError):
            resolver.resolve(['put the ring on red spoke'],state,observation,templates)
        with self.assertRaises(PublicRoleResolutionError):
            PublicRoleResolver(get_task_spec('insert_onto_rotating_peg'),resolver.artifact,
                               shared_semantic_permission=False,allow_synthetic=True)

    def test_author_missing_space_button_language_and_TRAIN_only_colour_labels(self):
        resolver,state,observation,templates = self.fixture('push_moving_button')
        binding,_ = resolver.resolve(['push down the movingred button',
            'press the moving button with the red base'],state,observation,templates)
        self.assertEqual(binding['selected_button'],('button',))
        sample = TRAINColourSample('push_moving_button','train:push_moving_button:1','button',
                                  ('paint',),'red',np.tile([255,0,0],(4,1)))
        with self.assertRaises(PublicRoleResolutionError):
            replace(sample,split='validation')


if __name__ == '__main__':
    unittest.main()
