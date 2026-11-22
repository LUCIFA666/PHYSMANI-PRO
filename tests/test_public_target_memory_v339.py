from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
import numpy as np
import test_colour_distribution_r10 as colour_tests
from our_mani.multitask.interaction import PolicyState
from our_mani.multitask.public_roles import PublicRoleResolutionError
from our_mani.multitask.public_target_memory_v339 import EpisodeTargetIdentity


class Tests(unittest.TestCase):
    def fixture(self):
        helper = colour_tests.ColourDistributionTests()
        self.addCleanup(helper.doCleanups)
        original, state, observation, templates = helper.runtime_fixture()
        return original, EpisodeTargetIdentity(original), state, observation, templates

    def test_original_current_colour_gate_is_required_before_any_identity_memory(self):
        original, memory, state, observation, templates = self.fixture()
        observation['observation/images']['front'][:] = 0
        with self.assertRaises(PublicRoleResolutionError):
            memory.resolve('insert onto the maroon peg', state, observation, templates)
        self.assertIsNone(memory.body_id)

    def test_current_colour_failure_does_not_switch_qualified_target_or_fabricate_pose(self):
        original, memory, state, observation, templates = self.fixture()
        roles, report = memory.resolve('insert onto the maroon peg', state, observation, templates)
        self.assertEqual(roles['selected_peg'], ('peg1',))
        observation['timestamp'] = .05
        observation['observation/images']['front'][:] = 0
        later = PolicyState(replace(state.robot, timestamp=.05), state.bodies,
            state.perception_metadata, state.role_bindings)
        with self.assertRaises(PublicRoleResolutionError):
            original.resolve('insert onto the maroon peg', later, observation, templates)
        result, report = memory.resolve('insert onto the maroon peg', later, observation, templates)
        self.assertEqual(result['selected_peg'], ('peg1',))
        self.assertFalse(report['current_colour_revalidated'])
        self.assertFalse(report['current_pose_or_geometry_qualification_granted'])
        self.assertEqual(report['identity_qualification_at'], 0.)
        self.assertIs(later.bodies, state.bodies)
        self.assertIs(later.perception_metadata, state.perception_metadata)

    def test_language_topology_or_artifact_change_invalidates_cached_identity(self):
        for kind in ('language', 'component', 'artifact'):
            original, memory, state, observation, templates = self.fixture()
            memory.resolve('insert onto the maroon peg', state, observation, templates)
            text = 'insert onto the blue peg' if kind == 'language' else 'insert onto the maroon peg'
            if kind == 'component':
                templates[1] = NS(role_id='peg1', components=(NS(name='paint', instance_ids=(999,)),))
            if kind == 'artifact':
                original.artifact['colours']['maroon']['acceptance_radius'] += .01
            with self.assertRaises(PublicRoleResolutionError):
                memory.resolve(text, state, observation, templates)

    def test_fresh_episode_does_not_inherit_previous_target(self):
        original, memory, state, observation, templates = self.fixture()
        memory.resolve('insert onto the maroon peg', state, observation, templates)
        new_episode = EpisodeTargetIdentity(original)
        observation['observation/images']['front'][:] = 0
        with self.assertRaises(PublicRoleResolutionError):
            new_episode.resolve('insert onto the maroon peg', state, observation, templates)
        self.assertIsNone(new_episode.body_id)


if __name__ == '__main__':
    unittest.main()
