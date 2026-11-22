"""Verify actual fitted-model admission and factory restoration boundaries."""
from pathlib import Path
import json
import unittest

from our_mani.multitask import public_roles
from our_mani.multitask.colour_runtime_v18 import binding,factory_colour_scope
from our_mani.multitask.evaluation import file_sha
from our_mani.multitask.protocol import MultitaskProtocolError
from our_mani.multitask.specs import get_task_spec

ROOT=Path(__file__).resolve().parents[1]
CANDIDATE=ROOT/'artifacts/framework/multitask_r10/TRAIN_colour_distribution_candidate_actual_v2/candidate.json'


class Tests(unittest.TestCase):
    def config(self):
        pointer=dict(path=str(CANDIDATE.relative_to(ROOT)),sha256=file_sha(CANDIDATE))
        return dict(learned_artifacts={task:dict(colour=pointer) for task in (
            'insert_onto_rotating_peg','push_moving_button')},colour_runtime_v18=binding(ROOT,pointer))

    def test_actual_fitted_candidate_constructs_both_task_resolvers_and_restores_factory(self):
        c=self.config();original=public_roles.PublicRoleResolver;artifact=json.loads(CANDIDATE.read_text())
        for task in ('insert_onto_rotating_peg','push_moving_button'):
            with factory_colour_scope(ROOT,c,task,purpose='train_collection'):
                resolver=public_roles.PublicRoleResolver(get_task_spec(task),artifact,shared_semantic_permission=True)
                self.assertEqual(resolver.spec.task_name,task)
                self.assertEqual(resolver.config.minimum_sources_per_colour,8)
                self.assertEqual(resolver.config.minimum_margin,.02)
            self.assertIs(public_roles.PublicRoleResolver,original)

    def test_unqualified_formal_purpose_is_rejected_and_exception_restores_factory(self):
        c=self.config();original=public_roles.PublicRoleResolver
        with self.assertRaises(MultitaskProtocolError):
            with factory_colour_scope(ROOT,c,'insert_onto_rotating_peg',purpose='evaluation'):pass
        with self.assertRaises(RuntimeError):
            with factory_colour_scope(ROOT,c,'insert_onto_rotating_peg',purpose='train_collection'):
                raise RuntimeError('fixture actor construction failed')
        self.assertIs(public_roles.PublicRoleResolver,original)


if __name__=='__main__':unittest.main()
