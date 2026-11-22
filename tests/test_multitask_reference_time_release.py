"""Cross-consumer support uses actual TRAIN reference-time derivation.

These are CPU tests against the same verified raw source used by the independent
reference-time regression. No simulator, optimization or policy success claim.
"""
from copy import deepcopy
from types import SimpleNamespace
import unittest

from our_mani.multitask.artifact_fitting import interaction_source
from our_mani.multitask.release_preflight import _completion_support
from tests.test_multitask_reference_time import ReferenceTimeTests


class ReferenceTimeReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ReferenceTimeTests.setUpClass()
        cls.raw, _ = interaction_source(deepcopy(ReferenceTimeTests.case))
        cls.derived, _ = interaction_source(deepcopy(ReferenceTimeTests.case),
            reference_time_rule=ReferenceTimeTests.rule)

    def template(self, source):
        event = next(event for event in source.completion_events if event.phase == 'approach')
        binding = event.binding
        return SimpleNamespace(phase=event.phase,anchor_id=binding.anchor_id,
            reference_role=binding.reference_role,reference_kind=binding.reference_kind,
            manipulated_role=binding.manipulated_role,reference_mode=event.reference_mode,
            reference_time_basis=binding.reference_time_basis,lag_seconds=0.)

    def test_actual_derived_source_supports_seven_field_issue_partition(self):
        key, parents, attempts = _completion_support([self.derived], self.template(self.derived))
        self.assertEqual(len(key), 7)
        self.assertEqual(key[-1], 'command_issue')
        self.assertEqual(parents, {self.derived.grouping_source})
        self.assertEqual(len(attempts), 1)

    def test_original_current_source_cannot_supply_issue_partition(self):
        _, parents, attempts = _completion_support([self.raw], self.template(self.derived))
        self.assertEqual(parents, set())
        self.assertEqual(attempts, set())

    def test_derived_issue_source_cannot_supply_current_partition(self):
        _, parents, attempts = _completion_support([self.derived], self.template(self.raw))
        self.assertEqual(parents, set())
        self.assertEqual(attempts, set())

    def test_unchanged_current_source_still_supplies_current_partition(self):
        key, parents, _ = _completion_support([self.raw], self.template(self.raw))
        self.assertEqual(key[-1], 'current')
        self.assertEqual(parents, {self.raw.grouping_source})


if __name__ == '__main__':
    unittest.main()
