from types import SimpleNamespace
import unittest
from our_mani.multitask.ring_affordance import _ring_library_fit_key


class RingReferenceTimePermission(unittest.TestCase):
    def template(self,basis='current'):
        return SimpleNamespace(phase='approach',anchor_id='ring-relative',reference_role='ring',
            reference_kind='observed_se3',manipulated_role='ring',reference_mode='external',reference_time_basis=basis)

    def test_actual_new_seven_field_fit_namespace_is_selected(self):
        template=self.template();key=_ring_library_fit_key(template)
        fits={'approach|ring-relative|ring|observed_se3|ring|external|current':{'completed_parent_count':8},
              'approach|ring-relative|ring|observed_se3|ring|external':{'completed_parent_count':1}}
        self.assertEqual(fits[key]['completed_parent_count'],8)

    def test_command_issue_or_missing_basis_cannot_receive_ring_equivalence_permission(self):
        template=self.template('command_issue')
        with self.assertRaisesRegex(ValueError,'current-reference'):_ring_library_fit_key(template)
        del template.reference_time_basis
        with self.assertRaisesRegex(ValueError,'current-reference'):_ring_library_fit_key(template)


if __name__=='__main__':unittest.main()
