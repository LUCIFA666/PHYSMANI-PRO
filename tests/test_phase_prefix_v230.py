import unittest
from our_mani.multitask.specs import get_task_spec
from our_mani.multitask.phase_prefix_v230 import require_phase_prefix


class Tests(unittest.TestCase):
    def setUp(self):
        self.skills=get_task_spec('insert_onto_rotating_peg_high_speed').phase_skills
        self.full=tuple(s.name for s in self.skills)

    def test_original_native_optional_release_can_be_omitted(self):
        self.assertTrue(self.skills[-1].optional_after_native_completion)
        self.assertEqual(require_phase_prefix(self.skills,self.full[:-1],[dict(phase='release')]),self.full[:-1])

    def test_original_mandatory_insert_cannot_be_omitted(self):
        with self.assertRaises(ValueError):require_phase_prefix(self.skills,self.full[:-2],[dict(phase='insert'),dict(phase='release')])

    def test_internal_missing_phase_or_wrong_omission_receipt_rejected(self):
        with self.assertRaises(ValueError):require_phase_prefix(self.skills,self.full[1:],[])
        with self.assertRaises(ValueError):require_phase_prefix(self.skills,self.full[:-1],[])
        self.assertEqual(require_phase_prefix(self.skills,self.full,[]),self.full)


if __name__=='__main__':unittest.main()
