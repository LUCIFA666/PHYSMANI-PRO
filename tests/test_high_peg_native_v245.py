from copy import deepcopy
from pathlib import Path
import json
import unittest
from our_mani.multitask import interaction,ring_affordance
from our_mani.multitask.high_peg_primary_fitting_v231 import PrimarySequenceLibrary,TASK
from our_mani.multitask.high_peg_native_v245 import binding,library_scope
from our_mani.multitask.protocol import MultitaskProtocolError

ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol=json.loads((ROOT/'artifacts/framework/multitask_r10/high_peg_initial_smoke_v248_setup/configuration.json').read_text())

    def test_actual_library_and_ring_scopes_restore_after_exception(self):
        old_library,old_ring=interaction.InteractionLibrary,ring_affordance.VerifiedRingAffordance
        with self.assertRaisesRegex(RuntimeError,'owned test failure'):
            with library_scope(ROOT,self.protocol,TASK):
                self.assertIs(interaction.InteractionLibrary,PrimarySequenceLibrary)
                self.assertIsNot(ring_affordance.VerifiedRingAffordance,old_ring)
                raise RuntimeError('owned test failure')
        self.assertIs(interaction.InteractionLibrary,old_library)
        self.assertIs(ring_affordance.VerifiedRingAffordance,old_ring)

    def test_other_task_or_missing_consumer_binding_rejected(self):
        with self.assertRaises(ValueError):
            with library_scope(ROOT,self.protocol,'insert_onto_rotating_peg'):pass
        changed=deepcopy(self.protocol);changed.pop('high_peg_primary_v245')
        with self.assertRaises(ValueError):
            with library_scope(ROOT,changed,TASK):pass

    def test_actual_compact_capability_loads_and_rejects_mutation(self):
        with library_scope(ROOT,self.protocol,TASK):
            capability=ring_affordance.VerifiedRingAffordance.load(ROOT,self.protocol['ring_affordances'][TASK])
            self.assertTrue(capability.permission['policy_skill_authorized'])
            self.assertEqual({p:v['completed_parent_count'] for p,v in capability.permission['phase_support'].items()},
                {'approach':100,'acquire':100})
            self.assertNotIn('actual_TRAIN_grasp_rows',capability.permission)
            capability.permission['policy_skill_authorized']=False
            with self.assertRaises(MultitaskProtocolError):capability._check()


if __name__=='__main__':unittest.main()
