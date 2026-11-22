from pathlib import Path
import json
import unittest
from our_mani.multitask import ring_affordance
from our_mani.multitask.ring_runtime_v25 import binding,factory_ring_scope,RuntimeRingAffordance
from our_mani.multitask.evaluation import file_sha
from our_mani.multitask.protocol import MultitaskProtocolError

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'artifacts/framework/multitask_r10/ring_full_TRAIN_affordance_v21/report.json'


class Tests(unittest.TestCase):
    def settings(self):
        report=json.loads(REPORT.read_text());row=next(r for r in report['entries'] if r['task']=='insert_onto_rotating_peg')
        pointer=row['permission'];replay=dict(path=str(REPORT.relative_to(ROOT)),sha256=file_sha(REPORT))
        settings=binding(ROOT,pointer,replay);task=settings['task']
        return task,dict(ring_affordances={task:pointer},ring_runtime_v25={task:settings},
            learned_artifacts={task:dict(interaction=settings['library']['file'])})

    def test_actual_full_proof_loads_original_compatible_capability_and_restores_factory(self):
        task,c=self.settings();old=ring_affordance.VerifiedRingAffordance
        with factory_ring_scope(ROOT,c,task):
            capability=ring_affordance.VerifiedRingAffordance.load(ROOT,c['ring_affordances'][task])
            self.assertIsInstance(capability,old);self.assertIsInstance(capability,RuntimeRingAffordance)
            self.assertTrue(capability.permission['policy_skill_authorized'])
            self.assertNotIn('actual_TRAIN_grasp_rows',capability.permission)
        self.assertIs(ring_affordance.VerifiedRingAffordance,old)
        capability._check()
        capability.permission['policy_skill_authorized']=False
        with self.assertRaises(MultitaskProtocolError):capability._check()

    def test_wrong_library_is_rejected_before_factory_mutation(self):
        task,c=self.settings();old=ring_affordance.VerifiedRingAffordance
        c['learned_artifacts'][task]['interaction']={'path':'wrong','sha256':'0'*64}
        with self.assertRaises(MultitaskProtocolError):
            with factory_ring_scope(ROOT,c,task):pass
        self.assertIs(ring_affordance.VerifiedRingAffordance,old)


if __name__=='__main__':unittest.main()
