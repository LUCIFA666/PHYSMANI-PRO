from pathlib import Path
from copy import deepcopy
import hashlib,json,unittest
from our_mani.multitask.normal_cup_primary_library_v109 import PrimarySequenceLibrary
from our_mani.multitask.interaction import InteractionLibrary,InteractionArtifactError

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'artifacts/framework/multitask_r10/normal_cup_primary_policy_v108/interaction.json'


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.payload=json.loads(MODEL.read_text())
    def test_json_key_order_does_not_change_complete_proof_or_semantic_fingerprint(self):
        p=deepcopy(self.payload);fits=p['metadata']['completed_skill_fitting']['phase_fits']
        p['metadata']['completed_skill_fitting']['phase_fits']=dict(reversed(list(fits.items())))
        a=PrimarySequenceLibrary(self.payload);b=PrimarySequenceLibrary(p)
        self.assertEqual(a.sha256,b.sha256)
        expected=hashlib.sha256(json.dumps(self.payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
        self.assertEqual(a.sha256,expected)
        self.assertEqual(len(a.templates),6)
        self.assertFalse(a.metadata['initial_sequence_policy_scope']['original_all_observed_partitions_gate_passed'])
        self.assertEqual(len(a.metadata['completed_skill_fitting']['failed_fit_keys']),2)

    def test_missing_original_failed_partition_cannot_be_hidden(self):
        p=deepcopy(self.payload);m=p['metadata'];key=m['initial_sequence_policy_scope']['unadmitted_partitions'][0]['key']
        m['completed_skill_fitting']['phase_fits'].pop(key)
        with self.assertRaisesRegex(InteractionArtifactError,'every original fit partition'):PrimarySequenceLibrary(p)

    def test_mandatory_support_failure_is_still_rejected(self):
        p=deepcopy(self.payload);m=p['metadata'];key=m['initial_sequence_policy_scope']['admitted_partition_keys'][0]
        m['completed_skill_fitting']['phase_fits'][key]['completed_parent_count']=7
        with self.assertRaisesRegex(InteractionArtifactError,'unqualified template'):PrimarySequenceLibrary(p)

    def test_original_full_scope_loader_refuses_distinct_policy_schema(self):
        with self.assertRaisesRegex(InteractionArtifactError,'completed TRAIN-fitted'):InteractionLibrary(self.payload)


if __name__=='__main__':unittest.main()
