"""Assignments only: never open simulation results or trajectory files."""
from pathlib import Path
from collections import Counter
import hashlib,json,unittest

ROOT=Path(__file__).resolve().parents[1]
PARTITION=ROOT/'config/framework/P4_new_heldout_TEST_partition_v59_20261103_v1.json'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def key(row):return tuple(round(v,12) for v in row['initial_xy']+row['initial_velocity_m_s']+[float(row['speed_change_at_tick'])]+row['changed_velocity_m_s'])


class V59PartitionTests(unittest.TestCase):
    def test_exact_fixed16_motion_mix_seed_range_and_domain(self):
        p=read(PARTITION);jobs=p['jobs']
        self.assertEqual(len(jobs),16)
        self.assertEqual(Counter(j['scene'] for j in jobs),dict(static=2,uniform=8,speed_change=6))
        self.assertEqual([j['seed'] for j in jobs],list(range(261103500,261103516)))
        self.assertEqual(len({tuple(j['initial_xy']) for j in jobs}),16)
        for j in jobs:
            self.assertEqual(j['split'],'test');self.assertTrue(j['behavior_annotations_are_not_policy_inputs'])
            self.assertEqual(j['initial_velocity_m_s'][2],0.);self.assertEqual(j['changed_velocity_m_s'][2],0.)
            self.assertTrue(j['boundary_bounce_preserved']);self.assertTrue(j['initial_uniform_is_not_a_no_bounce_guarantee'])

    def test_historical48_assignment_pins_and_complete_disjointness(self):
        p=read(PARTITION);evidence_path=ROOT/p['assignment_evidence']['path']
        self.assertEqual(sha(evidence_path),p['assignment_evidence']['sha256']);e=read(evidence_path)
        self.assertEqual(e['historical_assignment_count'],48)
        self.assertEqual(e['source_assignment_counts'],dict(train=16,onpolicy_train=8,devval=8,prior_test=16))
        for relative,digest in e['source_sha256'].items():
            path=(ROOT/relative).resolve();self.assertIn(ROOT/'config/framework',path.parents)
            self.assertEqual(sha(path),digest)
        old=e['historical'];new=p['jobs']
        self.assertFalse({j['parent_id'] for j in old}&{j['parent_id'] for j in new})
        self.assertFalse({j['seed'] for j in old}&{j['seed'] for j in new})
        self.assertFalse({tuple(j['initial_xy']) for j in old}&{tuple(j['initial_xy']) for j in new})
        self.assertFalse({key(j) for j in old}&{key(j) for j in new})
        self.assertTrue(all(e['checks'].values()))

    def test_conditional_gate_cannot_be_replaced_by_process_completion_or_everhold(self):
        p=read(PARTITION);g=p['execution_gate'];rules=p['evaluation_rules']
        self.assertEqual(p['execution_status'],'conditional_not_materialized');self.assertFalse(p['Native_started'])
        self.assertFalse(p['formal_PhysMani_TEST']);self.assertFalse(p['TEST_results_or_trajectories_read'])
        self.assertTrue(g['completion_or_exit0_alone_is_not_gate_pass'])
        self.assertEqual(g['pilot']['required_actualgoal_AND_stricthold'],3)
        self.assertEqual(g['pilot']['required_final_current_publichold'],3)
        self.assertTrue(g['pilot']['historical_ever_public_hold_is_not_final_current_pass'])
        self.assertEqual(g['full8_devVAL']['minimum_actualgoal_AND_stricthold'],7)
        self.assertEqual(g['full8_devVAL']['minimum_final_current_publichold'],7)
        self.assertEqual(g['full8_devVAL']['first_close_A_mean_error_max_m'],.03)
        self.assertEqual(g['full8_devVAL']['first_close_A_max_error_max_m'],.05)
        self.assertFalse(rules['automatic_retry']);self.assertFalse(rules['failed_parent_replacement'])


if __name__=='__main__':unittest.main()
