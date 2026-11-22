from copy import deepcopy
import unittest
from our_mani.aligned.protocol import TickLedger
from our_mani.multitask.evaluation import case_identity,validate_completed_case
from our_mani.multitask.physmani_block_record_v545 import completion_record


class Tests(unittest.TestCase):
    def test_real_legacy_payload_and_native_outcome_are_preserved_in_a_valid_common_case(self):
        source=dict(task='pick_moving_target_on_the_table',raw_episode_id='episode0',split='train',path='CPU_fixture')
        identity=case_identity(split='train',task=source['task'],raw_episode_id='episode0__policy_aug_r0',
            parent_raw_episode_id='episode0',method='context_geometry',source_row=source,
            config_sha256='a'*64,inputs_sha256='b'*64,demo_only=True)
        initial=dict(robot_joints=[.1]*7,sensor_protocol={'render_mode':'OPENGL'})
        original=deepcopy(initial);ledger=TickLedger(ticks=73)
        alignment=dict(clock=ledger.receipt(),actual_physics_seconds=3.65,actual_task_seconds=3.65,
                       native_success_latched=True,native_terminal_failure_latched=False)
        record=completion_record(identity,process_id=123,success=True,fatal=False,ticks=73,
            initial_conditions=initial,alignment_receipt=alignment)
        self.assertTrue(validate_completed_case(record,identity)['passed'])
        self.assertEqual(initial,original)
        self.assertEqual(record['recorded_initial_conditions']['legacy_parent_initial_conditions'],original)
        self.assertIs(record['official_task_success'],True)
        self.assertEqual(record['counted_clock']['ticks'],73)


if __name__=='__main__':unittest.main()
