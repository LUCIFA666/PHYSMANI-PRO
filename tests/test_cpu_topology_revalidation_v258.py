import unittest
from our_mani.multitask.cpu_topology_revalidation_v258 import revalidate_cpu_preflight,TRANSIENT_MESSAGES
from our_mani.multitask.protocol import MultitaskProtocolError


class Tests(unittest.TestCase):
    def test_original_topology_race_revalidates_complete_CPU_callback(self):
        calls=[]
        def verify():
            calls.append('full CPU validation')
            if len(calls)<3:raise MultitaskProtocolError(next(iter(TRANSIENT_MESSAGES)))
            return {'all_original_gates_passed':True}
        value,events=revalidate_cpu_preflight(verify)
        self.assertTrue(value['all_original_gates_passed']);self.assertEqual(len(calls),3)
        self.assertEqual(len(events),2);self.assertFalse(events[0]['native_or_GPU_or_rollout_retried'])

    def test_actual_SHA_or_model_error_is_never_retried(self):
        calls=[]
        def invalid():calls.append(1);raise MultitaskProtocolError('used input missing or conflicting in explicit frozen lock')
        with self.assertRaises(MultitaskProtocolError):revalidate_cpu_preflight(invalid)
        self.assertEqual(calls,[1])

    def test_repeated_topology_instability_has_a_hard_attempt_bound(self):
        calls=[]
        def noisy():calls.append(1);raise MultitaskProtocolError(next(iter(TRANSIENT_MESSAGES)))
        with self.assertRaises(MultitaskProtocolError):revalidate_cpu_preflight(noisy,maximum_attempts=3)
        self.assertEqual(len(calls),3)


if __name__=='__main__':unittest.main()
