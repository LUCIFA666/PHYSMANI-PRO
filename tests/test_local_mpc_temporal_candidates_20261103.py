import unittest
from copy import deepcopy
import numpy as np
from our_mani.local_mpc_control_20261103.temporal_candidates import CurrentWindowCandidates


def state(t,held=False):return dict(time=t,held=held,object_quaternion=[0,0,0,1])


class TemporalCandidatesTests(unittest.TestCase):
    def test_deadlines_are_fixed_and_half_expiry_does_not_replace(self):
        g=CurrentWindowCandidates();first=g.update(state(0));copy=deepcopy(first)
        later=g.update(state(.6));self.assertEqual(len(later),8)
        self.assertEqual([r['deadline'] for r in later],[r['deadline'] for r in first])
        self.assertEqual([r['id'] for r in later],[r['id'] for r in first]);self.assertEqual(first,copy)
        self.assertEqual(g.generation,0)
    def test_all_expired_creates_new_ids_without_400ms_pause(self):
        g=CurrentWindowCandidates();old=g.update(state(0));new=g.update(state(.9))
        self.assertEqual(g.generation,1);self.assertEqual(len(new),8)
        self.assertFalse({r['id'] for r in old}&{r['id'] for r in new})
        self.assertTrue(all(.9<r['deadline']<=1.7+1e-12 for r in new))
        self.assertEqual(g.events[-1]['retired'],[dict(id=r['id'],deadline=r['deadline']) for r in old])
        self.assertFalse(g.events[-1]['old_deadlines_modified'])
    def test_held_does_not_shift_committed_reference_and_release_retires(self):
        g=CurrentWindowCandidates();old=g.update(state(0));held=g.update(state(1.,True))
        self.assertEqual([r['deadline'] for r in old],[r['deadline'] for r in held]);self.assertEqual(g.generation,0)
        g.update(state(1.1,False));self.assertEqual(g.generation,1)
    def test_invalid_clock_fails_and_no_replacement_on_exact_last_deadline(self):
        g=CurrentWindowCandidates();g.update(state(0));g.update(state(.8));self.assertEqual(g.generation,0)
        with self.assertRaises(ValueError):g.update(state(.7))
        with self.assertRaises(ValueError):g.update(state(np.nan))


if __name__=='__main__':unittest.main()
