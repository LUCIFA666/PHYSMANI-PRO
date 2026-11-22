import unittest
import numpy as np
from our_mani.multitask.minimax_terminal_v220 import enclosing_center,fit_terminal


class Tests(unittest.TestCase):
    def test_extreme_terminal_is_retained_in_minimax_solution(self):
        c,a=enclosing_center([[0.,0.,0.],[1.,0.,0.],[10.,0.,0.]])
        np.testing.assert_allclose(c,[5.,0.,0.],atol=1e-6);self.assertAlmostEqual(a['radius_m'],5.,places=6)

    def test_duplicate_attempts_do_not_change_source_balanced_worst_case(self):
        base=[dict(source='a',terminal_event=True,position=[0.,0.,0.]),dict(source='b',terminal_event=True,position=[10.,0.,0.])]
        c,a=fit_terminal([base[0]]*20+[base[1]])
        np.testing.assert_allclose(c,[5.,0.,0.],atol=1e-6)
        self.assertEqual(a['independent_parents'],2);self.assertEqual(a['points'],21)

    def test_identical_terminal_rows_and_nonfinite_input(self):
        c,a=enclosing_center([[.1,.2,.3]]*10);np.testing.assert_allclose(c,[.1,.2,.3],rtol=0,atol=1e-15);self.assertEqual(a['radius_m'],0.)
        with self.assertRaises(ValueError):enclosing_center([[float('nan'),0,0]])


if __name__=='__main__':unittest.main()
