"""The convex solver preserves preview behavior and actual command limits."""
import unittest
from unittest.mock import patch
from tests import test_highspeed_predictive_tracking_20261111_v1 as base
from our_mani.highspeed_control_20261111_v1.qp_tracking import QPTrackingPlanner


class QPContracts(unittest.TestCase):
    def test_future_preview_and_reference_bounds_with_QP_solver(self):
        with patch.object(base,'PredictiveTrackingPlanner',QPTrackingPlanner):
            base.PredictiveContracts.test_future_reversal_changes_first_reference_without_changing_current_input(self)


if __name__=='__main__':unittest.main()
