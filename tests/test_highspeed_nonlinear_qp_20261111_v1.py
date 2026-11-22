"""Future FK relinearization keeps actual-reference limits and causal preview."""
import unittest
from unittest.mock import patch
from tests import test_highspeed_predictive_tracking_20261111_v1 as base
from our_mani.highspeed_control_20261111_v1.nonlinear_qp import NonlinearHybridTrackingPlanner,HybridTrackingConfig


class LocalNonlinearPlanner(NonlinearHybridTrackingPlanner):
    def __init__(self,*args,**kwargs):
        # The existing synthetic chain is at z=-.308; it is not the Native
        # Panda/table fixture. Production obtains workspace from its helper.
        super().__init__(*args,config=HybridTrackingConfig(local_mpc_radius=100.,terminal_velocity_zero=True,horizon_ticks=8),workspace_lower=(-2.,)*3,workspace_upper=(2.,)*3,**kwargs)


class NonlinearQPContracts(unittest.TestCase):
    def test_future_changes_first_command_while_native_reference_limits_hold(self):
        with patch.object(base,'PredictiveTrackingPlanner',LocalNonlinearPlanner):
            base.PredictiveContracts.test_future_reversal_changes_first_reference_without_changing_current_input(self)


if __name__=='__main__':unittest.main()
