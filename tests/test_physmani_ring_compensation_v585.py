"""Complete the CPU fixture's real-generator age interface; same safety checks.

The frozen580 fixture omitted generator.config. Production generators already
provide it. Keep that original failed fixture and its log for provenance.
"""
from types import SimpleNamespace as NS
import unittest
import test_physmani_ring_compensation_v580 as original


class Tests(original.Tests):
    def fixture(self):
        policy,frontend=super().fixture()
        frontend.actor.controller.generator.config=NS(maximum_observation_age=.15)
        return policy,frontend


if __name__=='__main__':unittest.main()
