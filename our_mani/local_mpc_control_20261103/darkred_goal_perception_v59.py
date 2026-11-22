"""V59 fixed dark-red public tracker: one declared color-prior hypothesis.

Only minimum_red changes from 170 to 100 for the entire episode. Geometry,
confidence, centre formula, causal history and initial goal-surface filtering
are inherited unchanged. Actual RGB values are never brightened or replaced.
This development variant has no new Native/task/precision qualification.
"""
from dataclasses import asdict
from ..benchmark_perception import BoxPrior,RedBoxTracker
from .goal_aware_perception import GoalAwareRedBoxTracker

MINIMUM_RED=100


def darkred_box_prior():
    return BoxPrior(minimum_red=MINIMUM_RED)


class DarkRedGoalAwareTrackerV59(GoalAwareRedBoxTracker):
    def __init__(self):
        super().__init__(tracker=RedBoxTracker(prior=darkred_box_prior()))

    def observe(self,observation):
        target,metadata=super().observe(observation)
        metadata.update(box_prior_unchanged=False,box_geometry_prior_unchanged=True,
            color_prior_changed=True,minimum_red=MINIMUM_RED,original_minimum_red=170,
            color_prior=asdict(self.prior),color_variant='fixed_darkred_v59_all_phases',
            phase_dependent_color_switch_used=False,raw_RGB_brightness_synthesized=False,
            preclose_bitexact_with_original_R170=False,
            color_change_is_development_hypothesis_not_runtime_qualification=True)
        return target,metadata
