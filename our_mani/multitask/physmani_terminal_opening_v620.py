"""Preserve original RML gripper request timing in terminal release policies.

An arm-path tick carrying the existing closed request cannot invent an open
edge. Once a terminal open request was actually dispatched, target motion
cannot manufacture another close edge in that same release/place phase.
"""
from copy import deepcopy
from .physmani_ring_release_v590 import AuthorRingReleaseIntents as RingBase
from .physmani_buzz_curve_v602 import BuzzCurveIntents as BuzzBase
from .physmani_remove_stages_v611 import RemoveAuthorIntents as RemoveBase


class OriginalTerminalOpening:
    def qualify_opening(self,requested,current_requested):
        goal=self.selected
        if goal is not None and goal['phase'] in ('release','place'):
            # Original RML asks for the new opening only on its final counted
            # gripper tick. Earlier ticks carry current_requested unchanged.
            if current_requested>=.5 or requested<.5:
                current=float(current_requested>=.5)
                return current,dict(time=float(self.frontend.state.robot.timestamp),phase=goal['phase'],
                    desired_opening=float(requested>=.5),current_requested_opening=float(current_requested),
                    qualified_opening=current,reason='preserve_actual_terminal_open_request' if current else
                    'await_original_RML_new_gripper_request',original_counted_gripper_timing_preserved=True,
                    repeated_terminal_close_edge_permitted=False,native_truth_read=False),None
        return super().qualify_opening(requested,current_requested)

    def receipt(self):return dict(super().receipt(),terminal_opening_interface=dict(
        schema='r10_original_RML_terminal_opening_request_timing_v620',
        no_early_open_invented_from_geometry=True,no_reclose_in_terminal_release_or_place=True,
        native_RML_gripper_tick_and_physical_close_gate_unchanged=True))


class AuthorRingReleaseIntents(OriginalTerminalOpening,RingBase):pass
class BuzzCurveIntents(OriginalTerminalOpening,BuzzBase):pass
class RemoveAuthorIntents(OriginalTerminalOpening,RemoveBase):pass


def binding(root):
    from pathlib import Path
    from .evaluation import file_sha
    names=('physmani_terminal_opening_v620.py','physmani_ring_release_v590.py',
           'physmani_buzz_curve_v602.py','physmani_remove_stages_v611.py','physmani_geometry_goals_v593.py')
    return dict(schema='r10_original_RML_terminal_opening_request_timing_v620',
        terminal_phases=['release','place'],new_open_requires_original_request=True,
        terminal_open_is_not_reclosed_from_pose_drift=True,original_gripper_cadence_unchanged=True,
        sources_sha256={f'our_mani/multitask/{n}':file_sha(Path(root)/'our_mani/multitask'/n) for n in names})
