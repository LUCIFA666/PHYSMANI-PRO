"""Pure public motion-mode/command segmentation shared by actor and collector.

Inputs are previous visible body estimates, robot proprioceptive contact and
commands the controller issued. No native held IDs, teacher labels, task
phases, environment handles, success sensors or future truth enter this code.
"""
import numpy as np

MODE_IDS = {'external': 0, 'attached': 1, 'released': 2}


class PublicModeError(ValueError):
    pass


class PublicModeObserver:
    """Conservative association from past visible estimates and own contact.

    Ambiguous graspable identities invalidate their history rows instead of
    silently labeling a possibly held object as externally moving. A known
    body remains attached until own contact clears, then enters released mode.
    """
    def __init__(self, body_ids, graspable_ids, maximum_distance):
        self.body_ids, self.graspable_ids = tuple(body_ids), tuple(graspable_ids)
        self.maximum_distance = float(maximum_distance)
        self.held = None
        self.released = set()

    def observe(self, previous_states, robot):
        modes = {body: 'released' if body in self.released else 'external' for body in self.body_ids}
        invalid = set()
        if robot.contact:
            if self.held is None:
                candidates = [state for state in previous_states if state.object_id in self.graspable_ids
                    and state.visible and np.linalg.norm(state.pose.position-robot.pose.position) <= self.maximum_distance]
                if len(candidates) == 1:
                    self.held = candidates[0].object_id
                else:
                    invalid.update(self.graspable_ids)
            if self.held is not None:
                modes[self.held] = 'attached'
        elif self.held is not None:
            self.released.add(self.held)
            modes[self.held] = 'released'
            self.held = None
        return modes, invalid


AttachmentModeObserver = PublicModeObserver


class PublicCommandSegments:
    """Exclude action-controlled plate motion using issued commands only.

    Press is latched for the remainder of the episode. A later move/hold
    command cannot certify clearance without a TRAIN-validated public detector.
    """
    def __init__(self, body_ids):
        self.body_ids = tuple(body_ids)
        self.non_exogenous = set()

    def issue(self, kind):
        if kind not in ('move', 'joint_trajectory', 'hold', 'press', 'unspecified', 'no_action_terminal', 'budget_stop'):
            raise PublicModeError('unknown public controller command kind: '+str(kind))
        if kind == 'press' and 'plate' in self.body_ids:
            self.non_exogenous.add('plate')
        return np.asarray([body not in self.non_exogenous for body in self.body_ids], dtype=bool)
