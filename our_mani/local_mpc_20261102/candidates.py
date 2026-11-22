"""Deterministic P1 geometric handoff modes, with explicit absolute deadlines."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class GeometricCandidates:
    """Four parallel-jaw yaw directions and two contact times.

    A deadline stays committed across re-plans. If all committed deadlines are
    missed, an explicit expiry event creates new identities. This diagnostic
    prior uses the author's downward TCP convention, not learned candidates.
    """

    def __init__(self, *, horizon=.8, pregrasp_offset=.07, contact_offset=-.015):
        self.horizon = float(horizon)
        self.pregrasp_offset = float(pregrasp_offset)
        self.contact_offset = float(contact_offset)
        self.generation = -1
        self.candidates = []
        self.events = []

    def update(self, state):
        now = float(state["time"])
        expired = self.candidates and now > max(c["deadline"] for c in self.candidates) + .4
        if not self.candidates or (expired and not state.get("held", False)):
            reason = "initial_geometry" if not self.candidates else "all_contact_deadlines_missed"
            self.generation += 1
            self.events.append(dict(time=now, generation=self.generation, reason=reason))
            object_rotation = Rotation.from_quat(state["object_quaternion"])
            down = Rotation.from_quat([0., 1., 0., 0.])
            self.candidates = []
            for yaw_index, yaw in enumerate((0., np.pi / 2, np.pi, 3 * np.pi / 2)):
                world_rotation = Rotation.from_rotvec([0., 0., yaw]) * down
                local_rotation = object_rotation.inv() * world_rotation
                for time_index, delay in enumerate((self.horizon * .625, self.horizon)):
                    self.candidates.append(dict(
                        id=f"geometry_g{self.generation}_yaw{yaw_index}_time{time_index}",
                        object_tcp_position=[0., 0., self.contact_offset],
                        object_tcp_quaternion=local_rotation.as_quat().tolist(),
                        deadline=now + delay,
                        approach_direction=[0., 0., 1.],
                        pregrasp_offset=self.pregrasp_offset,
                        opening=0.,
                        required_opening=1.,
                        required_width_m=.08,
                        allowed_relative_speed=.14,
                        provenance="P1 geometric downward parallel-jaw modes; not a Transformer output"))
        return [dict(c) for c in self.candidates]

