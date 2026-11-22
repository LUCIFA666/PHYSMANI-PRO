"""Explicit new generations when every fixed contact deadline has passed.

The older Cartesian diagnostic allowed 400ms of late candidates. A current
GRU Forecast cannot query those past times, so that grace interval otherwise
leaves all eight modes infeasible. Each generation here keeps its deadlines;
retirement and replacement are recorded as an event with new identities.
"""
from copy import deepcopy
import numpy as np

from ..local_mpc_20261102.candidates import GeometricCandidates


class CurrentWindowCandidates(GeometricCandidates):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not np.isfinite(self.horizon) or self.horizon<=0:
            raise ValueError('finite positive forecast horizon required')
        self.last_time = None

    def update(self, state):
        now = float(state['time'])
        if not np.isfinite(now) or (self.last_time is not None and now<self.last_time-1e-9):
            raise ValueError('finite causal candidate observation time required')
        self.last_time = now
        retired = None
        if (self.candidates and not state.get('held',False)
                and now>max(c['deadline'] for c in self.candidates)+1e-9):
            retired = deepcopy(self.candidates)
            self.candidates = []
        result = super().update(state)
        if retired is not None:
            self.events[-1].update(reason='all_fixed_deadlines_expired_replace_generation_immediately',
                retired=[dict(id=c['id'],deadline=c['deadline']) for c in retired],
                deadline_grace_seconds=0.,old_deadlines_modified=False)
        for row in result:
            row['expiration_policy'] = 'explicit_new_generation_no_past_forecast_grace'
        return result
