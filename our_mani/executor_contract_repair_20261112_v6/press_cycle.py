"""Measured robot feedback rearms a failed press; no target/task state input."""
from dataclasses import dataclass

@dataclass(frozen=True)
class PressState:
    phase: str='idle'
    since: object=None

def plan(state,*,now,opening,goal_z,tcp_z):
    if opening==1.:
        return PressState(),0.
    if opening!=0.:raise ValueError('Binary opening required')
    phase=state.phase
    since=state.since
    if phase=='idle':phase='press';since=None
    if phase=='press':
        since=(now if since is None else since)if tcp_z<=goal_z+.02 else None
        if since is not None and now-since>=.25-1e-8:
            phase='raise';since=None
    elif phase=='raise':
        since=(now if since is None else since)if tcp_z>=goal_z+.065 else None
        if since is not None and now-since>=.1-1e-8:
            phase='press';since=None
    else:raise ValueError('Unknown contact phase')
    return PressState(phase,since),.08 if phase=='raise'else 0.
