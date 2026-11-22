"""V72 private FREE-phase adapter: numerical alignment dwell equality only.

Physical dwell remains100ms. No timestamp/dwell value is shifted or rounded;
99ms cannot pass. A fixed absolute1ns comparison allowance handles binary
50/100ms clock subtraction tails. All other gates and phases are unchanged.
The adapter is used only by the independently typed V72 FREE planner.
"""
from copy import deepcopy
import numpy as np
from ..local_mpc_20261102 import mpc as original

VERSION='FREE_alignment_dwell_100ms_absolute1ns_comparison_20261103_v72'
ALIGNMENT_DWELL_NUMERICAL_ATOL_SECONDS=1e-9
MPCConfig=original.MPCConfig
PHASES=original.PHASES
validate_candidate=original.validate_candidate
_geometry=original._geometry


def alignment_dwell_reached(dwell,required):
    """Scalar/batched predicate; do not change either operand or the clock."""
    values=np.asarray(dwell)
    if values.dtype.kind not in 'fiu' or not np.isfinite(values).all():
        raise ValueError('Finite non-boolean accumulated alignment dwell required')
    if isinstance(required,(bool,np.bool_)) or not np.isscalar(required) or not np.isfinite(required) or required<=0:
        raise ValueError('Original positive finite dwell threshold required')
    result=(values>=required)|(np.abs(values-required)<=ALIGNMENT_DWELL_NUMERICAL_ATOL_SECONDS)
    return bool(result) if result.ndim==0 else result


def update_phase(state, candidate, previous=None, config=None):
    """Pure measured-state update; identical timestamp cannot add dwell twice."""
    c = config or MPCConfig()
    phase = deepcopy(previous) if previous else dict(phase="approach", dwell=0., lost_hold=0., updated_at=None)
    if phase["phase"] not in PHASES:
        raise ValueError("unknown geometric phase")
    now = state["time"]
    last = phase.get("updated_at")
    if last is not None and now < last-1e-9:
        raise ValueError("phase feedback time moved backwards")
    elapsed = 0. if last is None else max(now-last, 0.)
    if last is not None and abs(now-last) <= 1e-9:
        return phase
    metrics = _geometry(state, candidate)
    old = phase["phase"]
    if state["held"]:
        phase.update(phase="hold", dwell=0., lost_hold=0.)
        if "lift_goal_position" not in phase:
            phase["lift_goal_position"] = state.get("lift_goal_position", state["object_position"]+np.array([0.,0.,c.lift_height])).copy()
            phase["hold_tcp_quaternion"] = state["tcp_quaternion"].copy()
    elif old == "hold":
        phase["lost_hold"] = float(phase.get("lost_hold", 0.))+elapsed
        if phase["lost_hold"] >= c.lost_hold_dwell:
            phase.update(phase="align", dwell=0.)
    elif old == "approach":
        good = metrics["pregrasp_distance"] <= c.approach_enter and metrics["angle"] <= c.angle_enter
        phase["dwell"] = float(phase.get("dwell",0.))+elapsed if good else 0.
        if good and phase["dwell"] >= c.approach_dwell:
            phase.update(phase="align", dwell=0.)
    elif old == "align":
        if phase.get("reopening", False):
            phase["dwell"] = 0.
            if state["opening"] >= c.reopening_aperture and state.get("requested_opening",state["opening"]) >= .5:
                phase["reopening"] = False
        elif metrics["pregrasp_distance"] > c.approach_exit and metrics["distance"] > c.align_exit:
            phase.update(phase="approach", dwell=0.)
        else:
            good = (metrics["distance"] <= c.align_enter and metrics["angle"] <= c.angle_enter
                    and metrics["relative_speed"] <= candidate["allowed_relative_speed"]
                    and state["opening"] >= candidate["required_opening"]-.05)
            phase["dwell"] = float(phase.get("dwell",0.))+elapsed if good else 0.
            lead = c.close_lead_seconds if c.close_lead_seconds is not None else c.opening_command_delay+abs(state["opening"]-candidate["opening"])/c.opening_rate
            if good and alignment_dwell_reached(phase["dwell"],c.align_dwell) and now >= candidate["deadline"]-lead:
                phase.update(phase="close", dwell=0., close_entered_at=now)
    elif old == "close":
        # Native grasp() attempts attachment only on an open->close edge.
        # Continuing closed cannot capture a body that arrives later.
        attempted = now >= float(phase.get("close_entered_at",last if last is not None else now))+c.dt-1e-8
        if (metrics["distance"] > c.align_exit or metrics["angle"] > c.angle_exit
                or (attempted and (state["opening"] < c.reopening_aperture
                                   or state.get("requested_opening",state["opening"]) < .5))):
            phase.update(phase="align", dwell=0., reopening=True)
    phase.update(updated_at=now, metrics=metrics, held_is_measured=state["held"])
    return phase
