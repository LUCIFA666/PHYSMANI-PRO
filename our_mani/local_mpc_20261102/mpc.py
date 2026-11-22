"""NumPy P1 geometric MPC, with explicit clocks and a bounded CEM budget.

The object model is constant world linear/angular velocity. The execution model
is an acceleration-limited response approximation, not Native physics, GRU
accuracy, contact certification or a learned task-value function. All virtual
rollouts are pure data operations. Only the caller may execute the first action.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import time

import numpy as np


PHASES = ("approach", "align", "close", "hold")


def _vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"finite {name}[{size}] required")
    return result.copy()


def quaternion(value):
    q = _vector(value, 4, "xyzw quaternion")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("zero quaternion is invalid")
    return q / norm


def quat_multiply(left, right):
    x, w = left[:3], left[3]
    y, v = right[:3], right[3]
    return np.r_[w*y + v*x + np.cross(x, y), w*v - np.dot(x, y)]


def quat_inverse(q):
    return np.r_[-q[:3], q[3]]


def quat_exp(rotvec):
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-10:
        return quaternion(np.r_[.5*np.asarray(rotvec), 1.])
    return np.r_[np.asarray(rotvec)*(np.sin(.5*angle)/angle), np.cos(.5*angle)]


def quat_log(q):
    q = quaternion(q)
    if q[3] < 0:
        q = -q
    sine = float(np.linalg.norm(q[:3]))
    if sine < 1e-10:
        return 2*q[:3]
    return q[:3] * (2*np.arctan2(sine, q[3])/sine)


def quat_matrix(q):
    x, y, z, w = quaternion(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _limited(vector, maximum):
    norm = float(np.linalg.norm(vector))
    return vector * min(1., maximum/max(norm, 1e-12))


@dataclass
class MPCConfig:
    dt: float = .1
    native_dt: float = .05
    horizon: int = 8
    candidates: int = 8
    iteration_samples: tuple = (88, 88, 80)
    knots: int = 2
    seed: int = 20261102
    elite_fraction: float = .25
    update_fraction: float = .5
    position_std: float = .035
    position_std_floor: float = .003
    rotation_std: float = .12
    rotation_std_floor: float = .015
    max_position_perturbation: float = .12
    max_rotation_perturbation: float = .5
    max_tcp_speed: float = .7
    max_tcp_acceleration: float = 4.
    max_angular_speed: float = 3.
    max_angular_acceleration: float = 20.
    position_gain: float = 7.
    rotation_gain: float = 7.
    opening_rate: float = 4.  # normalized aperture per second; P1 assumption
    opening_command_delay: float = 0.  # P0 actuator applies opening in its first tick
    close_lead_seconds: float | None = None
    position_scale: float = .03
    orientation_scale: float = .2
    approach_enter: float = .045
    approach_exit: float = .075
    align_enter: float = .025
    align_exit: float = .055
    angle_enter: float = .2
    angle_exit: float = .4
    approach_dwell: float = .1
    align_dwell: float = .1
    lost_hold_dwell: float = .1
    reopening_aperture: float = .9
    lift_height: float = .12
    switch_penalty: float = 1e4
    weights: dict = field(default_factory=lambda: dict(position=1., orientation=.2, relative_speed=.2,
        opening=.3, smoothness=.02, effort=.001, deadline=.2, lift=2., slip=2., constraint=1.))

    def __post_init__(self):
        if (self.dt != .1 or self.native_dt != .05 or self.horizon != 8 or self.candidates != 8
                or tuple(self.iteration_samples) != (88, 88, 80)):
            raise ValueError("P1 fixes 2 Native ticks, H8, K8 and exactly 88+88+80=256 sequences")
        if self.knots not in (2, 3) or not 0 < self.elite_fraction <= 1 or not 0 < self.update_fraction <= 1:
            raise ValueError("two/three knots and finite CEM fractions required")
        for name, value in vars(self).items():
            if name in ("weights", "iteration_samples", "close_lead_seconds"):
                continue
            if not np.isfinite(value) or value < 0:
                raise ValueError("nonnegative finite configuration required: "+name)
        if self.close_lead_seconds is not None and (not np.isfinite(self.close_lead_seconds) or self.close_lead_seconds < 0):
            raise ValueError("finite nonnegative close lead required")
        if min(self.position_scale, self.orientation_scale, self.opening_rate, self.position_std_floor,
               self.rotation_std_floor, self.max_tcp_speed, self.max_tcp_acceleration) <= 0:
            raise ValueError("physical scales and response bounds must be positive")
        if not all(np.isfinite(v) and v >= 0 for v in self.weights.values()):
            raise ValueError("nonnegative finite cost weights required")


def validate_state(state):
    result = {}
    for key in ("tcp_position", "tcp_velocity", "object_position", "object_linear_velocity", "object_angular_velocity"):
        result[key] = _vector(state[key], 3, key)
    for key in ("tcp_quaternion", "object_quaternion"):
        result[key] = quaternion(state[key])
    result["tcp_angular_velocity"] = _vector(state.get("tcp_angular_velocity", np.zeros(3)), 3, "tcp_angular_velocity")
    result["opening"], result["time"] = float(state["opening"]), float(state["time"])
    if not np.isfinite(result["opening"]) or not 0 <= result["opening"] <= 1 or not np.isfinite(result["time"]):
        raise ValueError("finite physical clock and normalized actual aperture required")
    if not isinstance(state["held"], (bool, np.bool_)):
        raise ValueError("held must be explicitly measured boolean, not a closing command")
    result["held"] = bool(state["held"])
    for key in ("lift_goal_position", "requested_opening"):
        if key in state:
            result[key] = _vector(state[key], 3, key) if key == "lift_goal_position" else float(state[key])
    if "actuator_state" in state:
        if not isinstance(state["actuator_state"],dict):
            raise ValueError("actuator_state must be the explicit actual executor state dictionary")
        result["actuator_state"]=deepcopy(state["actuator_state"])
    if state.get("phase_state") is not None:
        if not isinstance(state["phase_state"],dict) or state["phase_state"].get("phase") not in PHASES:
            raise ValueError("explicit measured-controller phase_state dictionary required")
        result["phase_state"]=deepcopy(state["phase_state"])
    return result


def validate_candidate(candidate):
    """opening/closing_opening are the CLOSED command (0), never jaw clearance.

    required_opening is normalized pre-grasp aperture; required_width_m is an
    optional geometric width for a caller's feasibility callback.
    """
    value = dict(candidate)
    if not isinstance(value.get("id"), (str, int)) or isinstance(value.get("id"), bool):
        raise ValueError("stable candidate id required")
    value["object_tcp_position"] = _vector(value["object_tcp_position"], 3, "object_tcp_position")
    value["object_tcp_quaternion"] = quaternion(value["object_tcp_quaternion"])
    direction = _vector(value["approach_direction"], 3, "object-local approach_direction")
    if np.linalg.norm(direction) < 1e-10:
        raise ValueError("nonzero approach direction required")
    value["approach_direction"] = direction/np.linalg.norm(direction)
    for key, default in (("deadline", None), ("pregrasp_offset", .06), ("opening", 0.), ("allowed_relative_speed", .08)):
        value[key] = float(value.get(key, default))
        if not np.isfinite(value[key]):
            raise ValueError("finite candidate "+key+" required")
    if value["pregrasp_offset"] < 0 or not 0 <= value["opening"] <= 1 or value["allowed_relative_speed"] <= 0:
        raise ValueError("invalid geometry/aperture/relative-speed bounds")
    value["closing_opening"]=float(value.get("closing_opening",value["opening"]))
    value["required_opening"]=float(value.get("required_opening",1.))
    if value["opening"]!=0. or value["closing_opening"]!=0. or not np.isfinite(value["required_opening"]) or not 0<=value["required_opening"]<=1:
        raise ValueError("candidate opening/closing_opening must mean closed command0; use required_opening for grasp clearance")
    if "required_width_m" in value and (not np.isfinite(value["required_width_m"]) or value["required_width_m"]<0):
        raise ValueError("finite nonnegative required_width_m required")
    return value


def predict_object(state, absolute_time):
    """P1 analytical free-body CV/Cω, with world-frame angular velocity."""
    dt = float(absolute_time)-float(state["time"])
    if dt < -1e-9:
        raise ValueError("object prediction cannot consume a future state to reconstruct the past")
    return (state["object_position"] + dt*state["object_linear_velocity"],
            quaternion(quat_multiply(quat_exp(dt*state["object_angular_velocity"]), state["object_quaternion"])))


def contact_target(state, candidate, absolute_time, *, pregrasp=False):
    """Return point pose and velocity at the exact requested physical time.

    r is object-local, so world point velocity is v + omega × (R @ r).
    """
    position, orientation = predict_object(state, absolute_time)
    offset = candidate["object_tcp_position"].copy()
    if pregrasp:
        offset += candidate["approach_direction"]*candidate["pregrasp_offset"]
    lever_world = quat_matrix(orientation) @ offset
    return dict(position=position+lever_world,
                quaternion=quaternion(quat_multiply(orientation, candidate["object_tcp_quaternion"])),
                linear_velocity=state["object_linear_velocity"]+np.cross(state["object_angular_velocity"], lever_world),
                angular_velocity=state["object_angular_velocity"].copy(), time=float(absolute_time))


def _geometry(state, candidate):
    grasp = contact_target(state, candidate, state["time"])
    pregrasp = contact_target(state, candidate, state["time"], pregrasp=True)
    return dict(distance=float(np.linalg.norm(state["tcp_position"]-grasp["position"])),
                pregrasp_distance=float(np.linalg.norm(state["tcp_position"]-pregrasp["position"])),
                angle=float(np.linalg.norm(quat_log(quat_multiply(grasp["quaternion"], quat_inverse(state["tcp_quaternion"]))))),
                relative_speed=float(np.linalg.norm(state["tcp_velocity"]-grasp["linear_velocity"])))


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
            if good and phase["dwell"] >= c.align_dwell and now >= candidate["deadline"]-lead:
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


class SimpleExecutionModel:
    """Acceleration-limited P1 robot response, with two 50ms microsteps.

    Closing does not manufacture attachment. An observed held=True object
    follows its measured TCP-relative transform until actual aperture reopens.
    A P2 replacement must implement the same side-effect-free step contract.
    """
    def __init__(self, config=None):
        self.config = config or MPCConfig()

    def step(self, state, action, h=None, dt=.1):
        c = self.config
        if dt != c.dt:
            raise ValueError("execution macrostep must remain exactly two 50ms ticks")
        s, hidden = deepcopy(state), deepcopy(h or {})
        s.pop("actuator_state",None)  # future actuator state is predicted in h, not fresh actual feedback
        s.pop("phase_state",None)  # rollout's separate phase copy owns hypothetical updates
        target = _vector(action["tcp_position"],3,"TCP target")
        target_q = quaternion(action["tcp_quaternion"])
        opening = float(action["opening"])
        if opening not in (0.,1.):
            raise ValueError("Native opening command is binary; actual aperture is continuous")
        # The real P0 interface commands a pose, without Cartesian feedforward.
        # Reference velocities are cost/diagnostic data, not actuator inputs.
        previous_opening = float(hidden.get("requested_opening",s.get("requested_opening",0. if s["held"] else s["opening"])))
        if s["held"] and "object_in_tcp" not in hidden:
            inv = quat_matrix(s["tcp_quaternion"]).T
            hidden["object_in_tcp"] = inv @ (s["object_position"]-s["tcp_position"])
            hidden["object_quaternion_tcp"] = quat_multiply(quat_inverse(s["tcp_quaternion"]),s["object_quaternion"])
        for micro in range(2):
            before = s["object_position"].copy()
            wanted = _limited(c.position_gain*(target-s["tcp_position"]),c.max_tcp_speed)
            next_v = s["tcp_velocity"]+_limited(wanted-s["tcp_velocity"],c.max_tcp_acceleration*c.native_dt)
            s["tcp_position"] += .5*(s["tcp_velocity"]+next_v)*c.native_dt
            s["tcp_velocity"] = next_v
            error = quat_log(quat_multiply(target_q,quat_inverse(s["tcp_quaternion"])))
            desired_w = _limited(c.rotation_gain*error,c.max_angular_speed)
            next_w = s["tcp_angular_velocity"]+_limited(desired_w-s["tcp_angular_velocity"],c.max_angular_acceleration*c.native_dt)
            delta = .5*(s["tcp_angular_velocity"]+next_w)*c.native_dt
            s["tcp_quaternion"] = quaternion(quat_multiply(quat_exp(delta),s["tcp_quaternion"]))
            s["tcp_angular_velocity"] = next_w
            requested = previous_opening if micro*c.native_dt < c.opening_command_delay-1e-9 else opening
            s["opening"] += float(np.clip(requested-s["opening"],-c.opening_rate*c.native_dt,c.opening_rate*c.native_dt))
            s["held"] = bool(s["held"] and requested < .5)
            if s["held"]:
                s["object_position"] = s["tcp_position"]+quat_matrix(s["tcp_quaternion"])@hidden["object_in_tcp"]
                s["object_quaternion"] = quaternion(quat_multiply(s["tcp_quaternion"],hidden["object_quaternion_tcp"]))
                s["object_linear_velocity"] = (s["object_position"]-before)/c.native_dt
                s["object_angular_velocity"] = next_w.copy()
            else:
                s["object_position"],s["object_quaternion"] = predict_object(s,s["time"]+c.native_dt)
                hidden.pop("object_in_tcp",None);hidden.pop("object_quaternion_tcp",None)
            s["time"] += c.native_dt
        hidden["requested_opening"] = opening
        hidden["tcp_angular_velocity"] = s["tcp_angular_velocity"].copy()
        s["requested_opening"] = opening
        return s,hidden,dict(model="P1_simple_response_not_Native_or_GRU",microsteps=2,attachment_inferred_from_closed_command=False)

    __call__ = step


def _opening_command(state,candidate,phase,c):
    if phase["phase"]=="hold":return candidate["closing_opening"]
    if phase["phase"]!="close":return 1.
    requested=state.get("requested_opening",state["opening"])
    if requested>=.5:
        metrics=_geometry(state,candidate)
        if (metrics["distance"]>c.align_enter or metrics["angle"]>c.angle_enter
                or metrics["relative_speed"]>candidate["allowed_relative_speed"]
                or state["opening"]<candidate["required_opening"]-.05):
            return 1.  # no one-shot Native grasp edge outside measured geometry
    return candidate["closing_opening"]


def _nominal_action(state,candidate,phase,perturb,c):
    target = contact_target(state,candidate,state["time"]+c.dt,pregrasp=phase["phase"]=="approach")
    if phase["phase"] == "hold":
        object_goal = phase.get("lift_goal_position",state.get("lift_goal_position",state["object_position"]))
        target["position"] = object_goal + quat_matrix(state["object_quaternion"])@candidate["object_tcp_position"]
        target["quaternion"] = phase.get("hold_tcp_quaternion",state["tcp_quaternion"])
        target["linear_velocity"],target["angular_velocity"] = np.zeros(3),np.zeros(3)
    return dict(tcp_position=target["position"]+perturb[:3],
                tcp_quaternion=quaternion(quat_multiply(quat_exp(perturb[3:]),target["quaternion"])),
                opening=_opening_command(state,candidate,phase,c),
                tcp_velocity_reference=target["linear_velocity"].copy(),
                angular_velocity_reference=target["angular_velocity"].copy(),dt=c.dt)


def stage_cost(state,action,next_state,candidate,phase,*,config=None,previous_action=None,constraint_cost=None):
    """Explicit nonnegative geometric terms at the action's endpoint time."""
    c=config or MPCConfig()
    target=contact_target(next_state,candidate,next_state["time"],pregrasp=phase["phase"]=="approach")
    position=np.linalg.norm(next_state["tcp_position"]-target["position"])/c.position_scale
    angle=np.linalg.norm(quat_log(quat_multiply(target["quaternion"],quat_inverse(next_state["tcp_quaternion"]))))/c.orientation_scale
    speed=np.linalg.norm(next_state["tcp_velocity"]-target["linear_velocity"])/candidate["allowed_relative_speed"]
    wanted_opening=0. if phase["phase"] in ("close","hold") else 1.
    terms=dict(position=position**2,orientation=angle**2,relative_speed=speed**2,
        opening=(next_state["opening"]-wanted_opening)**2,smoothness=0.,
        effort=float(np.dot(next_state["tcp_velocity"],next_state["tcp_velocity"])),deadline=0.,lift=0.,slip=0.,constraint=0.)
    late=max(next_state["time"]-candidate["deadline"],0.)
    if phase["phase"] != "hold":
        terms["deadline"]=(late/c.dt)**2*(1.+position**2)
    if previous_action is not None:
        terms["smoothness"]=float(np.sum(((action["tcp_position"]-previous_action["tcp_position"])/c.position_scale)**2))
    if phase["phase"] == "hold":
        goal=phase.get("lift_goal_position",next_state.get("lift_goal_position",next_state["object_position"]))
        terms["lift"]=float(np.sum(((next_state["object_position"]-goal)/c.position_scale)**2))
        terms["slip"]=0. if next_state["held"] else 25.
        terms["position"]=0.  # held object, so lift/object error is the useful term
        terms["relative_speed"]=0.
    if constraint_cost is not None:
        result=constraint_cost(state,action,next_state,candidate,next_state["time"])
        values=list(result.values()) if isinstance(result,dict) else [result]
        if any(np.isnan(value) or value < 0 for value in values):
            raise ValueError("constraint callback must return nonnegative costs; +inf is infeasible")
        terms["constraint"]=float(sum(values))
    total=np.inf if any(np.isposinf(value) for value in terms.values()) else sum(c.weights[key]*value for key,value in terms.items())
    return float(total),terms


class MPCPlanner:
    """K8 geometric modes share a total of 256 complete H8 rollouts."""
    def __init__(self,config=None,execution_model=None,constraint_cost=None):
        self.config=config if isinstance(config,MPCConfig) else MPCConfig(**(config or {}))
        self.execution_model=execution_model or SimpleExecutionModel(self.config)
        self.constraint_cost=constraint_cost

    def _rollout(self,state,candidate,controls,previous_phase,initial_h,committed_id,fixed_sequence=None):
        c=self.config
        s,h,phase=deepcopy(state),deepcopy(initial_h),deepcopy(previous_phase)
        actions,states,costs=[],[deepcopy(s)],[]
        terms={key:0. for key in c.weights}
        for step in range(c.horizon):
            phase=update_phase(s,candidate,phase,c)
            action=_nominal_action(s,candidate,phase,controls[step],c)
            if fixed_sequence is not None:
                action=deepcopy(fixed_sequence[step])
                action["dt"]=c.dt
                # A shifted close command cannot bypass current geometric
                # readiness or the explicit missed-grasp reopening state.
                action["opening"]=_opening_command(s,candidate,phase,c)
            fn=self.execution_model.step if hasattr(self.execution_model,"step") else self.execution_model
            nxt,h,diagnostic=fn(s,action,h,c.dt)
            nxt=validate_state(nxt)
            if abs(nxt["time"]-(state["time"]+(step+1)*c.dt))>1e-8:
                raise ValueError("execution model endpoint clock differs from the forecast/cost clock")
            value,parts=stage_cost(s,action,nxt,candidate,phase,config=c,
                                  previous_action=actions[-1] if actions else None,constraint_cost=self.constraint_cost)
            for key in terms:terms[key]+=parts[key]
            actions.append(action);states.append(deepcopy(nxt));costs.append(value);s=nxt
        switch=c.switch_penalty if previous_phase and previous_phase.get("phase") in ("close","hold") and committed_id is not None and candidate["id"]!=committed_id else 0.
        return dict(cost=float(sum(costs)+switch),cost_terms=dict(terms,switch=switch),sequence=actions,states=states,
                    candidate_id=candidate["id"],deadline=candidate["deadline"])

    def plan(self,state,candidates,*,warm_start=None):
        started=time.perf_counter();c=self.config
        state=validate_state(state);candidates=[validate_candidate(row) for row in candidates]
        if len(candidates)!=8 or len({row["id"] for row in candidates})!=8:
            raise ValueError("exactly eight distinct geometry/deadline candidates required")
        warm=deepcopy(warm_start or {})
        if warm and state["time"] < float(warm["time"])-1e-9:
            raise ValueError("warm start cannot come from a future feedback state")
        committed=warm.get("selected_candidate_id")
        if warm and committed not in {row["id"] for row in candidates}:
            raise ValueError("candidate generation expired: clear its warm start explicitly")
        for row in candidates:
            if row["id"]==committed and abs(row["deadline"]-warm["deadline"])>1e-8:
                raise ValueError("committed candidate deadline cannot be repeatedly pushed forward; expire it explicitly")
        old_phase=deepcopy(state.get("phase_state",warm.get("phase_state",dict(phase="approach",dwell=0.,lost_hold=0.,updated_at=None))))
        # Optimization warm state never becomes execution feedback. RML/joint
        # references must be supplied freshly by the real executor, and contact
        # transforms are rebuilt from the currently measured poses.
        initial_h=deepcopy(state.get("actuator_state",{}))
        initial_h["source"]="current_measured_state_and_actual_actuator_state"
        initial_h["requested_opening"]=float(state.get("requested_opening",initial_h.get("requested_opening",0. if state["held"] else state["opening"])))
        initial_h.pop("object_in_tcp",None);initial_h.pop("object_quaternion_tcp",None)
        if state["held"]:
            initial_h["object_in_tcp"]=quat_matrix(state["tcp_quaternion"]).T@(state["object_position"]-state["tcp_position"])
            initial_h["object_quaternion_tcp"]=quat_multiply(quat_inverse(state["tcp_quaternion"]),state["object_quaternion"])
        means=np.zeros((8,c.knots,6));std=np.ones_like(means)*np.r_[np.full(3,c.position_std),np.full(3,c.rotation_std)]
        floors=np.r_[np.full(3,c.position_std_floor),np.full(3,c.rotation_std_floor)]
        grid=np.linspace(0.,1.,c.horizon);knots=np.linspace(0.,1.,c.knots)
        shifted=deepcopy(warm.get("sequence",[]))
        if shifted:
            if len(shifted)!=8:raise ValueError("warm sequence must contain exactly H8 actions")
            shift=min(8,max(0,int(round((state["time"]-float(warm["time"]))/c.dt))))
            shifted=shifted[shift:]+[deepcopy(shifted[-1]) for _ in range(shift)]
        if "knot_means" in warm and list(warm.get("candidate_ids",[]))==[row["id"] for row in candidates]:
            old_means=np.asarray(warm["knot_means"],float);old_stds=np.asarray(warm["knot_stds"],float)
            if old_means.shape!=means.shape or old_stds.shape!=std.shape or not np.isfinite(old_means).all() or not np.isfinite(old_stds).all():
                raise ValueError("finite aligned warm control statistics required")
            shift=min(8,max(0,int(round((state["time"]-float(warm["time"]))/c.dt))))
            for mode in range(8):
                for axis in range(6):
                    dense=np.interp(grid,knots,old_means[mode,:,axis])
                    dense=np.r_[dense[shift:],np.repeat(dense[-1],shift)]
                    means[mode,:,axis]=np.interp(knots,grid,dense)
            std=np.maximum(old_stds,floors)
        if committed in [row["id"] for row in candidates] and shifted:
            index=next(i for i,row in enumerate(candidates) if row["id"]==committed)
            base_phase=update_phase(state,candidates[index],old_phase,c)
            nominal=_nominal_action(state,candidates[index],base_phase,np.zeros(6),c)
            delta=shifted[0]["tcp_position"]-nominal["tcp_position"]
            means[index,:,:3]=np.clip(delta,-c.max_position_perturbation,c.max_position_perturbation)
        rng=np.random.default_rng(c.seed+int(round(state["time"]/c.dt)))
        best=None;counts=[];evaluated=0;warm_evaluations=0
        for iteration,batch in enumerate(c.iteration_samples):
            per_mode=batch//8;mode_costs=[]
            for mode,candidate in enumerate(candidates):
                samples=rng.normal(means[mode],std[mode],size=(per_mode,c.knots,6))
                samples[0]=means[mode]  # geometric seed / shifted warm mean occupies the budget
                samples[...,:3]=np.clip(samples[...,:3],-c.max_position_perturbation,c.max_position_perturbation)
                samples[...,3:]=np.clip(samples[...,3:],-c.max_rotation_perturbation,c.max_rotation_perturbation)
                scores=[]
                reserved_warm=bool(shifted and candidate["id"]==committed)
                for sample_index,sample in enumerate(samples):
                    controls=np.stack([np.interp(grid,knots,sample[:,axis]) for axis in range(6)],-1)
                    rollout=self._rollout(state,candidate,controls,old_phase,initial_h,committed,
                                          fixed_sequence=shifted if reserved_warm and sample_index==0 else None)
                    evaluated+=1;scores.append(rollout["cost"])
                    if best is None or rollout["cost"]<best["cost"]:best=rollout
                if reserved_warm:warm_evaluations+=1
                fit=np.arange(1,per_mode) if reserved_warm else np.arange(per_mode)
                elite=fit[np.argsort(np.asarray(scores)[fit],kind="stable")[:max(2,int(np.ceil(per_mode*c.elite_fraction)))]];selected=samples[elite]
                a=c.update_fraction
                means[mode]=(1-a)*means[mode]+a*selected.mean(0)
                std[mode]=np.maximum(floors,(1-a)*std[mode]+a*selected.std(0))
                mode_costs.append(float(min(scores)))
            counts.append(dict(iteration=iteration,sequences=batch,per_candidate=per_mode,minimum_cost_by_candidate=mode_costs))
        if evaluated!=256:raise AssertionError("CEM budget invariant broken")
        if best is None or not np.isfinite(best["cost"]):
            return dict(action=None,sequence=[],selected_candidate_id=None,phase_state=old_phase,warm_start=None,
                        diagnostics=dict(reason="all_sequences_infeasible",iterations=counts,phase_real_updates=0,
                                         wall_seconds=time.perf_counter()-started,real_time_guaranteed=False,value_network_used=False),
                        budget=dict(sequences=evaluated,horizon=8,virtual_macrosteps=2048,virtual_50ms_substeps=4096,Native_steps=0))
        chosen=next(row for row in candidates if row["id"]==best["candidate_id"])
        actual_phase=update_phase(state,chosen,old_phase,c)  # the sole committed measured-state phase update
        warm_out=dict(time=state["time"],selected_candidate_id=chosen["id"],deadline=chosen["deadline"],
                      phase_state=deepcopy(actual_phase),sequence=deepcopy(best["sequence"]),
                      candidate_ids=[row["id"] for row in candidates],
                      knot_means=means.copy(),knot_stds=std.copy())
        elapsed=time.perf_counter()-started
        return dict(action=deepcopy(best["sequence"][0]),sequence=deepcopy(best["sequence"]),
                    selected_candidate_id=chosen["id"],phase_state=actual_phase,warm_start=warm_out,
                    cost=best["cost"],cost_terms=best["cost_terms"],predicted_states=best["states"],
                    diagnostics=dict(object_prediction="P1_analytical_constant_world_velocity_and_omega_not_GRU_accuracy",
                        execution_prediction="P1_simple_response_not_Native_or_learned_transition",iterations=counts,
                        endpoint_times=[state["time"]+(i+1)*c.dt for i in range(8)],deadline=chosen["deadline"],
                        warm_start_evaluations_counted=warm_evaluations,phase_real_updates=1,wall_seconds=elapsed,
                        execution_state_source="measured_state_and_explicit_actual_actuator_state",warm_prediction_used_as_feedback=False,
                        wall_time_below_100ms=elapsed<=.1,real_time_guaranteed=False,value_network_used=False),
                    budget=dict(sequences=evaluated,horizon=8,candidate_modes=8,iterations=3,
                                iteration_sequences=list(c.iteration_samples),virtual_macrosteps=2048,
                                virtual_50ms_substeps=4096,Native_steps=0))


def plan_mpc(state,candidates,*,config=None,warm_start=None,execution_model=None,constraint_cost=None):
    """Stateless convenience entry; the caller owns the measured feedback/warm state."""
    return MPCPlanner(config,execution_model,constraint_cost).plan(state,candidates,warm_start=warm_start)
