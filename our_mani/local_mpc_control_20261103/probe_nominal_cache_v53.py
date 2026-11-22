"""Fixed-probe nominal IK branch cache, with fresh feedback and qualifications.

Only a command destination is cached. Neither actual/virtual execution state,
actuator h, collision results, nor a path certificate enters the cache. The
unchanged VerificationProbePlanner still optimizes 256 H8 sequences and checks
both predicted 50 ms endpoints. This changes the nominal IK branch reuse; it
does not claim trajectory equivalence, contact inference or physical accuracy.

``nominal_mask`` is explicit public candidate/representative availability,
before Native collision qualification. A current collision failure masks the
cached branch; it does not silently start another IK branch-recovery policy.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import time

import numpy as np

from .public_grasp_probe import VerificationProbeConfig, VerificationProbePlanner
from .joint_goal_cem import norm, qlog, qmul, qinv
from ..local_mpc_p2_20261103.action_adapter import matrix_quaternion
from ..local_mpc_p2_20261103.execution_models import validate_feedback_state


SCHEMA = "fixed_probe_nominal_joint_cache_v53_20261103_v1"


def _plain(value):
    if is_dataclass(value):return _plain(asdict(value))
    if isinstance(value,np.ndarray):return _plain(value.tolist())
    if isinstance(value,np.generic):return _plain(value.item())
    if isinstance(value,dict):return {str(key):_plain(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(item) for item in value]
    if isinstance(value,float) and not np.isfinite(value):raise ValueError("nonfinite identity/feedback value")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _vector(value,size,name):
    result=np.asarray(value,dtype=np.float64)
    if result.shape!=(size,) or not np.isfinite(result).all():raise ValueError("finite "+name+" required")
    return result.copy()


def _source(value):
    if not isinstance(value,dict) or not value:raise ValueError("explicit frozen source/config/model SHA map required")
    for key,digest in value.items():
        if (not isinstance(key,str) or not key or not isinstance(digest,str) or len(digest)!=64
                or any(ch not in "0123456789abcdef" for ch in digest)):
            raise ValueError("canonical source names and lowercase SHA256 values required")
    return dict(sorted(value.items()))


def _mask(candidates,nominal_mask):
    if len(candidates)!=8 or len({c["id"] for c in candidates})!=8:raise ValueError("K8 distinct candidate ledger required")
    value=np.asarray(nominal_mask)
    if value.shape!=(8,) or value.dtype!=np.bool_:raise ValueError("explicit boolean nominal_mask[8] required")
    # An undefined public symmetry representative cannot be made valid by the
    # caller's mask. No previous native_pose_qualified flag is consulted.
    defined=np.asarray(["parallel_gripper_symmetry_choice" not in c or
        c["parallel_gripper_symmetry_choice"] is not None for c in candidates],bool)
    return value.copy()&defined


def _spec(spec):
    if not isinstance(spec,dict) or not isinstance(spec.get("identity"),str) or not spec["identity"]:
        raise ValueError("fixed admitted probe identity required")
    config=VerificationProbeConfig(**spec["config"])
    if asdict(config)!=asdict(VerificationProbeConfig()):
        raise ValueError("v53 preserves the original 12mm/15mm/.04rad probe cost and bounds")
    position=_vector(spec["target_position"],3,"fixed target position")
    quaternion=_vector(spec["target_quaternion"],4,"fixed target quaternion")
    anchor=_vector(spec["anchor_position"],3,"fixed actual trigger anchor")
    anchor_q=_vector(spec["anchor_quaternion"],4,"fixed trigger quaternion")
    if abs(norm(quaternion)-1)>1e-5 or abs(norm(anchor_q)-1)>1e-5:
        raise ValueError("normalized probe quaternions required")
    if not np.allclose(position-anchor,[0.,0.,.012],atol=1e-12,rtol=0) or not np.array_equal(quaternion,anchor_q):
        raise ValueError("fixed admission TCP +12mm world-Z and unchanged attitude required")
    if isinstance(spec["trigger_time"],(bool,np.bool_)):raise ValueError("numeric absolute probe time required")
    trigger=float(spec["trigger_time"])
    if not np.isfinite(trigger) or trigger<0:raise ValueError("finite absolute probe trigger time required")
    result=dict(identity=spec["identity"],attempt_id=spec.get("attempt_id"),trigger_time=trigger,
        target_position=position,target_quaternion=quaternion,anchor_position=anchor,
        anchor_quaternion=anchor_q,config=asdict(config))
    if "deadline" in spec:
        if isinstance(spec["deadline"],(bool,np.bool_)):raise ValueError("numeric absolute probe deadline required")
        deadline=float(spec["deadline"])
        if not np.isfinite(deadline) or deadline<=trigger:raise ValueError("fixed future probe deadline required")
        result["deadline"]=deadline  # Preserve; never extend or reset a budget.
    return result


def _fresh_feedback(actual):
    state=validate_feedback_state(actual)
    if len(state.get("q",[]))!=7 or "finger_open_amounts" not in state:
        raise ValueError("actual q/dq and independently measured two finger openings required")
    h=state.get("actuator_state")
    if not isinstance(h,dict):raise ValueError("fresh actual actuator state required")
    clock=h.get("reference_clock")
    if (isinstance(clock,(bool,np.bool_)) or not isinstance(clock,(float,int,np.floating,np.integer))
            or not np.isfinite(clock) or abs(float(clock)-state["time"])>1e-8):
        raise ValueError("fresh same-time actual actuator h.reference_clock required")
    if ("predicted" in h and (not isinstance(h["predicted"],(bool,np.bool_)) or bool(h["predicted"]))) or "virtual" in str(h.get("source","")).lower():
        raise ValueError("virtual execution h must not become current feedback")
    grip=h.get("gripper_target")
    if (isinstance(grip,(bool,np.bool_)) or not isinstance(grip,(float,int,np.floating,np.integer))
            or not np.isfinite(grip) or state["held"] or state.get("requested_opening")!=0. or grip!=0.):
        raise ValueError("probe requires unknown/unheld state with actual requested close target0")
    return state


def _identity(candidates,spec,helper,source_sha256,nominal_mask):
    spec=_spec(spec);mask=_mask(candidates,nominal_mask)
    limits={key:np.asarray(getattr(helper.limits,key),dtype=np.float64).copy() for key in
        ("lower","upper","velocity","acceleration","jerk","reference_velocity_cap")}
    if any(not np.isfinite(value).all() for value in limits.values()):raise ValueError("finite current helper limits required")
    chain=helper.kinematics
    value=dict(schema=SCHEMA,probe=spec,candidate_ids=[c["id"] for c in candidates],
        candidate_deadlines=[float(c["deadline"]) for c in candidates],nominal_mask=mask,
        limits=limits,adapter_lower=np.asarray(helper.adapter.lower),adapter_upper=np.asarray(helper.adapter.upper),
        helper_config=asdict(helper.config),DLS_config=asdict(helper.adapter.config),
        PoE=dict(q_ref=chain.q_ref,screws=chain.screws,reference_tcp=chain.reference_tcp),
        source_sha256=_source(source_sha256))
    return value,_digest(value),mask


class ProbeNominalQCache:
    """One immutable destination entry, not a state estimator or certificate.

    First call uses the current actual q as DLS seed. Identical fixed identity
    reuses that exact float64 q destination. Changing spec/attempt/pose/deadline,
    candidate IDs, public availability mask, limits, PoE, DLS or source/config
    SHA invalidates it and solves from the new current actual q. ``reset`` is
    appropriate at episode/attempt disposal. Collision outputs are never stored.
    """
    def __init__(self):
        self._entry=None;self.solve_count=0;self.reuse_count=0

    def reset(self):
        self._entry=None

    def resolve(self,actual_q,candidates,spec,helper,*,source_sha256,nominal_mask):
        q=_vector(actual_q,7,"current actual q seed")
        identity,key,mask=_identity(candidates,spec,helper,source_sha256,nominal_mask)
        reused=self._entry is not None and self._entry["cache_key"]==key
        if not reused:
            p=identity["probe"]["target_position"];r=identity["probe"]["target_quaternion"]
            result=helper.adapter.solve(np.broadcast_to(q,(8,7)).copy(),
                np.broadcast_to(p,(8,3)).copy(),np.broadcast_to(r,(8,4)).copy())
            goals=np.asarray(result["joint_goal"],dtype=np.float64)
            feasible=np.asarray(result["joint_goal_feasible"])
            if goals.shape!=(8,7) or not np.isfinite(goals).all() or feasible.shape!=(8,) or feasible.dtype!=np.bool_:
                raise ValueError("aligned finite DLS target q and boolean feasibility required")
            self._entry=dict(cache_key=key,identity=deepcopy(identity),joint_goal=goals.copy(),
                algebraic_feasible=feasible.copy(),pose=np.c_[np.broadcast_to(p,(8,3)),np.broadcast_to(r,(8,4))].copy())
            self.solve_count+=1
        else:self.reuse_count+=1
        return deepcopy(self._entry),dict(schema=SCHEMA,cache_key=key,reused=reused,
            DLS_calls=0 if reused else 1,poses=8,solve_count=self.solve_count,reuse_count=self.reuse_count,
            cached_destination_only=True,actual_or_virtual_state_cached=False,actuator_h_cached=False,
            collision_certificate_cached=False,source_identity_sha256=_digest(identity["source_sha256"]),
            fixed_target_joint_goal_sha256=_digest(self._entry["joint_goal"]))

    def lookup(self,key):
        if self._entry is None or self._entry["cache_key"]!=key:raise ValueError("current nominal cache identity is unavailable")
        return deepcopy(self._entry)


@dataclass(frozen=True)
class CachedProbeQualification:
    """Ephemeral current-frame collision evidence; deliberately outside cache."""
    cache_key:str
    feedback_sha256:str
    joint_goal_sha256:str
    candidate_ids:tuple
    qualified:tuple
    collision_checked:tuple
    cases:tuple
    cache_diagnostic:dict

    def receipt(self):
        return dict(schema="cached_probe_current_frame_qualification_v53_20261103_v1",
            cache_key=self.cache_key,feedback_sha256=self.feedback_sha256,
            joint_goal_sha256=self.joint_goal_sha256,cases=deepcopy(self.cases),cache=deepcopy(self.cache_diagnostic),
            batch_DLS_calls=self.cache_diagnostic["DLS_calls"],Native_IK_calls=0,physical_steps=0,
            cached_branch_checked_now=True,old_fresh_DLS_certificate_reused=False,
            selected_action_and_current_path_recheck_still_required=True,
            whole_path_certified=False,formal_qualification=False)


def qualify_cached_probe_candidates(backend,state,candidates,helper,spec,*,nominal_cache,source_sha256,nominal_mask):
    """Resolve/reuse a branch, then recheck that branch's K8 static endpoints.

    This never calls Native IK or steps. ``backend.probe_joint_collision`` must
    be the existing independently audited no-step collision oracle. A previous
    fresh-DLS candidate mask/certificate is ignored, not transferred to cached q.
    """
    if type(nominal_cache) is not ProbeNominalQCache:raise TypeError("explicit independent nominal cache required")
    actual=_fresh_feedback(state)
    entry,diagnostic=nominal_cache.resolve(actual["q"],candidates,spec,helper,
        source_sha256=source_sha256,nominal_mask=nominal_mask)
    mask=np.asarray(entry["identity"]["nominal_mask"],bool)
    rows=[];cases=[];checked=[];qualified=[]
    for index,candidate in enumerate(candidates):
        goal=entry["joint_goal"][index]
        # Re-evaluate algebraic FK/bounds against today's helper and actual q;
        # never promote a failed original solve or a stale fresh-DLS branch.
        action=dict(tcp_position=entry["pose"][index,:3],tcp_quaternion=entry["pose"][index,3:],
                    joint_goal=goal,joint_goal_feasible=bool(entry["algebraic_feasible"][index]))
        _,reason,fk_receipt=helper.solve_action_target(actual,action,require_predicted_joint_goal=True)
        algebraic=bool(mask[index] and reason is None)
        collision=None
        if algebraic:
            collision=backend.probe_joint_collision(goal.copy())
            if not isinstance(collision,(bool,np.bool_)):raise TypeError("explicit boolean no-step collision result required")
            collision=bool(collision)
        valid=bool(algebraic and not collision)
        row=deepcopy(candidate);row.update(native_pose_qualified=valid,qualification_joint_goal=goal.copy(),
            joint_goal_qualification_method="cached_nominal_q_fresh_current_static_collision_v53")
        rows.append(row);checked.append(algebraic);qualified.append(valid)
        cases.append(dict(id=candidate["id"],deadline=candidate["deadline"],deadline_shifted=False,
            qualified=valid,static_collision=collision,collision_checked_current_frame=algebraic,
            public_nominal_mask=bool(mask[index]),algebraic_reason=reason,
            joint_goal=goal.copy(),target_position=entry["pose"][index,:3].copy(),
            target_quaternion=entry["pose"][index,3:].copy(),FK=deepcopy(fk_receipt),
            target_source="fixed_probe_cached_nominal_q",Native_IK_calls=0,physical_steps=0))
    receipt=CachedProbeQualification(entry["cache_key"],_digest(actual),_digest(entry["joint_goal"]),
        tuple(c["id"] for c in candidates),tuple(qualified),tuple(checked),tuple(cases),diagnostic)
    return rows,receipt


def _assert_qualification(actual,candidates,spec,helper,cache,source_sha256,nominal_mask,qualification):
    if type(qualification) is not CachedProbeQualification:
        raise TypeError("current typed cached-branch qualification required; serialized/old DLS evidence is insufficient")
    if (len(qualification.qualified)!=8 or len(qualification.collision_checked)!=8
            or any(type(x) is not bool for x in qualification.qualified+qualification.collision_checked)
            or any(ok and not checked for ok,checked in zip(qualification.qualified,qualification.collision_checked))):
        raise ValueError("each qualified mode needs an explicit current collision check")
    identity,key,_=_identity(candidates,spec,helper,source_sha256,nominal_mask)
    entry=cache.lookup(key)
    if (qualification.cache_key!=key or qualification.feedback_sha256!=_digest(actual)
            or qualification.joint_goal_sha256!=_digest(entry["joint_goal"])
            or qualification.candidate_ids!=tuple(c["id"] for c in candidates)
            or qualification.qualified!=tuple(c.get("native_pose_qualified") for c in candidates)):
        raise ValueError("fresh actual state+h and exact current cached-branch qualification must agree")
    return entry


class CachedNominalVerificationProbePlanner(VerificationProbePlanner):
    """Unchanged probe CEM/cost/model; only _prepare_nominals uses cached q.

    Create after ``qualify_cached_probe_candidates`` at the current decision.
    A new planner may share the cache across decisions. Warm proposals remain
    optimizer proposals only and are cleared when this nominal identity changes.
    """
    def __init__(self,*args,nominal_cache,source_sha256,nominal_mask,qualification,**kwargs):
        super().__init__(*args,**kwargs)
        if type(nominal_cache) is not ProbeNominalQCache:raise TypeError("explicit independent nominal cache required")
        self.nominal_cache=nominal_cache;self.source_sha256=_source(source_sha256)
        self.nominal_mask=np.asarray(nominal_mask).copy();self.qualification=qualification
        if (self.config.dt!=.1 or self.config.native_dt!=.05 or tuple(self.config.iteration_samples)!=(88,88,80)):
            raise ValueError("unchanged 100ms/50ms 256 H8 probe CEM required")

    def _prepare_nominals(self,initial,candidates,old):
        start=time.perf_counter();_fresh_feedback(initial)
        # JointGoalCEMPlanner already validated this exact raw feedback once,
        # just as the qualification helper did. A second quaternion unitization
        # can differ by 1 ulp; retain that once-validated state for the hash.
        entry=_assert_qualification(initial,candidates,self.probe_spec,self.goal_helper,self.nominal_cache,
            self.source_sha256,self.nominal_mask,self.qualification)
        feasible=entry["algebraic_feasible"]&np.asarray(entry["identity"]["nominal_mask"],bool)
        self._nominals=dict(joint_goal=np.broadcast_to(entry["joint_goal"],(3,8,7)).copy(),
            feasible=np.broadcast_to(feasible,(3,8)).copy(),pose=np.broadcast_to(entry["pose"],(3,8,7)).copy(),
            time=np.full((3,8),self.probe_spec["trigger_time"]))
        self.nominal_receipt=dict(DLS_calls=0,poses=8,wall_seconds=time.perf_counter()-start,
            basis_version=SCHEMA,probe=deepcopy(self.probe_spec),cache=deepcopy(self.qualification.cache_diagnostic),
            original_candidate_ids_and_deadlines_preserved=True,distinct_probe_pose_modes=False,
            qualification_resolved_DLS_calls=self.qualification.cache_diagnostic["DLS_calls"],
            same_current_actual_state_and_h_used=True,collision_certificate_reused=False)

    def plan(self,state,candidates,*,warm_start=None):
        actual=_fresh_feedback(state)
        entry=_assert_qualification(actual,candidates,self.probe_spec,self.goal_helper,self.nominal_cache,
            self.source_sha256,self.nominal_mask,self.qualification)
        reason=None;warm=warm_start
        if warm and (warm.get("probe_nominal_cache_identity")!=entry["cache_key"] or
                     warm.get("verification_probe_identity")!=self.probe_spec["identity"]):
            reason="incompatible_probe_or_fixed_nominal_cache_identity";warm=None
        result=super().plan(state,candidates,warm_start=warm)
        if result.get("warm_start") is not None:result["warm_start"]["probe_nominal_cache_identity"]=entry["cache_key"]
        result["diagnostics"].update(nominal_joint_cache_v53=True,cache_key=entry["cache_key"],
            incompatible_warm_cleared=reason,actual_state_and_h_reused=False,
            cached_branch_collision_checked_current_frame=True,
            selected_action_point_and_current_sampled_path_recheck_required=True,
            nominal_parameterization_changed_not_trajectory_equivalence=True,
            budget_or_probe_threshold_changed=False)
        return result


def qualify_cached_probe_action(backend,state,action,helper,*,candidates,probe_spec,nominal_cache,
        source_sha256,nominal_mask,qualification,selected_candidate_id=None):
    """Exact selected q reuse plus the original endpoint + seven path probes.

    No second IK solve, target clipping, gripper override or physical stepping.
    The sampled Native collision oracle is not continuous swept-volume proof.
    """
    actual=_fresh_feedback(state)
    _assert_qualification(actual,candidates,probe_spec,helper,nominal_cache,source_sha256,nominal_mask,qualification)
    selected=selected_candidate_id if selected_candidate_id is not None else action.get("candidate_id")
    if selected not in qualification.candidate_ids or not qualification.qualified[qualification.candidate_ids.index(selected)]:
        return None,"selected_cached_probe_mode_not_currently_qualified",dict(physical_steps=0,Native_IK_calls=0)
    if action.get("opening")!=0.:
        return None,"probe_selected_action_is_not_original_close_command",dict(physical_steps=0,Native_IK_calls=0)
    q=_vector(action.get("joint_goal"),7,"selected modeled joint goal")
    pose=helper.kinematics(q);p=pose[:3,3];quaternion=matrix_quaternion(pose[:3,:3]);spec=_spec(probe_spec)
    angle=float(norm(qlog(qmul(quaternion,qinv(spec["anchor_quaternion"])))))
    if norm(p-spec["anchor_position"])>.015+1e-10 or angle>.04+1e-10:
        return None,"selected_probe_command_outside_original_anchor_bounds",dict(physical_steps=0,Native_IK_calls=0)
    command_position=_vector(action.get("tcp_position"),3,"selected command position")
    command_quaternion=_vector(action.get("tcp_quaternion"),4,"selected command quaternion")
    if (norm(command_quaternion)<1e-12 or not np.allclose(command_position,p,rtol=0,atol=1e-12) or
            norm(qlog(qmul(command_quaternion,qinv(quaternion))))>1e-12):
        return None,"selected_probe_pose_does_not_match_exact_modeled_joint_goal_FK",dict(physical_steps=0,Native_IK_calls=0)
    goal,reason,receipt=helper.qualify_action(backend,actual,action,require_predicted_joint_goal=True)
    receipt.update(cache_key=qualification.cache_key,current_cached_branch_qualification_verified=True,
        endpoint_and_seven_path_samples_rechecked_now=True,selected_action_postprocessed=False,
        original_opening_command=action["opening"],physical_safety_certificate=False)
    return goal,reason,receipt
