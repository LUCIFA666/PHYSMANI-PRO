"""Candidate conditional RGBD orientation tracking, with an initial-only prior.

The physical CAD is asymmetric: no finite symmetry or rigid-pair permission
is claimed. The original initialization chooses a branch of the actual RGBD
registration at t=0. Later observations use only causal continuity and current
RGBD residuals; the pair prior is never reused after a robot action.
"""
from copy import deepcopy
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from ..contracts import Pose
from .two_center_pose import PublicCenter,estimate_two_center_pose

SCHEMA='r10_candidate_initial_prior_conditioned_RGBD_branches_v63'


def admissible_hypotheses(poses,rms,reference_rotation,*,rotation_bound_rad,rms_slack_m,rotation_ambiguity_rad):
    if len(poses)!=len(rms) or not poses:raise ValueError('complete current registration hypotheses required')
    if not 0<=rotation_bound_rad<np.pi/2:raise ValueError('branch prior must stay narrower than a half-turn ambiguity')
    rms=np.asarray(rms,float)
    prior_admissible=[i for i in range(len(poses)) if Rotation.from_matrix(poses[i].rotation@reference_rotation.T).magnitude()<=rotation_bound_rad]
    # Conditional MAP: evaluate relative likelihood within the admissible
    # prior support. Absolute original RMS/inlier gates are still mandatory.
    allowed=([i for i in prior_admissible if rms[i]<=min(rms[j] for j in prior_admissible)+rms_slack_m]
        if prior_admissible else [])
    if not allowed:return None,dict(admissible_indices=[],all_retained_hypotheses=len(poses),reason='no_current_RGBD_hypothesis_matches_conditional_prior')
    selected=min(allowed,key=lambda i:(rms[i],i))
    ambiguous=any(Rotation.from_matrix(poses[i].rotation@poses[selected].rotation.T).magnitude()>rotation_ambiguity_rad for i in allowed)
    return selected,dict(admissible_indices=allowed,all_retained_hypotheses=len(poses),rotation_ambiguous=ambiguous,
        reason='conditional_RGBD_cluster_ambiguous' if ambiguous else 'one_conditional_current_RGBD_orientation_cluster',
        rotation_bound_rad=rotation_bound_rad,rms_slack_m=rms_slack_m,
        prior_admissible_indices=prior_admissible,relative_residual_basis='best_prior_admissible_current_RGBD_candidate',
        absolute_registration_quality_gate_unchanged=True)


class PriorConditionedRegistration:
    def __init__(self,tracker,prior,*,maximum_angular_speed=2.,maximum_age=.1):
        if set(tracker.templates)!={'frame','wand'}:raise ValueError('exact buzz role inventory required')
        if any(t.orientation_observability!='full' for t in tracker.templates.values()):raise ValueError('explicit candidate conditional full-orientation protocol required')
        self.tracker=tracker;self.prior=prior;self.original=tracker._registration
        self.maximum_angular_speed=float(maximum_angular_speed);self.maximum_age=float(maximum_age)
        self.previous={};self.events=[];self.initial_rotations=None;self.context=None;self.raw_initial={}

    def set_public_context(self,*,timestamp,gripper_open,contact):
        self.context=dict(timestamp=float(timestamp),gripper_open=bool(gripper_open),contact=bool(contact))

    def initialize(self,clouds):
        c=self.context;now=float(self.tracker._current_timestamp)
        if c is None or abs(c['timestamp']-now)>1e-9 or now!=0. or not c['gripper_open'] or c['contact']:return False
        centers={}
        for role in ('frame','wand'):
            fit=self.original({role:Pose(np.zeros(3))},clouds,None)
            if fit is None or not fit['valid']:return False
            self.raw_initial[role]=fit
            centers[role]=PublicCenter(fit['pose'].position,True,True,'external',now)
        estimate=estimate_two_center_pose(self.prior,centers,current_timestamp=now)
        if not estimate.valid:return False
        self.initial_rotations={role:getattr(estimate,role+'_rotation') for role in centers}
        return True

    def register(self,members,clouds,previous=None):
        if len(members)!=1 or next(iter(members)) not in ('frame','wand'):raise ValueError('no cross-body rigid registration permitted')
        role,local=next(iter(members.items()))
        if np.linalg.norm(local.position)>1e-12 or np.linalg.norm(local.rotation-np.eye(3))>1e-12:raise ValueError('native TRAIN role-root frame required')
        now=float(self.tracker._current_timestamp)
        if self.initial_rotations is None and not self.initialize(clouds):
            self.events.append(dict(time=now,role=role,qualified=False,reason='initial_public_branch_prior_unavailable'));return None
        fit=self.raw_initial.pop(role,None) if now==0. else None
        if fit is None:fit=self.original(members,clouds,previous)
        if fit is None or not fit['valid']:return None
        if now==0.:reference=self.initial_rotations[role];bound=.15;basis='original_TRAIN_initialization_only'
        else:
            prior=self.previous.get(role)
            if prior is None or not 0<now-prior[0]<=self.maximum_age+1e-9:
                self.events.append(dict(time=now,role=role,qualified=False,reason='causal_orientation_history_stale'));return None
            reference=prior[1].rotation;bound=self.maximum_angular_speed*(now-prior[0])+.15;basis='previous_qualified_public_RGBD_orientation'
        index,evidence=admissible_hypotheses(fit['retained_root_candidate_poses'],fit['retained_candidate_rms_m'],reference,
            rotation_bound_rad=bound,rms_slack_m=self.tracker.config.ambiguity_rms_slack_m,
            rotation_ambiguity_rad=self.tracker.config.rotation_ambiguity_radians)
        row=dict(time=now,role=role,prior_basis=basis,qualified=False,evidence=evidence,
            physical_CAD_symmetry_claimed=False,native_dynamic_labels_used=False)
        if index is None or evidence['rotation_ambiguous']:self.events.append(row);return None
        selected=fit['retained_root_candidate_poses'][index];distances=[]
        for component in self.tracker.templates[role].components:
            points=clouds[role][component.name]
            if len(points)>=3:distances.extend(cKDTree(component.points_body).query(selected.inverse_apply(points))[0].tolist())
        distances=np.asarray(distances,float);inliers=distances<=self.tracker.config.maximum_registration_rms_m*2
        rms=float(np.sqrt(np.mean(distances[inliers]**2))) if inliers.any() else float('inf')
        fraction=float(inliers.mean()) if len(inliers) else 0.
        valid=rms<=self.tracker.config.maximum_registration_rms_m and fraction>=self.tracker.config.minimum_inlier_fraction
        row.update(qualified=bool(valid),selected_index=index,current_rms_m=rms,current_inlier_fraction=fraction);self.events.append(row)
        if not valid:return None
        result=deepcopy(fit);result.update(pose=selected,rms_m=rms,valid=True,rotation_ambiguous=False,
            per_role={role:dict(rms_m=rms,inlier_fraction=fraction,valid=True)},conditional_branch_v63=row)
        self.previous[role]=(now,selected)
        return result
