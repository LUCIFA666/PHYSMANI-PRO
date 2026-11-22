"""Causal two-centroid orientation estimates under a frozen TRAIN prior.

This helper grants no orientation permission and is not connected to a live
tracker. Its rigid-pair assumption is conditional on both public modes being
external. Attached/released/unknown/stale data never reuse an older estimate.
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional

import numpy as np

MODEL_VERSION='r10_TRAIN_learned_public_two_center_pose_prior_v1'
METHOD='r10_current_public_two_center_zero_bias_static_axis_v1'
TASKS=('beat_the_rotating_buzz','beat_the_rotating_buzz_high_speed')
NUMERIC_RADIAL_EPS_M=1e-8
TIMESTAMP_NUMERIC_TOLERANCE_SECONDS=1e-9


def require(value,message):
    if not value:raise ValueError('two-center pose: '+message)


def readonly(value,shape):
    result=np.array(value,dtype=float,copy=True)
    require(result.shape==shape and np.isfinite(result).all(),'finite static parameter shape differs')
    result.setflags(write=False)
    return result


def proper_rotation(value):
    value=readonly(value,(3,3))
    require(np.max(np.abs(value.T@value-np.eye(3)))<=1e-6 and abs(np.linalg.det(value)-1)<=1e-6,
        'static rotation/basis is not SO3')
    return value


@dataclass(frozen=True)
class TwoCenterPosePrior:
    task: str
    model_sha256: str
    canonical_asset_sha256: str
    local_axis: np.ndarray
    world_axis: np.ndarray
    local_frame_to_wand_offset: np.ndarray
    local_basis: np.ndarray
    frame_to_wand_rotation: np.ndarray
    evidence: Mapping = field(default_factory=dict)

    def __post_init__(self):
        require(self.task in TASKS,'unknown task')
        for checksum in (self.model_sha256,self.canonical_asset_sha256):
            require(isinstance(checksum,str) and len(checksum)==64 and all(c in '0123456789abcdef' for c in checksum),'missing immutable model/CAD binding')
        for name in ('local_axis','world_axis','local_frame_to_wand_offset'):
            object.__setattr__(self,name,readonly(getattr(self,name),(3,)))
        for name in ('local_axis','world_axis'):
            require(abs(np.linalg.norm(getattr(self,name))-1)<=1e-6,'static axis is not unit length')
        object.__setattr__(self,'local_basis',proper_rotation(self.local_basis))
        object.__setattr__(self,'frame_to_wand_rotation',proper_rotation(self.frame_to_wand_rotation))
        radial=self.local_frame_to_wand_offset-self.local_axis*np.dot(self.local_axis,self.local_frame_to_wand_offset)
        require(np.linalg.norm(radial)>NUMERIC_RADIAL_EPS_M,'static pair has no identifiable radial direction')
        radial=radial/np.linalg.norm(radial)
        expected=np.column_stack((radial,np.cross(self.local_axis,radial),self.local_axis))
        require(np.max(np.abs(expected-self.local_basis))<=1e-6,'static basis differs from learned axis/offset')
        object.__setattr__(self,'evidence',MappingProxyType(dict(self.evidence)))


@dataclass(frozen=True)
class PublicCenter:
    position: np.ndarray
    valid: bool
    mode_valid: bool
    mode: str
    observed_at: float

    def __post_init__(self):
        position=np.array(self.position,dtype=float,copy=True)
        require(position.shape==(3,),'public center position shape differs')
        position.setflags(write=False);object.__setattr__(self,'position',position)
        require(isinstance(self.valid,(bool,np.bool_)) and isinstance(self.mode_valid,(bool,np.bool_)),
            'public validity must be boolean')
        require(isinstance(self.mode,str),'public mode must be explicit text')
        require(isinstance(self.observed_at,(int,float,np.floating)),'public observation timestamp is absent')


@dataclass(frozen=True)
class TwoCenterPoseEstimate:
    valid: bool
    reason: str
    frame_rotation: Optional[np.ndarray] = None
    wand_rotation: Optional[np.ndarray] = None
    radial_length_m: Optional[float] = None
    axial_offset_m: Optional[float] = None
    model_sha256: Optional[str] = None
    grants_runtime_full_orientation: bool = False


def estimate_two_center_pose(prior:TwoCenterPosePrior,centers:Mapping[str,PublicCenter],*,current_timestamp:float):
    """Only current public centers, masks, modes and timestamps are consumed.

    The output remains a conditional estimate. A correct public `external`
    label alone cannot prove the two objects are still a rigid pair; callers
    must establish a separately reviewed phase/affordance permission before
    integrating this helper into a controller.
    """
    if set(centers)!={'frame','wand'}:return TwoCenterPoseEstimate(False,'unknown_missing_or_extra_role')
    if not all(isinstance(center,PublicCenter) for center in centers.values()):
        return TwoCenterPoseEstimate(False,'unknown_public_center_type')
    if not isinstance(current_timestamp,(int,float,np.floating)) or not np.isfinite(current_timestamp):
        return TwoCenterPoseEstimate(False,'invalid_current_timestamp')
    for role in ('frame','wand'):
        center=centers[role]
        if not center.valid:return TwoCenterPoseEstimate(False,'public_invalid:'+role)
        if not center.mode_valid:return TwoCenterPoseEstimate(False,'public_mode_unknown:'+role)
        if center.mode!='external':return TwoCenterPoseEstimate(False,'public_mode_not_external:'+role)
        if not np.isfinite(center.position).all():return TwoCenterPoseEstimate(False,'public_position_nonfinite:'+role)
        if not np.isfinite(center.observed_at):return TwoCenterPoseEstimate(False,'public_timestamp_nonfinite:'+role)
        age=float(current_timestamp)-float(center.observed_at)
        if age < -TIMESTAMP_NUMERIC_TOLERANCE_SECONDS:return TwoCenterPoseEstimate(False,'public_observation_from_future:'+role)
        if age > TIMESTAMP_NUMERIC_TOLERANCE_SECONDS:return TwoCenterPoseEstimate(False,'public_observation_stale:'+role)
    delta=centers['wand'].position-centers['frame'].position
    axial=float(np.dot(delta,prior.world_axis))
    radial=delta-prior.world_axis*axial;length=float(np.linalg.norm(radial))
    if not np.isfinite(length) or length<=NUMERIC_RADIAL_EPS_M:
        return TwoCenterPoseEstimate(False,'public_centroid_radial_vector_degenerate')
    radial=radial/length
    frame=np.column_stack((radial,np.cross(prior.world_axis,radial),prior.world_axis))@prior.local_basis.T
    wand=frame@prior.frame_to_wand_rotation
    frame.setflags(write=False);wand.setflags(write=False)
    return TwoCenterPoseEstimate(True,'conditional_external_pair_estimate',frame,wand,length,axial,prior.model_sha256)


def checked_json(root,pointer):
    root=Path(root).resolve()
    require(isinstance(pointer,dict) and set(pointer)=={'path','sha256'},'immutable evidence pointer schema differs')
    path=root/pointer['path']
    require(not Path(pointer['path']).is_absolute() and path.resolve()==path and path.is_relative_to(root) and path.is_file(),
        'evidence path is outside ROOT or symlinked')
    payload=path.read_bytes()
    require(hashlib.sha256(payload).hexdigest()==pointer['sha256'],'evidence bytes changed')
    return json.loads(payload)


def load_two_center_pose_prior(root,pointer,*,canonical_asset_pointer):
    """Strict immutable TRAIN/CAD/evidence loader; it changes no protocol."""
    value=checked_json(root,pointer)
    require(set(value)=={'schema','method','task','roles','static_prior','canonical_basis','TRAIN_evidence',
        'helper_source','configuration','runtime_scope','orientation_permission_granted'},'model fields differ')
    require(value['schema']==MODEL_VERSION and value['method']==METHOD and value['task'] in TASKS
        and value['roles']=={'frame':'frame','wand':'wand'} and value['orientation_permission_granted'] is False,
        'model method/task/roles/permission differs')
    require(value['runtime_scope']=='conditional_fresh_external_pair_only','unknown pose inference scope')
    require(value['configuration']==dict(public_centroid_bias_correction=False,numeric_radial_epsilon_m=NUMERIC_RADIAL_EPS_M,
        timestamp_numeric_tolerance_seconds=TIMESTAMP_NUMERIC_TOLERANCE_SECONDS),'fixed zero-bias/current-only configuration differs')
    basis=value['canonical_basis']
    require(set(basis)=={'TRAIN_CAD','asset_sha256','role_native_roots','frame_definition'}
        and basis['TRAIN_CAD']==canonical_asset_pointer and basis['role_native_roots']=={'frame':'Cuboid','wand':'wand'}
        and basis['frame_definition']=='Native role-root local axes used both by frozen TRAIN CAD export and TRAIN target_R labels',
        'canonical CAD/role-root basis differs')
    asset=checked_json(root,canonical_asset_pointer)
    require(asset.get('split')=='train' and asset.get('task')==value['task'] and asset.get('asset_sha256')==basis['asset_sha256']
        and {'frame','wand'}<=set(asset.get('bodies',{})),'canonical TRAIN CAD identity differs')
    helper=value['helper_source'];helper_path=Path(root).resolve()/helper['path']
    require(set(helper)=={'path','sha256'} and helper_path.resolve()==Path(__file__).resolve()
        and hashlib.sha256(helper_path.read_bytes()).hexdigest()==helper['sha256'],'model helper source differs')
    evidence=value['TRAIN_evidence']
    require(set(evidence)=={'parent_crossvalidation','tails_and_canonical_basis','training_model_receipt'}
        and all(isinstance(p,dict) for p in evidence.values()),'TRAIN model evidence is incomplete')
    crossvalidation=checked_json(root,evidence['parent_crossvalidation'])
    tails=checked_json(root,evidence['tails_and_canonical_basis'])
    receipt=checked_json(root,evidence['training_model_receipt'])
    prior=value['static_prior']
    require(set(prior)=={'local_axis','world_axis','local_frame_to_wand_offset','local_basis','frame_to_wand_rotation'},
        'unknown or dynamic prior parameters')
    require(crossvalidation.get('schema')=='r10_TRAIN_buzz_public_centroid_pose_fixed_parent_crossvalidation_v1'
        and crossvalidation.get('all_200_original_hold_parents_retained') is True
        and crossvalidation.get('original_8_actual_registrations_verified') is True
        and crossvalidation.get('runtime_future_or_native_driver_inputs') is False
        and crossvalidation.get('no_outlier_filter_or_threshold_search') is True
        and crossvalidation.get('orientation_none_protocol_changed') is False,
        'TRAIN parent-CV provenance differs')
    task_cv=crossvalidation['tasks'][value['task']]
    metrics=task_cv['variants']['no_public_bias_correction']
    require(task_cv['parents']==100 and task_cv['frames']==6100 and metrics['total_parents']==100
        and metrics['total_frames']==6100 and metrics['predicted_frames']==6100 and metrics['failures']=={}
        and metrics['frame_SO3_error_rad']['q95']<.15 and metrics['wand_SO3_error_rad']['q95']<.15,
        'fixed TRAIN hold coverage or original angular gate failed')
    require(tails.get('schema')=='r10_TRAIN_buzz_two_center_pose_source_basis_and_tail_audit_v1'
        and tails.get('parent_crossvalidation')==evidence['parent_crossvalidation']
        and tails['canonical_bases'][value['task']]==basis
        and tails.get('outliers_removed') is False,'TRAIN tails/canonical basis evidence differs')
    require(receipt.get('schema')=='r10_buzz_two_center_TRAIN_static_prior_receipt_v1'
        and receipt.get('task')==value['task'] and receipt.get('split')=='train'
        and receipt.get('behavior')=='TRAIN_hold_motion_identification'
        and receipt.get('fixed_original_hold_parents')==100 and receipt.get('static_prior')==prior
        and receipt.get('canonical_basis')==basis and receipt.get('helper_source')==helper
        and receipt.get('parent_crossvalidation')==evidence['parent_crossvalidation']
        and receipt.get('tails_and_canonical_basis')==evidence['tails_and_canonical_basis']
        and receipt.get('public_centroid_bias_correction') is False
        and receipt.get('orientation_permission_granted') is False,
        'full TRAIN static-fit receipt/model differs')
    parents=receipt.get('training_parent_source_groups',[])
    require(len(parents)==len(set(parents))==100
        and all(p.startswith('train:'+value['task']+':') for p in parents),'TRAIN parent roster differs')
    cases=receipt.get('actual_case_provenance',[])
    require(len(cases)==100 and len({case['case_id'] for case in cases})==100
        and {case['parent_source_group'] for case in cases}==set(parents)
        and all(case['task']==value['task'] and case['behavior']=='TRAIN_hold_motion_identification'
            and len(case['actual_plan_sha256'])==64 for case in cases),
        'actual TRAIN hold case identities differ from the fixed parent roster')
    registrations=checked_json(root,receipt['actual_registration_list'])
    require(isinstance(registrations,list) and len(registrations)==8,
        'original eight-registration TRAIN model lineage differs')
    for fold in ('0','1'):
        item=task_cv['folds'][fold];train=set(item['prior']['training_parents']);held=set(item['held_parents'])
        require(len(train)==len(held)==50 and not train&held and train|held==set(parents),
            'TRAIN parent fold separation differs')
    return TwoCenterPosePrior(value['task'],pointer['sha256'],basis['asset_sha256'],evidence=evidence,**prior)
