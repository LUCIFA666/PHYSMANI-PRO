"""Once-per-generation representatives for the existing parallel-jaw symmetry.

The frozen pilot and raw candidate ledger are unchanged. The equivalence used
by its training loss is R and R @ Rz(pi), a TCP-local right multiplication.
Selection changes no point/time/score and grants no reachability certificate.
"""
from copy import deepcopy
import math

import numpy as np
from scipy.spatial.transform import Rotation

from ..contracts import Forecast
from .data import unit_quaternion
from .runtime import LearnedCandidates


_LOCAL_Z_PI=np.diag([-1.,-1.,1.])
_TIE_RAD=1e-12
_CREATION_REASONS=frozenset(('initial_model_generation','all_fixed_deadlines_expired_new_model_generation'))


def _canonical_xyzw(matrix):
    q=Rotation.from_matrix(matrix).as_quat()
    first=next((v for v in q[:3] if abs(v)>1e-12),0.)
    if q[3]<0 or (abs(q[3])<=1e-12 and first<0):q=-q
    return q.tolist()


def choose_pi_representative(raw_quaternion,forecast,deadline,current_TCP_quaternion):
    """Use a predicted world orientation at the UNCHANGED original deadline.

    Bounds are a strict closed interval, with no outside-domain tolerance. This
    precheck prevents Forecast.at's 1e-8 roundoff clipping from moving a sample.
    Inside the domain, Forecast.at uses adjacent-pose SO(3) Slerp. Ties within
    1e-12 rad retain the raw representation, never a policy-dependent flip.
    """
    raw=unit_quaternion(raw_quaternion,'raw object-local TCP quaternion')
    tcp=unit_quaternion(current_TCP_quaternion,'actual current public TCP quaternion')
    if (isinstance(deadline,(bool,np.bool_)) or not np.isscalar(deadline)
            or not math.isfinite(deadline) or deadline<0):raise ValueError('finite original absolute deadline required')
    deadline=float(deadline)
    record=dict(raw_quaternion_xyzw=np.asarray(raw_quaternion,dtype=float).tolist(),
        selected_quaternion_xyzw=np.asarray(raw_quaternion,dtype=float).tolist(),selection_index=None,
        angles_rad=None,available=False,original_absolute_deadline=deadline,sample_time=None,
        actual_public_TCP_quaternion_xyzw=np.asarray(current_TCP_quaternion,dtype=float).tolist(),
        symmetry='raw_R @ Rz(pi), TCP-local right multiplication',
        interpolation='Forecast.at adjacent SO3 Slerp; strict-domain precheck, zero outside-domain tolerance',
        deadline_or_sample_clamped=False,reachability_or_success_qualified=False)
    if type(forecast) is not Forecast or 'target' not in forecast.object_ids:
        record['reason']='current_predicted_target_forecast_unavailable';return record
    record['forecast_domain']=[float(forecast.times[0]),float(forecast.times[-1])]
    if deadline<forecast.times[0] or deadline>forecast.times[-1]:
        record['reason']='original_deadline_outside_current_forecast_domain';return record
    object_pose,_,_=forecast.at('target',deadline)
    world0=object_pose.rotation@raw.as_matrix()
    world1=world0@_LOCAL_Z_PI
    angle0=float((tcp.inv()*Rotation.from_matrix(world0)).magnitude())
    angle1=float((tcp.inv()*Rotation.from_matrix(world1)).magnitude())
    selected=1 if angle1<angle0-_TIE_RAD else 0
    record.update(available=True,selection_index=selected,angles_rad=[angle0,angle1],sample_time=deadline,
        object_world_rotation_at_deadline=object_pose.rotation.tolist(),reason='fixed_generation_nearest_equivalent_representative')
    if selected:record['selected_quaternion_xyzw']=_canonical_xyzw(raw.as_matrix()@_LOCAL_Z_PI)
    return record


def _forecast(row):
    if row is None:return None
    value=row['result'].get('forecast')
    if value is None:return None
    return Forecast(tuple(value['object_ids']),value['times'],value['positions'],value['rotations'],
        value['linear_velocities'],value['angular_velocities'],metadata={})


class SymmetricLearnedCandidates(LearnedCandidates):
    """Raw ledger stays immutable; returned pose clones use one fixed choice.

    A slot whose original deadline cannot be sampled at creation stays masked
    for that generation. Later observations never retroactively choose a pose.
    All original expiry, cache-epoch retirement and warm-clear events survive.
    """
    def __init__(self,*args,**kwargs):
        self._representatives={};super().__init__(*args,**kwargs)
    @property
    def raw_candidates(self):return self.ledger.candidates
    @property
    def candidates(self):
        return [] if self.last_snapshot is None else deepcopy(self.last_snapshot['candidates'])
    @property
    def events(self):
        events=[]
        for event in super().events:
            events.append(event)
            selection=self._representatives.get(event.get('generation'))
            if (selection is not None and event.get('reason') in _CREATION_REASONS
                    and event.get('cache_epoch')==selection['cache_epoch']):
                events.append(deepcopy(selection))
        return events
    def update(self,state):
        raw_rows=super().update(state);generation=self.generation
        if generation not in self._representatives:
            row=self.cache.latest;forecast=_forecast(row)
            choices=[]
            for raw in raw_rows:
                choice=choose_pi_representative(raw['object_tcp_quaternion'],forecast,
                    raw['deadline'],state['tcp_quaternion'])
                choice.update(id=raw['id'],object_position_m=deepcopy(raw['object_tcp_position']),
                    original_relative_time_seconds=raw['relative_time_seconds'])
                choices.append(choice)
            self._representatives[generation]=dict(time=float(state['time']),generation=generation,
                cache_epoch=self.cache.epoch,reason='parallel_gripper_pi_representatives_selected_once_at_generation_creation',
                slots=choices,actual_public_TCP_quaternion_xyzw=np.asarray(state['tcp_quaternion']).tolist(),
                warm_start_clear_required=True,raw_model_output_or_raw_ledger_modified=False,
                object_positions_times_IDs_scores_unchanged=True,active_generation_reselection_allowed=False,
                current_public_forecast_only=True,collision_reachability_or_task_success_qualified=False)
        selection=self._representatives[generation]
        by_id={choice['id']:choice for choice in selection['slots']}
        chosen=deepcopy(raw_rows);mask=np.asarray(self.last_snapshot['available_mask'],dtype=bool).copy()
        for index,candidate in enumerate(chosen):
            choice=by_id[candidate['id']]
            candidate['raw_model_object_tcp_quaternion']=deepcopy(choice['raw_quaternion_xyzw'])
            candidate['object_tcp_quaternion']=deepcopy(choice['selected_quaternion_xyzw'])
            candidate['parallel_gripper_symmetry_choice']=choice['selection_index']
            candidate['symmetry_choice_anchor_time']=selection['time']
            if not choice['available']:
                mask[index]=False;candidate['proposal_available']=False
                why='symmetry_original_deadline_forecast_unavailable_at_generation_creation'
                candidate['availability_reasons']=list(candidate.get('availability_reasons',[]))
                if why not in candidate['availability_reasons']:candidate['availability_reasons'].append(why)
        self.last_snapshot.update(candidates=chosen,raw_candidates=deepcopy(raw_rows),available_mask=mask,
            mask_reasons=[c['availability_reasons'] for c in chosen],events=self.events,
            symmetry_generation_record=deepcopy(selection),raw_model_pose_unmodified_in_ledger=True,
            representative_fixed_within_generation=True,availability_is_reachability=False)
        return chosen
