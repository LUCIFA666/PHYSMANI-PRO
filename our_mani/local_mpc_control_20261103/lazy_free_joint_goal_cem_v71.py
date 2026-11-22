"""V71 FREE-only stable top8 selection using the original first-action checks.

The original three CEM batches/RNG/elites/cost/model remain unchanged. Native
endpoint/seven-path queries run only after all256 scores, before phase/warm
commit. Hold/probe/response-seed planners do not use this class. The runner
performs the original final action qualification AGAIN, counted separately.
"""
from copy import deepcopy
from pathlib import Path
from dataclasses import asdict
import hashlib
import time
import numpy as np

from .joint_goal_cem import JointGoalCEMPlanner,PARAM_SPACE,NOMINAL_BASIS_VERSION,reference,validate_feedback_state
from .forecast_control import ForecastJointGoalCEMPlanner,BoundForecastExecutionModel
from .lazy_ranked_action_qualification_prototype_20261103 import RankedRolloutPool,select_first_qualified,_digest
from .public_candidate_local_hold_v66 import _assert_current_clock

VERSION='FREE_GRU_existing256_stable_top8_original_qualification_before_phase_warm_20261103_v71'


def lazy_free_planner_source_map():
    root=Path(__file__).resolve().parents[2]
    names=(
        'our_mani/local_mpc_control_20261103/lazy_ranked_action_qualification_prototype_20261103.py',
        'our_mani/local_mpc_control_20261103/joint_goal_cem.py',
        'our_mani/local_mpc_control_20261103/forecast_control.py',
        'our_mani/local_mpc_control_20261103/joint_goals.py',
    )
    result={str((root/name).resolve()):hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    path=Path(__file__).resolve();result[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class _LazyJointGoalCEMPlannerV71(JointGoalCEMPlanner):
    def plan(self,state,candidates,*,warm_start=None):
        started=time.perf_counter();c=self.config
        self.last_model_diagnostics=[]
        state=validate_feedback_state(state);candidates=[reference.validate_candidate(row) for row in candidates]
        self._assert_caller_current()
        pool=RankedRolloutPool(state)
        if len(candidates)!=8 or len({row['id'] for row in candidates})!=8:raise ValueError('K8 distinct candidates required')
        warm=deepcopy(warm_start or {});committed=warm.get('selected_candidate_id')
        if warm and (warm.get('param_space')!=PARAM_SPACE or warm.get('nominal_basis_version')!=NOMINAL_BASIS_VERSION):
            raise ValueError('joint-goal v1 warm required; Cartesian/unknown basis warm cannot be reused')
        if warm and state['time']<float(warm['time'])-1e-9:raise ValueError('future warm feedback')
        if warm and committed not in {row['id'] for row in candidates}:raise ValueError('clear expired generation explicitly')
        for row in candidates:
            if row['id']==committed and abs(row['deadline']-warm['deadline'])>1e-8:raise ValueError('deadline cannot be pushed forward')
        old=deepcopy(state.get('phase_state',warm.get('phase_state',dict(phase='approach',dwell=0.,lost_hold=0.,updated_at=None))))
        self._prepare_nominals(state,candidates,old)
        means=np.zeros((8,2,7));std=np.full_like(means,self.joint_config.initial_std_rad)
        floors=np.full(7,self.joint_config.std_floor_rad)
        grid=np.linspace(0.,1.,8);knots=np.linspace(0.,1.,c.knots)
        shifted=deepcopy(warm.get('sequence',[]));shift=0
        if shifted:
            if len(shifted)!=8:raise ValueError('H8 warm sequence required')
            shift=min(8,max(0,int(round((state['time']-float(warm['time']))/c.dt))))
            shifted=shifted[shift:]+[deepcopy(shifted[-1]) for _ in range(shift)]
        if 'knot_means' in warm:
            old_means=np.asarray(warm['knot_means']);old_stds=np.asarray(warm['knot_stds'])
            if old_means.shape!=means.shape or old_stds.shape!=std.shape or not np.isfinite(old_means).all() or not np.isfinite(old_stds).all():
                raise ValueError('finite joint warm statistics [8,2,7] required')
            std=np.maximum(old_stds,floors)
            # Old delta means are not blindly transplanted onto new object/IK
            # bases. Noncommitted modes restart around their fresh nominal.
        if committed in [row['id'] for row in candidates] and shifted:
            index=next(i for i,row in enumerate(candidates) if row['id']==committed)
            absolute=np.stack([np.asarray(a['joint_goal'],float) for a in shifted])
            kinds=np.asarray([int(a['nominal_kind']) for a in shifted])
            if absolute.shape!=(8,7) or not np.isfinite(absolute).all() or np.any((kinds<0)|(kinds>2)):
                raise ValueError('finite absolute joint warm sequence with nominal-kind ledger required')
            dense=absolute-self._nominals['joint_goal'][kinds,index]
            for axis in range(7):means[index,:,axis]=np.interp(knots,grid,dense[:,axis])
        weights=np.array([np.interp(grid,knots,np.eye(c.knots)[i]) for i in range(c.knots)]).T
        rng=np.random.default_rng(c.seed+int(round(state['time']/c.dt)))
        best=None;counts=[];warm_count=0
        for iteration,total in enumerate(c.iteration_samples):
            per=total//8;modes=np.repeat(np.arange(8),per);pieces=[]
            for mode in range(8):
                samples=rng.normal(means[mode],std[mode],size=(per,2,7));samples[0]=means[mode]
                pieces.append(samples)
            samples=np.concatenate(pieces);controls=np.einsum('hk,nkj->hnj',weights,samples)
            scores,parts,states,actions=self._rollout_batch(state,candidates,modes,controls,old,committed,shifted if shifted else None)
            self._assert_caller_current()
            pool.offer_batch(iteration,scores,modes,candidates,parts,states,actions,c.dt)
            win=int(np.argmin(scores))
            if best is None or scores[win]<best['cost']:
                candidate=candidates[modes[win]]
                seq=[{key:values[win,k].copy() if values[win,k].shape else values[win,k].item() for key,values in actions.items()} for k in range(8)]
                for action in seq:action['dt']=c.dt
                predicted=[deepcopy(state)]
                for k in range(1,9):
                    item={key:values[win,k].copy() if values[win,k].shape else bool(values[win,k]) if key=='held' else float(values[win,k]) for key,values in states.items()}
                    if 'lift_goal_position' in state:item['lift_goal_position']=state['lift_goal_position'].copy()
                    predicted.append(item)
                best=dict(cost=float(scores[win]),candidate_id=candidate['id'],deadline=candidate['deadline'],sequence=seq,
                          cost_terms={key:float(value[win]) for key,value in parts.items()},states=predicted)
            mode_costs=[]
            for mode in range(8):
                a,b=mode*per,(mode+1)*per;reserved=bool(shifted and candidates[mode]['id']==committed)
                if reserved:warm_count+=1
                local=scores[a:b];fit=np.arange(1,per) if reserved else np.arange(per)
                elite=fit[np.argsort(local[fit],kind='stable')[:max(2,int(np.ceil(per*c.elite_fraction)))]]
                chosen=pieces[mode][elite];fraction=c.update_fraction
                means[mode]=(1-fraction)*means[mode]+fraction*chosen.mean(0)
                std[mode]=np.maximum(floors,(1-fraction)*std[mode]+fraction*chosen.std(0))
                mode_costs.append(float(min(local)))
            counts.append(dict(iteration=iteration,sequences=total,per_candidate=per,minimum_cost_by_candidate=mode_costs))
        budget=dict(sequences=256,horizon=8,candidate_modes=8,iterations=3,iteration_sequences=[88,88,80],virtual_macrosteps=2048,virtual_50ms_substeps=4096,Native_steps=0)
        self._assert_caller_current()
        lazy=select_first_qualified(pool,self.goal_helper,self.backend,state,mode='FREE_GRU')
        self._assert_caller_current()
        selected=lazy.pop('selection_bundle')
        lazy['selected_rank']=None if selected is None else selected['selected_rank']
        lazy['retained_order']=[dict(cost=row['cost'],iteration=row['iteration'],row_index=row['row_index'],
            candidate_id=row['candidate_id'],deadline=row['deadline'],first_q=row['sequence'][0]['joint_goal'].copy())
            for row in pool.ranked()]
        all_model_infeasible=best is None or not np.isfinite(best['cost'])
        if selected is not None:
            best=dict(cost=selected['cost'],candidate_id=selected['candidate_id'],deadline=selected['deadline'],
                sequence=selected['sequence'],cost_terms=selected['cost_terms'],states=selected['predicted_states'])
        else:
            best=None
        no_action_reason='all_sequences_infeasible' if all_model_infeasible else 'all_retained_first_actions_rejected_by_original_qualification'
        if best is None or not np.isfinite(best['cost']):
            return dict(action=None,sequence=[],selected_candidate_id=None,phase_state=old,warm_start=None,budget=budget,
                        diagnostics=dict(reason=no_action_reason,V71_lazy_free_qualification=deepcopy(lazy),iterations=counts,phase_real_updates=0,
                            wall_seconds=time.perf_counter()-started,real_time_guaranteed=False,
                            action_parameterization=PARAM_SPACE,trajectory_equivalent_to_pose_CEM=False,
                            execution_model_calls=len(self.last_model_diagnostics),
                            nominal_DLS=deepcopy(self.nominal_receipt),virtual_step_IK_calls=0,
                            joint_CEM_config=asdict(self.joint_config),diagnostic_only=True,formal_qualification=False))
        candidate=next(row for row in candidates if row['id']==best['candidate_id'])
        actual=reference.update_phase(state,candidate,old,c)
        warm_out=dict(time=state['time'],selected_candidate_id=candidate['id'],deadline=candidate['deadline'],phase_state=deepcopy(actual),
                      sequence=deepcopy(best['sequence']),param_space=PARAM_SPACE,nominal_basis_version=NOMINAL_BASIS_VERSION,candidate_ids=[row['id'] for row in candidates],knot_means=means.copy(),knot_stds=std.copy())
        return dict(action=deepcopy(best['sequence'][0]),sequence=best['sequence'],selected_candidate_id=best['candidate_id'],
                    phase_state=actual,warm_start=warm_out,cost=best['cost'],cost_terms=best['cost_terms'],predicted_states=best['states'],budget=budget,
                    diagnostics=dict(V71_lazy_free_qualification=deepcopy(lazy),iterations=counts,warm_start_evaluations_counted=warm_count,phase_real_updates=1,
                                     wall_seconds=time.perf_counter()-started,real_time_guaranteed=False,batch_prototype=True,value_network_used=False,
                                     execution_model_calls=len(self.last_model_diagnostics),execution_model_diagnostics=deepcopy(self.last_model_diagnostics),
                                     virtual_h_used_as_next_real_feedback=False,
                                     action_parameterization=PARAM_SPACE,trajectory_equivalent_to_pose_CEM=False,
                                     joint_CEM_config=asdict(self.joint_config),nominal_DLS=deepcopy(self.nominal_receipt),
                                     virtual_step_IK_calls=0,diagnostic_only=True,formal_qualification=False))


class LazyQualifiedForecastJointGoalCEMPlannerV71(ForecastJointGoalCEMPlanner,_LazyJointGoalCEMPlannerV71):
    """Original forecast/warm wrapper around the one changed FREE selection."""
    def __init__(self,*args,backend,**kwargs):
        self.backend=backend
        if not callable(getattr(backend,'probe_joint_collision',None)) or not callable(getattr(backend,'clock',None)):
            raise TypeError('Original same-frame no-step collision query service required')
        super().__init__(*args,**kwargs)
        if type(self.execution_model) is not BoundForecastExecutionModel:
            raise TypeError('Exact current public Forecast execution binding required')
        if self.execution_model.public_bridge_verified is not True or self.execution_model.held_source!='public_estimate':
            raise ValueError('Genuine current public FREE Forecast binding required')
        self._lazy_sources=lazy_free_planner_source_map()
        self._caller_state=None;self._caller_sha=None

    def _assert_caller_current(self):
        if self._caller_state is None or _digest(self._caller_state)!=self._caller_sha:
            raise ValueError('Original same-frame public feedback changed during FREE planning')
        _assert_current_clock(self.backend,self._caller_state['time'])
        if self._lazy_sources!=lazy_free_planner_source_map():
            raise ValueError('V71 FREE planner/source closure changed')

    def plan(self,state,candidates,*,warm_start=None):
        if (state.get('grasp_motion_mode')!='FREE_GRU' or state.get('held') is not False
                or any(state.get(key) is not False for key in
                    ('Native_object_state_used','Native_held_used','Native_success_used'))):
            raise ValueError('V71 lazy qualification accepts original public FREE_GRU feedback only')
        if self._caller_state is not None:raise ValueError('No nested current feedback binding')
        self._caller_state=state;self._caller_sha=_digest(state)
        try:
            self._assert_caller_current()
            result=super().plan(state,candidates,warm_start=warm_start)
            self._assert_caller_current()
            result['diagnostics']['V71_lazy_free_qualification'].update(
                original_public_feedback_sha256=self._caller_sha,
                applies_only_to='FREE_GRU',hold_probe_response_seed_unchanged=True,
                final_runner_original_qualification_repeated=True,
                final_runner_query_maximum=8,total_first_action_query_maximum=72,
                CEM_rng_elites_costs_unchanged=True)
            return result
        finally:
            self._caller_state=None;self._caller_sha=None
