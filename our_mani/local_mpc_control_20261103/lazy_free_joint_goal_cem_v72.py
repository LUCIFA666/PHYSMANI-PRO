"""V72 FREE-only stable top8 selection using the original first-action checks.

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

from .joint_goal_cem import (JointGoalCEMPlanner,PARAM_SPACE,NOMINAL_BASIS_VERSION,
    validate_feedback_state,SimpleBatchExecutionModel,validate_step_batch,terminal_cost_batch,
    norm,unit,cross,qmul,qinv,qexp,qlog,rotate,limited,matrix_quaternion)
from . import free_phase_dwell_v72 as reference
from .free_phase_dwell_v72 import alignment_dwell_reached,ALIGNMENT_DWELL_NUMERICAL_ATOL_SECONDS
from .forecast_control import ForecastJointGoalCEMPlanner,BoundForecastExecutionModel
from .lazy_ranked_action_qualification_prototype_20261103 import RankedRolloutPool,select_first_qualified,_digest
from .public_candidate_local_hold_v66 import _assert_current_clock

VERSION='FREE_GRU_existing256_top8_alignment_dwell_absolute1ns_20261103_v72'


def stable_dwell_planner_source_map():
    root=Path(__file__).resolve().parents[2]
    names=(
        'our_mani/local_mpc_control_20261103/lazy_ranked_action_qualification_prototype_20261103.py',
        'our_mani/local_mpc_control_20261103/lazy_free_joint_goal_cem_v71.py',
        'our_mani/local_mpc_control_20261103/free_phase_dwell_v72.py',
        'our_mani/local_mpc_control_20261103/joint_goal_cem.py',
        'our_mani/local_mpc_control_20261103/forecast_control.py',
        'our_mani/local_mpc_control_20261103/joint_goals.py',
    )
    result={str((root/name).resolve()):hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    path=Path(__file__).resolve();result[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class _LazyJointGoalCEMPlannerV72(JointGoalCEMPlanner):
    def _rollout_batch(self,initial,candidates,modes,controls,old,committed,fixed_sequence):
        c=self.config;n=len(modes);dt=c.native_dt
        s,h=self.execution_model.initialize_batch(initial,n)
        if getattr(self.execution_model,'requires_joint_state',False) and ('q' not in s or 'dq' not in s):
            raise ValueError('joint-response model requires actual q/dq initialization')
        r=np.asarray([candidates[m]['object_tcp_position'] for m in modes])
        rq=np.asarray([candidates[m]['object_tcp_quaternion'] for m in modes])
        direction=np.asarray([candidates[m]['approach_direction'] for m in modes])
        offset=np.asarray([candidates[m]['pregrasp_offset'] for m in modes])
        required=np.asarray([candidates[m]['required_opening'] for m in modes])
        speed_limit=np.asarray([candidates[m]['allowed_relative_speed'] for m in modes])
        deadline=np.asarray([candidates[m]['deadline'] for m in modes])
        qualified=np.asarray([candidates[m].get('native_pose_qualified',True) for m in modes],bool)
        phases=np.full(n,reference.PHASES.index(old['phase']),int)
        dwell=np.full(n,float(old.get('dwell',0.)));lost=np.full(n,float(old.get('lost_hold',0.)))
        last=np.full(n,np.nan if old.get('updated_at') is None else old['updated_at'])
        close_entered=np.full(n,float(old.get('close_entered_at',np.nan)))
        reopening=np.full(n,bool(old.get('reopening',False)))
        goal_has=np.full(n,'lift_goal_position' in old)
        hold_q_has=np.full(n,'hold_tcp_quaternion' in old)
        goal=np.broadcast_to(old.get('lift_goal_position',initial.get('lift_goal_position',initial['object_position']+np.array([0.,0.,c.lift_height]))),(n,3)).copy()
        hold_q=np.broadcast_to(old.get('hold_tcp_quaternion',initial['tcp_quaternion']),(n,4)).copy()
        arrays={key:np.empty((n,9)+value.shape[1:],dtype=value.dtype) for key,value in s.items()}
        for key in arrays:arrays[key][:,0]=s[key]
        action_arrays={key:np.empty((n,8,size)) for key,size in
                       (('tcp_position',3),('tcp_quaternion',4),('tcp_velocity_reference',3),('angular_velocity_reference',3))}
        action_arrays['opening']=np.empty((n,8))
        for key in ('joint_goal','nominal_joint_goal','nominal_tcp_reference','endpoint_tcp_reference'):
            action_arrays[key]=np.empty((n,8,7))
        action_arrays['joint_goal_feasible']=np.empty((n,8),dtype=bool)
        action_arrays['phase_index']=np.empty((n,8),dtype=np.int64)
        action_arrays['nominal_kind']=np.empty((n,8),dtype=np.int64)
        action_arrays['nominal_reference_time']=np.empty((n,8))
        names=('position','orientation','relative_speed','opening','smoothness','effort','deadline','lift','slip','constraint')
        parts_total={key:np.zeros(n) for key in names};cost=np.zeros(n)
        warm_mask=np.zeros(n,bool)
        if fixed_sequence is not None:
            per_mode=n//8
            warm_mask[next(i for i,row in enumerate(candidates) if row['id']==committed)*per_mode]=True

        def target(at,pre):
            elapsed=at-s['time'];p=s['object_position']+elapsed[:,None]*s['object_linear_velocity']
            q=unit(qmul(qexp(elapsed[:,None]*s['object_angular_velocity']),s['object_quaternion']))
            lever=rotate(q,r+direction*(offset*pre)[:,None])
            return p+lever,unit(qmul(q,rq)),s['object_linear_velocity']+cross(s['object_angular_velocity'],lever),s['object_angular_velocity'].copy()

        def geometry():
            grasp,q,v,_=target(s['time'],np.zeros(n))
            pre,*_=target(s['time'],np.ones(n))
            return norm(s['tcp_position']-grasp),norm(s['tcp_position']-pre),norm(qlog(qmul(q,qinv(s['tcp_quaternion'])))),norm(s['tcp_velocity']-v)

        for k in range(8):
            distance,pre_distance,angle,relative=geometry()
            now=s['time'];fresh=np.isnan(last)|(np.abs(now-last)>1e-9)
            if np.any(now<last-1e-9):raise ValueError('phase feedback time moved backwards')
            elapsed=np.where(np.isnan(last),0.,np.maximum(now-last,0.))
            prior_phase=phases.copy()
            held=fresh&s['held'];phases[held]=3;dwell[held]=0.;lost[held]=0.
            new_goal=held&(~goal_has)
            if np.any(new_goal):
                goal[new_goal]=initial['lift_goal_position'] if 'lift_goal_position' in initial else s['object_position'][new_goal]+np.array([0.,0.,c.lift_height])
                hold_q[new_goal]=s['tcp_quaternion'][new_goal]
                goal_has[new_goal]=True;hold_q_has[new_goal]=True
            mask=fresh&(~s['held'])&(prior_phase==3)
            lost[mask]+=elapsed[mask];change=mask&(lost>=c.lost_hold_dwell);phases[change]=1;dwell[change]=0.
            mask=fresh&(~s['held'])&(prior_phase==0)
            good=(pre_distance<=c.approach_enter)&(angle<=c.angle_enter)
            dwell[mask]=np.where(good[mask],dwell[mask]+elapsed[mask],0.)
            change=mask&good&(dwell>=c.approach_dwell);phases[change]=1;dwell[change]=0.
            mask=fresh&(~s['held'])&(prior_phase==1)
            reopen=mask&reopening;dwell[reopen]=0.
            reopening[reopen&(s['opening']>=c.reopening_aperture)&(s['requested_opening']>=.5)]=False
            normal=mask&(~reopen)
            retreat=normal&(pre_distance>c.approach_exit)&(distance>c.align_exit)
            phases[retreat]=0;dwell[retreat]=0.
            good=(distance<=c.align_enter)&(angle<=c.angle_enter)&(relative<=speed_limit)&(s['opening']>=required-.05)
            check=normal&(~retreat);dwell[check]=np.where(good[check],dwell[check]+elapsed[check],0.)
            lead=np.full(n,c.close_lead_seconds) if c.close_lead_seconds is not None else c.opening_command_delay+s['opening']/c.opening_rate
            change=check&good&alignment_dwell_reached(dwell,c.align_dwell)&(now>=deadline-lead)
            phases[change]=2;dwell[change]=0.;close_entered[change]=now[change]
            mask=fresh&(~s['held'])&(prior_phase==2)
            entered=np.where(np.isnan(close_entered),np.where(np.isnan(last),now,last),close_entered)
            attempted=now>=entered+c.dt-1e-8
            bad=(distance>c.align_exit)|(angle>c.angle_exit)|(attempted&((s['opening']<c.reopening_aperture)|(s['requested_opening']<.5)))
            change=mask&bad;phases[change]=1;dwell[change]=0.;reopening[change]=True
            last[fresh]=now[fresh]

            at=s['time']+c.dt
            ap,aq,av,aw=target(at,(phases==0).astype(float))
            hold=phases==3
            local_goal=goal.copy();missing=~goal_has
            local_goal[missing]=initial['lift_goal_position'] if 'lift_goal_position' in initial else s['object_position'][missing]
            ap[hold]=local_goal[hold]+rotate(s['object_quaternion'][hold],r[hold])
            local_q=np.where(hold_q_has[:,None],hold_q,s['tcp_quaternion']);aq[hold]=local_q[hold]
            av[hold]=0.;aw[hold]=0.
            endpoint_reference=np.concatenate((ap,aq),axis=-1)
            nominal_kind=np.where(phases==0,0,np.where(phases==3,2,1))
            nominal_joint=self._nominals['joint_goal'][nominal_kind,modes]
            nominal_valid=self._nominals['feasible'][nominal_kind,modes]
            qgoal=nominal_joint+controls[k]
            opening=np.where((phases==2)|hold,0.,1.)
            pending=(phases==2)&(s['requested_opening']>=.5)
            opening[pending&((distance>c.align_enter)|(angle>c.angle_enter)|(relative>speed_limit)|(s['opening']<required-.05))]=1.
            if np.any(warm_mask):
                qgoal[warm_mask]=np.asarray(fixed_sequence[k]['joint_goal'],float)
                # A usable absolute warm proposal does not depend on whether
                # this replan's separately optimized nominal IK converged.
                nominal_valid=nominal_valid.copy();nominal_valid[warm_mask]=True
            bounds=np.all((qgoal>=self.goal_helper.adapter.lower-1e-8)&(qgoal<=self.goal_helper.adapter.upper+1e-8),axis=-1)
            perturb_ok=np.all(np.abs(qgoal-nominal_joint)<=self.joint_config.max_perturbation_rad+1e-12,axis=-1)
            feasible=nominal_valid&bounds&perturb_ok
            # No clipping of joint goals: invalid controls remain explicit
            # invalid rows and still consume their complete CEM budget.
            command_pose=self.goal_helper.kinematics(qgoal)
            ap=command_pose[:,:3,3];aq=matrix_quaternion(command_pose[:,:3,:3])
            action=dict(tcp_position=ap,tcp_quaternion=aq,opening=opening,
                tcp_velocity_reference=av,angular_velocity_reference=aw,
                joint_goal=qgoal,joint_goal_feasible=feasible,
                nominal_joint_goal=nominal_joint.copy(),nominal_kind=nominal_kind.copy(),phase_index=phases.copy(),
                nominal_tcp_reference=self._nominals['pose'][nominal_kind,modes].copy(),
                endpoint_tcp_reference=endpoint_reference,nominal_reference_time=self._nominals['time'][nominal_kind,modes].copy())
            for key in action_arrays:action_arrays[key][:,k]=action[key]
            before_macro=s
            if type(self.execution_model) is SimpleBatchExecutionModel:
                # This exact implementation is tested functional and copies x
                # before writes. Unknown replacements retain protected inputs.
                model_x,model_u,model_h=s,action,h
            else:
                model_x={key:value.copy() for key,value in s.items()}
                model_u={key:value.copy() for key,value in action.items()}
                model_h=deepcopy(h)
            nxt,h_next,info=self.execution_model.step_batch(model_x,model_u,model_h,c.dt)
            model_feasible=validate_step_batch(before_macro,action,nxt,h_next,info,c.dt)
            if 'joint_goal_feasible' in action:model_feasible=model_feasible&action['joint_goal_feasible']
            s,h=nxt,h_next
            self.last_model_diagnostics.append(dict(model=info.get('model'),batch=n,microsteps=info['microsteps'],
                feasible_sequences=int(model_feasible.sum()),joint_prediction=info.get('joint_prediction'),
                actual_feedback_reused_as_prediction=False))
            for key in arrays:arrays[key][:,k+1]=s[key]

            tp,tq,tv,_=target(s['time'],(phases==0).astype(float))
            position=norm(s['tcp_position']-tp)/c.position_scale
            orientation=norm(qlog(qmul(tq,qinv(s['tcp_quaternion']))))/c.orientation_scale
            relative_speed=norm(s['tcp_velocity']-tv)/speed_limit
            parts={'position':position**2,'orientation':orientation**2,'relative_speed':relative_speed**2,
                   'opening':(s['opening']-np.where((phases==2)|(phases==3),0.,1.))**2,
                   'smoothness':np.zeros(n),'effort':np.sum(s['tcp_velocity']**2,-1),
                   'deadline':np.where(phases==3,0.,(np.maximum(s['time']-deadline,0.)/c.dt)**2*(1.+position**2)),
                   'lift':np.zeros(n),'slip':np.zeros(n),'constraint':np.zeros(n)}
            if k:parts['smoothness']=np.sum(((ap-action_arrays['tcp_position'][:,k-1])/c.position_scale)**2,-1)
            hold=phases==3
            parts['position'][hold]=0.;parts['relative_speed'][hold]=0.
            local_goal=goal.copy();missing=~goal_has
            local_goal[missing]=initial['lift_goal_position'] if 'lift_goal_position' in initial else s['object_position'][missing]
            parts['lift'][hold]=np.sum(((s['object_position'][hold]-local_goal[hold])/c.position_scale)**2,-1)
            parts['slip'][hold]=np.where(s['held'][hold],0.,25.)
            if self.constraint_cost is not None:
                constraints=np.asarray(self.constraint_cost.batch(before_macro,action,s,qualified,s['time']),float)
                if constraints.shape!=(n,) or np.isnan(constraints).any() or np.any(constraints<0):raise ValueError('invalid batch constraints')
                parts['constraint']=constraints
            parts['constraint']=np.where(model_feasible,parts['constraint'],np.inf)
            step_cost=np.zeros(n)
            for key in names:
                parts_total[key]+=parts[key]
                if c.weights[key]:step_cost+=c.weights[key]*parts[key]
            any_inf=np.zeros(n,bool)
            for value in parts.values():any_inf|=np.isposinf(value)
            step_cost[any_inf]=np.inf
            cost+=step_cost
        switch=np.array([c.switch_penalty if old['phase'] in ('close','hold') and committed is not None and candidates[m]['id']!=committed else 0. for m in modes])
        cost+=switch;parts_total['switch']=switch
        if self.terminal_config.enabled:
            value,terminal_parts=terminal_cost_batch(initial,{key:values[:,-1] for key,values in arrays.items()},candidates,modes,old,
                config=c,terminal_config=self.terminal_config)
            cost+=value;parts_total['terminal_phi']=value
            for key,value in terminal_parts.items():parts_total['terminal_'+key]=value
        return cost,parts_total,arrays,action_arrays

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


class LazyQualifiedForecastJointGoalCEMPlannerV72(ForecastJointGoalCEMPlanner,_LazyJointGoalCEMPlannerV72):
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
        self._lazy_sources=stable_dwell_planner_source_map()
        self._caller_state=None;self._caller_sha=None

    def _assert_caller_current(self):
        if self._caller_state is None or _digest(self._caller_state)!=self._caller_sha:
            raise ValueError('Original same-frame public feedback changed during FREE planning')
        _assert_current_clock(self.backend,self._caller_state['time'])
        if self._lazy_sources!=stable_dwell_planner_source_map():
            raise ValueError('V72 FREE planner/source closure changed')

    def plan(self,state,candidates,*,warm_start=None):
        if (state.get('grasp_motion_mode')!='FREE_GRU' or state.get('held') is not False
                or any(state.get(key) is not False for key in
                    ('Native_object_state_used','Native_held_used','Native_success_used'))):
            raise ValueError('V72 lazy qualification accepts original public FREE_GRU feedback only')
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
            result['diagnostics']['V72_stable_FREE_alignment_dwell']=dict(
                numerical_absolute_tolerance_seconds=ALIGNMENT_DWELL_NUMERICAL_ATOL_SECONDS,
                original_align_dwell_seconds=self.config.align_dwell,
                scalar_and_vector_predicates_consistent=True,timestamp_or_dwell_value_modified=False,
                other_phase_position_velocity_angle_deadline_gates_unchanged=True,
                applies_only_to='FREE_GRU',physical_grasp_success_claimed=False)
            return result
        finally:
            self._caller_state=None;self._caller_sha=None
