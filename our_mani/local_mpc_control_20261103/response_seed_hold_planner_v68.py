"""V68 cold hold CEM: one explicitly fresh row inside the original256.

The fixed inverse has additional separately reported computation. It never
uses history warm state, changes global cost, or inherits collision evidence.
"""
from copy import deepcopy
from pathlib import Path
from dataclasses import asdict
import hashlib,json,time
import numpy as np

from .public_candidate_local_hold_v66 import LocalHoldPlannerV66,_assert_current_clock,position_producer_source_map
from .hold_orientation_guard_prototype_20261103 import HoldMicroOrientationGuard,_identity
from .response_seed_generator_v68 import generate_response_seed,VERSION as GENERATOR_VERSION
from .joint_goal_cem import (reference,PARAM_SPACE,NOMINAL_BASIS_VERSION,norm,unit,qmul,qexp,qlog,qinv,rotate,cross,
    SimpleBatchExecutionModel,matrix_quaternion,validate_feedback_state,validate_step_batch,terminal_cost_batch)
from ..local_mpc_20261102.records import plain

HOLD_BASIS_VERSION='V68_original_static12_nominal_fresh_response_seed_cold_v1'
PROTOTYPE_SHA='95dbcfb36fdd9c70e45f64efb613ff5ddcbe0ec9debaed4763d6cd926225823b'


def response_seed_planner_source_map():
    sources=position_producer_source_map()
    for name in ('hold_response_feedforward_prototype_20261103.py','response_seed_generator_v68.py',Path(__file__).name):
        path=Path(__file__).resolve().with_name(name);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if name=='hold_response_feedforward_prototype_20261103.py' and digest!=PROTOTYPE_SHA:
            raise ValueError('Frozen original response study source required')
        sources[str(path)]=digest
    return sources


def _complete_feedback_sha(state):
    return hashlib.sha256(json.dumps(plain(state),sort_keys=True,allow_nan=False).encode()).hexdigest()


class ResponseSeedHoldPlannerV68(LocalHoldPlannerV66):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.joint_config.max_perturbation_rad!=.30 or tuple(self.config.iteration_samples)!=(88,88,80):
            raise ValueError('Original static +/-0.30 joint domain and256 CEM budget required')
        self._response_sources=response_seed_planner_source_map()
        self.response_seed_receipt=None

    def _compute_response_seed(self,state,candidates):
        now=float(state['time']);_assert_current_clock(self.backend,now)
        identity=_identity(state);complete=_complete_feedback_sha(state)
        receipt=dict(generator_version=GENERATOR_VERSION,feedback_time=now,feedback_sha256=identity,
            complete_feedback_sha256=complete,history_warm_used=False,sequence_shift_steps=0,
            injected=False,rejection_reason=None,generation_model_calls=None,Native_collision_qualified=False,
            selected_action_requires_original_endpoint_and_7_path=True)
        self.response_seed_receipt=receipt
        eligible=[i for i,c in enumerate(candidates) if c.get('native_pose_qualified')is True
            and bool(self._nominals['feasible'][2,i])]
        if not eligible:
            receipt['rejection_reason']='no_fresh_qualified_static_nominal';return None
        index=eligible[0];basis=self._nominals['joint_goal'][2,index].copy()
        generated=generate_response_seed(self._base_execution,self.goal_helper,state,basis)
        _assert_current_clock(self.backend,now)
        if _identity(state)!=identity or _complete_feedback_sha(state)!=complete:
            raise ValueError('Seed computation changed actual feedback or actuator h')
        if (generated.get('version')!=GENERATOR_VERSION or generated.get('feedback_sha256')!=identity
            or isinstance(generated.get('feedback_time'),(bool,np.bool_))
            or generated.get('feedback_time')!=now or generated.get('history_warm_used')is not False
            or type(generated.get('sequence_shift_steps'))is not int or generated.get('sequence_shift_steps')!=0
            or not np.array_equal(generated.get('static_nominal_q'),basis)):
            raise ValueError('Fresh same-frame source and zero-shift seed identity required')
        schedule=generated.get('schedule',{})
        if (not np.array_equal(schedule.get('current_TCP'),state['tcp_position'])
            or not np.array_equal(schedule.get('current_object'),state['object_position'])
            or not np.array_equal(schedule.get('global_object_goal'),state['phase_state']['lift_goal_position'])
            or not np.allclose(schedule.get('times'),now+.1*np.arange(1,9),atol=1e-8,rtol=0)):
            raise ValueError('Seed reference must originate from this exact current observation')
        receipt.update(candidate_id=candidates[index]['id'],candidate_index=index,static_nominal_q=basis.copy(),
            generation_model_calls=deepcopy(generated['model_calls']),
            generation_wall_seconds=generated['generation_wall_seconds'],generation=generated)
        if generated.get('model_feasible_horizon')is not True:
            receipt['rejection_reason']='response_inverse_horizon_infeasible';return None
        sequence=np.asarray(generated.get('proposed_q_sequence'),float)
        if sequence.shape!=(8,7)or not np.isfinite(sequence).all():
            receipt['rejection_reason']='invalid_complete_response_sequence';return None
        if (np.any(abs(sequence-basis)>.30+1e-12)
            or np.any(sequence<self.goal_helper.adapter.lower-1e-8)
            or np.any(sequence>self.goal_helper.adapter.upper+1e-8)):
            receipt['rejection_reason']='response_sequence_outside_original_static_domain';return None
        receipt['injected']=True
        return dict(candidate_index=index,feedback_time=now,feedback_sha256=identity,
            complete_feedback_sha256=complete,history_warm_used=False,sequence_shift_steps=0,
            sequence=[dict(joint_goal=q.copy())for q in sequence])

    def _validate_fresh_rollout_binding(self,initial,candidates,modes,proposal):
        index=proposal.get('row_index');mode=proposal.get('candidate_index')
        if (type(index)is not int or not 0<=index<len(modes) or type(mode)is not int
            or not 0<=mode<8 or int(modes[index])!=mode
            or proposal.get('history_warm_used')is not False or type(proposal.get('sequence_shift_steps'))is not int
            or proposal.get('sequence_shift_steps')!=0 or isinstance(proposal.get('feedback_time'),(bool,np.bool_))
            or proposal.get('feedback_time')!=initial['time'] or proposal.get('feedback_sha256')!=_identity(initial)
            or proposal.get('complete_feedback_sha256')!=_complete_feedback_sha(initial)):
            raise ValueError('Explicit fresh current row binding required; no history warm sequence')
        if (candidates[mode].get('native_pose_qualified')is not True
            or not bool(self._nominals['feasible'][2,mode])):
            raise ValueError('Fresh seed cannot bypass original current nominal qualification')
        q=np.asarray([step['joint_goal']for step in proposal.get('sequence',[])],float)
        basis=self._nominals['joint_goal'][2,mode]
        if q.shape!=(8,7)or not np.isfinite(q).all()or np.any(abs(q-basis)>.30+1e-12):
            raise ValueError('Unshifted complete H8 proposal in original static domain required')

    def plan(self,state,candidates,*,warm_start=None):
        if (state.get('grasp_motion_mode')!='ATTACHED_PUBLIC'or state.get('held')is not True
            or (state.get('phase_state')or{}).get('phase')!='hold'):
            raise ValueError('V68 accepts only current publicly confirmed hold feedback')
        if self._response_sources!=response_seed_planner_source_map():
            raise ValueError('Frozen V68 response-seed source changed')
        _assert_current_clock(self.backend,state['time'])
        guard=HoldMicroOrientationGuard(self._base_execution,state,maximum_rotation_rad=.04)
        self._actual_hold_state=deepcopy(state);self.execution_model=guard
        try:
            result=self._plan_cold_with_fresh_seed(state,candidates)
        finally:self.execution_model=self._base_execution
        _assert_current_clock(self.backend,state['time'])
        if self._response_sources!=response_seed_planner_source_map():
            raise ValueError('Frozen V68 source changed during planning')
        if result.get('warm_start')is not None:
            result['warm_start']['nominal_basis_version']=HOLD_BASIS_VERSION
            result['warm_start']['hold_cold_initialization_required']=True
        result['diagnostics']['fresh_response_seed_evaluations_counted']=int(
            'CEM_row_index'in self.response_seed_receipt)
        self.response_seed_receipt['selected']=bool(result.get('sequence')
            and self.response_seed_receipt.get('injected')
            and result.get('selected_candidate_id')==self.response_seed_receipt.get('candidate_id')
            and np.array_equal([step['joint_goal']for step in result['sequence']],
                               self.response_seed_receipt['generation']['proposed_q_sequence']))
        result['diagnostics']['V68_fresh_response_seed']=deepcopy(self.response_seed_receipt)
        result['diagnostics']['V68_hold_control']=dict(initialization='cold_each_actual_hold_replan',
            incoming_warm_discarded=warm_start is not None,feedback_sha256=guard.feedback_sha256,
            maximum_model_micro_rotation_rad=.04,maximum_nominal_TCP_translation_m=.012,
            model_guard_calls=deepcopy(guard.guard_calls),original_global_goal_and_full_cost_unchanged=True,
            fresh_sequence_only_iteration0=True,CEM_total_scoring_rows=256,
            inverse_generation_compute_additional_and_separately_reported=True,
            actual_or_continuous_orientation_guaranteed=False,Native_steps=0)
        return result


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
        fresh_mask=np.zeros(n,bool)
        if fixed_sequence is not None:
            self._validate_fresh_rollout_binding(initial,candidates,modes,fixed_sequence)
            fresh_mask[fixed_sequence['row_index']]=True

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
            change=check&good&(dwell>=c.align_dwell)&(now>=deadline-lead)
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
            if np.any(fresh_mask):
                qgoal[fresh_mask]=np.asarray(fixed_sequence['sequence'][k]['joint_goal'],float)
                # Fresh sequence never overrides the current local IK/native mask.
                # Original static perturbation, bounds and model checks follow.
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

    def _plan_cold_with_fresh_seed(self,state,candidates,*,warm_start=None):
        started=time.perf_counter();c=self.config
        self.last_model_diagnostics=[]
        state=validate_feedback_state(state);candidates=[reference.validate_candidate(row) for row in candidates]
        if len(candidates)!=8 or len({row['id'] for row in candidates})!=8:raise ValueError('K8 distinct candidates required')
        committed=None
        old=deepcopy(state['phase_state'])
        self._prepare_nominals(state,candidates,old)
        means=np.zeros((8,2,7));std=np.full_like(means,self.joint_config.initial_std_rad)
        floors=np.full(7,self.joint_config.std_floor_rad)
        grid=np.linspace(0.,1.,8);knots=np.linspace(0.,1.,c.knots)
        proposal=self._compute_response_seed(state,candidates)
        weights=np.array([np.interp(grid,knots,np.eye(c.knots)[i]) for i in range(c.knots)]).T
        rng=np.random.default_rng(c.seed+int(round(state['time']/c.dt)))
        best=None;counts=[];fresh_count=0
        for iteration,total in enumerate(c.iteration_samples):
            per=total//8;modes=np.repeat(np.arange(8),per);pieces=[]
            for mode in range(8):
                samples=rng.normal(means[mode],std[mode],size=(per,2,7));samples[0]=means[mode]
                pieces.append(samples)
            samples=np.concatenate(pieces);controls=np.einsum('hk,nkj->hnj',weights,samples)
            fresh=None
            if iteration==0 and proposal is not None:
                fresh=deepcopy(proposal);fresh['row_index']=fresh['candidate_index']*per+1
            scores,parts,states,actions=self._rollout_batch(state,candidates,modes,controls,old,committed,fresh)
            if fresh is not None:
                self.response_seed_receipt['CEM_score']=float(scores[fresh['row_index']])
                self.response_seed_receipt['CEM_row_index']=fresh['row_index']
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
                a,b=mode*per,(mode+1)*per;reserved=bool(fresh is not None and mode==fresh['candidate_index'])
                if reserved:fresh_count+=1
                local=scores[a:b]
                if proposal is not None and mode==proposal['candidate_index']:
                    fit=np.array([j for j in range(per) if (not reserved or j!=1) and np.isfinite(local[j])],dtype=int)
                else:
                    fit=np.arange(per)
                if len(fit):
                    elite=fit[np.argsort(local[fit],kind='stable')[:max(2,int(np.ceil(per*c.elite_fraction)))]]
                    chosen=pieces[mode][elite];fraction=c.update_fraction
                    means[mode]=(1-fraction)*means[mode]+fraction*chosen.mean(0)
                    std[mode]=np.maximum(floors,(1-fraction)*std[mode]+fraction*chosen.std(0))
                mode_costs.append(float(min(local)))
            counts.append(dict(iteration=iteration,sequences=total,per_candidate=per,minimum_cost_by_candidate=mode_costs))
        budget=dict(sequences=256,horizon=8,candidate_modes=8,iterations=3,iteration_sequences=[88,88,80],virtual_macrosteps=2048,virtual_50ms_substeps=4096,Native_steps=0)
        if best is None or not np.isfinite(best['cost']):
            return dict(action=None,sequence=[],selected_candidate_id=None,phase_state=old,warm_start=None,budget=budget,
                        diagnostics=dict(reason='all_sequences_infeasible',iterations=counts,phase_real_updates=0,
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
                    diagnostics=dict(iterations=counts,warm_start_evaluations_counted=0,fresh_response_seed_evaluations_counted=fresh_count,phase_real_updates=1,
                                     wall_seconds=time.perf_counter()-started,real_time_guaranteed=False,batch_prototype=True,value_network_used=False,
                                     execution_model_calls=len(self.last_model_diagnostics),execution_model_diagnostics=deepcopy(self.last_model_diagnostics),
                                     virtual_h_used_as_next_real_feedback=False,
                                     action_parameterization=PARAM_SPACE,trajectory_equivalent_to_pose_CEM=False,
                                     joint_CEM_config=asdict(self.joint_config),nominal_DLS=deepcopy(self.nominal_receipt),
                                     virtual_step_IK_calls=0,diagnostic_only=True,formal_qualification=False))
