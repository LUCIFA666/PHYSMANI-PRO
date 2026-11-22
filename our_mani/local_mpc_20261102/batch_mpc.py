"""Independent batch prototype; never imports or steps a Native environment.

The active project's frozen scalar planner is the reference. This prototype
supports its SimpleExecutionModel and a batch constraint callback only.
"""
from copy import deepcopy
import sys
import time
from pathlib import Path
import numpy as np

from . import mpc as reference


def norm(x):return np.sqrt(np.sum(x*x,axis=-1))
def unit(q):return q/np.maximum(norm(q)[...,None],1e-30)
def cross(a,b):
    return np.stack((a[...,1]*b[...,2]-a[...,2]*b[...,1],a[...,2]*b[...,0]-a[...,0]*b[...,2],a[...,0]*b[...,1]-a[...,1]*b[...,0]),-1)
def qmul(a,b):return np.concatenate((a[...,3,None]*b[...,:3]+b[...,3,None]*a[...,:3]+cross(a[...,:3],b[...,:3]),(a[...,3]*b[...,3]-np.sum(a[...,:3]*b[...,:3],axis=-1))[...,None]),-1)
def qinv(q):return np.concatenate((-q[...,:3],q[...,3,None]),-1)
def qexp(v):
    a=norm(v);small=a<1e-10
    factor=np.where(small,.5,np.sin(.5*a)/np.maximum(a,1e-30))
    result=np.concatenate((v*factor[...,None],np.where(small,1.,np.cos(.5*a))[...,None]),-1)
    return unit(result)
def qlog(q):
    q=unit(q);q=np.where(q[...,3,None]<0,-q,q);s=norm(q[...,:3])
    factor=np.where(s<1e-10,2.,2*np.arctan2(s,q[...,3])/np.maximum(s,1e-30))
    return q[...,:3]*factor[...,None]
def rotate(q,v):
    q=unit(q);u=q[...,:3]
    return v+2*cross(u,cross(u,v)+q[...,3,None]*v)
def limited(v,maximum):return v*np.minimum(1.,maximum/np.maximum(norm(v),1e-12))[...,None]


class BatchBounds:
    """Same point/end-state workspace predicate as DiagnosticConstraints."""
    def __init__(self,lower=(-.1,-.65,.76),upper=(.65,.65,1.65)):
        self.lower=np.asarray(lower);self.upper=np.asarray(upper)
    def __call__(self,state,action,nxt,candidate,at):
        if not candidate.get('native_pose_qualified',True):return np.inf
        for p in (action['tcp_position'],nxt['tcp_position']):
            if not np.isfinite(p).all() or np.any(p<self.lower) or np.any(p>self.upper):return np.inf
        return 0.
    def batch(self,state,action,nxt,qualified,at):
        p,a=nxt['tcp_position'],action['tcp_position']
        bad=(~qualified)|np.any((p<self.lower)|(p>self.upper),-1)|np.any((a<self.lower)|(a>self.upper),-1)
        bad|=~np.isfinite(p).all(-1)|~np.isfinite(a).all(-1)
        return np.where(bad,np.inf,0.)


class BatchMPCPlanner:
    def __init__(self,config=None,constraint_cost=None):
        self.config=config if isinstance(config,reference.MPCConfig) else reference.MPCConfig(**(config or {}))
        if constraint_cost is not None and not callable(getattr(constraint_cost,'batch',None)):
            raise ValueError('batch callback required: arbitrary scalar constraints are not silently bypassed')
        self.constraint_cost=constraint_cost

    def _rollout_batch(self,initial,candidates,modes,controls,old,committed,fixed_sequence):
        c=self.config;n=len(modes);dt=c.native_dt
        s={key:np.broadcast_to(initial[key],(n,)+np.shape(initial[key])).copy() for key in
           ('tcp_position','tcp_quaternion','tcp_velocity','tcp_angular_velocity','object_position','object_quaternion',
            'object_linear_velocity','object_angular_velocity','opening','held')}
        s['time']=np.full(n,initial['time'])
        actuator=initial.get('actuator_state',{})
        requested=float(initial.get('requested_opening',actuator.get('requested_opening',0. if initial['held'] else initial['opening'])))
        s['requested_opening']=np.full(n,requested)
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
        # Exact actual measured transform; nothing from a warm predicted h.
        object_in_tcp=rotate(qinv(s['tcp_quaternion']),s['object_position']-s['tcp_position'])
        object_q_tcp=qmul(qinv(s['tcp_quaternion']),s['object_quaternion'])
        arrays={key:np.empty((n,9)+value.shape[1:],dtype=value.dtype) for key,value in s.items()}
        for key in arrays:arrays[key][:,0]=s[key]
        action_arrays={key:np.empty((n,8,size)) for key,size in
                       (('tcp_position',3),('tcp_quaternion',4),('tcp_velocity_reference',3),('angular_velocity_reference',3))}
        action_arrays['opening']=np.empty((n,8))
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
            ap=ap+controls[k,:,:3];aq=unit(qmul(qexp(controls[k,:,3:]),aq))
            opening=np.where((phases==2)|hold,0.,1.)
            pending=(phases==2)&(s['requested_opening']>=.5)
            opening[pending&((distance>c.align_enter)|(angle>c.angle_enter)|(relative>speed_limit)|(s['opening']<required-.05))]=1.
            if np.any(warm_mask):
                a=fixed_sequence[k]
                ap[warm_mask]=a['tcp_position'];aq[warm_mask]=a['tcp_quaternion']
                av[warm_mask]=a['tcp_velocity_reference'];aw[warm_mask]=a['angular_velocity_reference']
            action={'tcp_position':ap,'tcp_quaternion':aq,'opening':opening,
                    'tcp_velocity_reference':av,'angular_velocity_reference':aw}
            for key in action_arrays:action_arrays[key][:,k]=action[key]
            before_macro={key:value.copy() for key,value in s.items()} if self.constraint_cost is not None else None
            previous_opening=s['requested_opening'].copy()
            for micro in range(2):
                before_object=s['object_position'].copy()
                wanted=limited(c.position_gain*(ap-s['tcp_position']),c.max_tcp_speed)
                next_v=s['tcp_velocity']+limited(wanted-s['tcp_velocity'],c.max_tcp_acceleration*dt)
                s['tcp_position']+=.5*(s['tcp_velocity']+next_v)*dt;s['tcp_velocity']=next_v
                error=qlog(qmul(aq,qinv(s['tcp_quaternion'])))
                desired_w=limited(c.rotation_gain*error,c.max_angular_speed)
                next_w=s['tcp_angular_velocity']+limited(desired_w-s['tcp_angular_velocity'],c.max_angular_acceleration*dt)
                delta=.5*(s['tcp_angular_velocity']+next_w)*dt
                s['tcp_quaternion']=unit(qmul(qexp(delta),s['tcp_quaternion']));s['tcp_angular_velocity']=next_w
                request=previous_opening if micro*dt<c.opening_command_delay-1e-9 else opening
                s['opening']+=np.clip(request-s['opening'],-c.opening_rate*dt,c.opening_rate*dt)
                s['held']&=request<.5
                attached=s['held'];free=~attached
                if np.any(attached):
                    s['object_position'][attached]=s['tcp_position'][attached]+rotate(s['tcp_quaternion'][attached],object_in_tcp[attached])
                    s['object_quaternion'][attached]=unit(qmul(s['tcp_quaternion'][attached],object_q_tcp[attached]))
                    s['object_linear_velocity'][attached]=(s['object_position'][attached]-before_object[attached])/dt
                    s['object_angular_velocity'][attached]=next_w[attached]
                if np.any(free):
                    s['object_position'][free]+=dt*s['object_linear_velocity'][free]
                    s['object_quaternion'][free]=unit(qmul(qexp(dt*s['object_angular_velocity'][free]),s['object_quaternion'][free]))
                s['time']+=dt
            s['requested_opening']=opening.copy()
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
        return cost,parts_total,arrays,action_arrays

    def plan(self,state,candidates,*,warm_start=None):
        started=time.perf_counter();c=self.config
        state=reference.validate_state(state);candidates=[reference.validate_candidate(row) for row in candidates]
        if len(candidates)!=8 or len({row['id'] for row in candidates})!=8:raise ValueError('K8 distinct candidates required')
        warm=deepcopy(warm_start or {});committed=warm.get('selected_candidate_id')
        if warm and state['time']<float(warm['time'])-1e-9:raise ValueError('future warm feedback')
        if warm and committed not in {row['id'] for row in candidates}:raise ValueError('clear expired generation explicitly')
        for row in candidates:
            if row['id']==committed and abs(row['deadline']-warm['deadline'])>1e-8:raise ValueError('deadline cannot be pushed forward')
        old=deepcopy(state.get('phase_state',warm.get('phase_state',dict(phase='approach',dwell=0.,lost_hold=0.,updated_at=None))))
        means=np.zeros((8,c.knots,6));std=np.ones_like(means)*np.r_[np.full(3,c.position_std),np.full(3,c.rotation_std)]
        floors=np.r_[np.full(3,c.position_std_floor),np.full(3,c.rotation_std_floor)]
        grid=np.linspace(0.,1.,8);knots=np.linspace(0.,1.,c.knots)
        shifted=deepcopy(warm.get('sequence',[]));shift=0
        if shifted:
            if len(shifted)!=8:raise ValueError('H8 warm sequence required')
            shift=min(8,max(0,int(round((state['time']-float(warm['time']))/c.dt))))
            shifted=shifted[shift:]+[deepcopy(shifted[-1]) for _ in range(shift)]
        if 'knot_means' in warm and list(warm.get('candidate_ids',[]))==[row['id'] for row in candidates]:
            old_means=np.asarray(warm['knot_means']);old_stds=np.asarray(warm['knot_stds'])
            if old_means.shape!=means.shape or old_stds.shape!=std.shape or not np.isfinite(old_means).all() or not np.isfinite(old_stds).all():raise ValueError('invalid warm statistics')
            for mode in range(8):
                for axis in range(6):
                    dense=np.interp(grid,knots,old_means[mode,:,axis]);dense=np.r_[dense[shift:],np.repeat(dense[-1],shift)]
                    means[mode,:,axis]=np.interp(knots,grid,dense)
            std=np.maximum(old_stds,floors)
        if committed in [row['id'] for row in candidates] and shifted:
            mode=next(i for i,row in enumerate(candidates) if row['id']==committed)
            phase=reference.update_phase(state,candidates[mode],old,c)
            nominal=reference._nominal_action(state,candidates[mode],phase,np.zeros(6),c)
            means[mode,:,:3]=np.clip(shifted[0]['tcp_position']-nominal['tcp_position'],-c.max_position_perturbation,c.max_position_perturbation)
        weights=np.array([np.interp(grid,knots,np.eye(c.knots)[i]) for i in range(c.knots)]).T
        rng=np.random.default_rng(c.seed+int(round(state['time']/c.dt)))
        best=None;counts=[];warm_count=0
        for iteration,total in enumerate(c.iteration_samples):
            per=total//8;modes=np.repeat(np.arange(8),per);pieces=[]
            for mode in range(8):
                samples=rng.normal(means[mode],std[mode],size=(per,c.knots,6));samples[0]=means[mode]
                samples[...,:3]=np.clip(samples[...,:3],-c.max_position_perturbation,c.max_position_perturbation)
                samples[...,3:]=np.clip(samples[...,3:],-c.max_rotation_perturbation,c.max_rotation_perturbation)
                pieces.append(samples)
            samples=np.concatenate(pieces);controls=np.einsum('hk,nkj->hnj',weights,samples)
            scores,parts,states,actions=self._rollout_batch(state,candidates,modes,controls,old,committed,shifted if shifted else None)
            win=int(np.argmin(scores))
            if best is None or scores[win]<best['cost']:
                candidate=candidates[modes[win]]
                seq=[{key:values[win,k].copy() if values[win,k].shape else float(values[win,k]) for key,values in actions.items()} for k in range(8)]
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
        if best is None or not np.isfinite(best['cost']):
            minimal={key:budget[key] for key in ('sequences','horizon','virtual_macrosteps','virtual_50ms_substeps','Native_steps')}
            return dict(action=None,sequence=[],selected_candidate_id=None,phase_state=old,warm_start=None,budget=minimal,
                        diagnostics=dict(reason='all_sequences_infeasible',iterations=counts,phase_real_updates=0,wall_seconds=time.perf_counter()-started,real_time_guaranteed=False))
        candidate=next(row for row in candidates if row['id']==best['candidate_id'])
        actual=reference.update_phase(state,candidate,old,c)
        warm_out=dict(time=state['time'],selected_candidate_id=candidate['id'],deadline=candidate['deadline'],phase_state=deepcopy(actual),
                      sequence=deepcopy(best['sequence']),candidate_ids=[row['id'] for row in candidates],knot_means=means.copy(),knot_stds=std.copy())
        return dict(action=deepcopy(best['sequence'][0]),sequence=best['sequence'],selected_candidate_id=best['candidate_id'],
                    phase_state=actual,warm_start=warm_out,cost=best['cost'],cost_terms=best['cost_terms'],predicted_states=best['states'],budget=budget,
                    diagnostics=dict(iterations=counts,warm_start_evaluations_counted=warm_count,phase_real_updates=1,
                                     wall_seconds=time.perf_counter()-started,real_time_guaranteed=False,batch_prototype=True,value_network_used=False))
