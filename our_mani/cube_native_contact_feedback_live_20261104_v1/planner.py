"""Native contact owns attachment transitions; original CEM response/limits.

Only while an actual unfinished close has no Native membership, the predicted
close phase remains closed. It is not reopened by visual/geometric timeout.
Actual completed-close/nonmembership decides a real miss in the coordinator.
"""
from copy import deepcopy
from ..local_mpc_control_20261103.lazy_free_joint_goal_cem_v72 import *
from ..local_mpc_control_20261103.lazy_free_joint_goal_cem_v72 import _LazyJointGoalCEMPlannerV72,_digest
from ..local_mpc_control_20261103.public_candidate_local_hold_v66 import LocalHoldPlannerV66,_assert_current_clock
from ..local_mpc_control_20261103.hold_local_nominal_prototype_20261103 import LocalHoldNominalPlanner
from ..local_mpc_control_20261103.hold_orientation_guard_prototype_20261103 import HoldMicroOrientationGuard


class NativePendingForecastPlannerV1(LazyQualifiedForecastJointGoalCEMPlannerV72):
    def plan(self,state,candidates,*,warm_start=None):
        if (state.get('Native_contact_feedback_used')is not True or state.get('Native_held_used')is not True
                or state.get('Native_object_state_used')is not False or state.get('Native_success_used')is not False
                or state.get('held')is not False or state.get('grasp_motion_mode')!='FREE_GRU'):
            raise ValueError('Truthful actual Native nonmembership with public FREE kinematics required')
        if self._caller_state is not None:raise ValueError('No nested current feedback')
        self._native_pending_close=state['actuator_state']['gripper_target']==0.
        self._caller_state=state;self._caller_sha=_digest(state)
        try:
            self._assert_caller_current()
            # Enter the unchanged forecast and top8 numeric planner below the
            # old public-only provenance wrapper. No provenance is fabricated.
            result=ForecastJointGoalCEMPlanner.plan(self,state,candidates,warm_start=warm_start)
            self._assert_caller_current()
            result['diagnostics']['Native_contact_pending_protocol']=dict(
                actual_unfinished_close_kept_closed=self._native_pending_close,
                visual_timeout_or_probe_failure_can_reopen=False,Native_contact_feedback_used=True,
                Native_feedback_sha256=self._caller_sha,CEM_budget_and_costs_unchanged=True,
                original_top8_and_final_action_checks_retained=True)
            return result
        finally:self._caller_state=None;self._caller_sha=None

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
            change=mask&bad&(~np.full(n,self._native_pending_close));phases[change]=1;dwell[change]=0.;reopening[change]=True
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

class NativeContactHoldPlannerV1(LocalHoldPlannerV66):
    def _prepare_nominals(self,initial,candidates,old):
        super()._prepare_nominals(initial,candidates,old)
        # Retain numerical reuse while correcting its former public-proof label.
        self.fresh_nominal_qualification.update(original_numeric_ABI_mode='ATTACHED_PUBLIC',
            mode='NATIVE_CARRY',actual_Native_membership_used=True,public_attachment_qualified=False)

    def plan(self,state,candidates,*,warm_start=None):
        if (state.get('Native_contact_feedback_used')is not True or state.get('Native_held_used')is not True
                or state.get('Native_object_state_used')is not False or state.get('held')is not True
                or state.get('grasp_motion_mode')!='NATIVE_CARRY'):
            raise ValueError('Actual membership plus explicitly estimated carry geometry required')
        _assert_current_clock(self.backend,state['time'])
        guard=HoldMicroOrientationGuard(self._base_execution,state,maximum_rotation_rad=.04)
        self.execution_model=guard;self._actual_hold_state=deepcopy(state)
        try:
            # Direct numerical reuse, not the old V69 public-only certificate.
            result=LocalHoldNominalPlanner.plan(self,state,candidates,warm_start=None)
        finally:self.execution_model=self._base_execution
        result['diagnostics']['Native_contact_hold']=dict(Native_contact_feedback_used=True,
            Native_target_pose_used=False,carry_pose_is_public_or_proprioceptive_estimate=True,
            old_visual_attachment_authentication_bypassed_explicitly=True,
            old_V69_public_only_response_seed_not_used=True,original256_CEM_and_response_models=True,
            maximum_local_nominal_translation_m=.012,maximum_predicted_micro_rotation_rad=.04,
            fresh_nominal_qualification=deepcopy(self.fresh_nominal_qualification))
        return result
