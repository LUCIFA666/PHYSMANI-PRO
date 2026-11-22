"""Public CV/Context forecast frontend and TRAIN button pose intents for RML.

Reuse trained perception, forecasting, role binding and author command models.
Do not run the old Hermite/reference managers or their local actuator patches.
The first integration is explicitly limited to the two original button tasks.
"""
from copy import deepcopy
import numpy as np
from .interaction import PolicyState, _reference_at, _world_target
from .pipeline import PipelineActor, PipelineInputError
from .public_roles import PublicRoleResolutionError


class PublicModelFrontend:
    def __init__(self, actor):
        if actor.predictor not in ('cv','context') or actor.ranking!='geometry':
            raise ValueError('exactly the two geometry predictors; no learned value')
        self.actor=actor
        self.state=self.forecast=self.observation=None
        self.role_ready=False

    def observe(self, observation, robot):
        a=self.actor
        PipelineActor._observation(a,observation,robot,max(0.,10.-robot.timestamp))
        if a.requested_opening is None:
            a.requested_opening=float(robot.gripper_open)
        modes,invalid=a.mode_observer.observe(a.previous_states,robot)
        states,metadata=a.tracker.observe(observation,modes=modes)
        bodies={body.object_id:body for body in states}
        if len(bodies)!=len(states) or set(bodies)-set(a.body_ids):
            raise PipelineInputError('unknown or duplicate observed body identity')
        state=PolicyState(robot,bodies,metadata,{})
        state=PolicyState(robot,bodies,metadata,a._roles(state))
        a.previous_states=list(states)
        a._attachments(state,modes,set(invalid))
        observed=a.bridge.observe(states,metadata,robot,modes=modes,invalid_mode_ids=invalid,
            non_exogenous_ids=a.command_segments.non_exogenous,attachments=a.attachments)
        a._synchronise_attachments(state)
        forecast=a.bridge.predict(robot,horizon=a.bridge.config.maximum_forecast_seconds,
                                dt=a.controller.generator.config.dt)
        self.role_ready=not invalid and all(modes.get(k) in ('external','attached','released') for k in a.body_ids)
        if a.role_resolver is not None and self.role_ready:
            try:
                bindings,_=a.role_resolver.resolve(a.descriptions,state,observation,a.templates)
                state=PolicyState(robot,bodies,metadata,bindings)
            except PublicRoleResolutionError:
                self.role_ready=False
        state=PolicyState(robot,bodies,metadata,{**state.role_bindings,**a.controller.committed_roles})
        a.collision_port.configure_frame(state,forecast,attachment_ids=a.bound_attachments,
            non_exogenous_ids=a.command_segments.non_exogenous)
        a.last_diagnostics=dict(observed_until=float(robot.timestamp),mode_estimates=dict(modes),
            invalid_mode_ids=sorted(invalid),forecast_observation=observed,
            forecast_metadata=forecast.metadata,old_reference_controller_executed=False)
        self.state,self.forecast,self.observation=state,forecast,observation
        return state,forecast

    def ready(self, phase):
        if self.state is None or self.forecast is None or not self.role_ready:
            return False
        invalid=self.actor.last_diagnostics['invalid_mode_ids']
        modes=self.actor.last_diagnostics['mode_estimates']
        if invalid or any(modes.get(k) not in ('external','attached','released') for k in self.actor.body_ids):
            return False
        return bool(self.actor.collision_port.readiness(self.state,self.forecast,phase)['ready'])

    def issued(self, phase, opening):
        self.actor.command_segments.issue('press' if phase=='press' else 'joint_trajectory')
        self.actor.requested_opening=float(opening)


class ButtonGeometryIntents:
    def __init__(self, frontend):
        self.frontend=frontend
        c=frontend.actor.controller
        if (c.library.task_spec.task_name not in ('push_moving_button','push_moving_button_high_speed')
                or tuple(c.library.phase_order)!=('approach','press')
                or c.library.model['source_stage_guard']['approach_to_press_distance_m']!=.3):
            raise ValueError('original TRAIN author button stages and0.3m guard required')
        self.phase_index=0
        self.selected=None
        self.events=[]

    def candidates(self, phase):
        f=self.frontend;a=f.actor;c=a.controller;state,forecast=f.state,f.forecast
        now=float(state.robot.timestamp)
        rows=[]
        bindings={**state.role_bindings,**c.committed_roles}
        for template in c.library.for_phase(phase):
            if template.reference_kind not in ('translation_only','observed_se3'):
                raise ValueError('button target must use its observed public reference')
            for reference_id in bindings.get(template.reference_role,()):
                try:
                    current=c.generator.observed_reference(state,reference_id,template,c.origin)
                    lookahead=.5
                    query=now+lookahead+template.lag_seconds
                    if query>forecast.times[-1]+1e-8:
                        continue
                    ref,velocity,angular=_reference_at(forecast,reference_id,template,query,c.origin)
                    target=_world_target(ref,velocity,angular,template,1.)[0]
                    present=_world_target(current.pose,current.linear_velocity,current.angular_velocity,template,1.)[0]
                    # Geometry chooses a pose intent. The original PhysMani
                    # planner performs IK/path generation; no old Hermite
                    # reachability veto is used as an RML admission proxy.
                    score=-float(np.linalg.norm(target.position-state.robot.pose.position))
                    rows.append(dict(template=template,reference_id=reference_id,
                        manipulated_ids=(reference_id,),pose=target,present_pose=present,
                        opening=float(template.at(1.)[3]),score=score,
                        selected_at=now,predicted_reference_time=query,lookahead_seconds=lookahead))
                except (KeyError,ValueError,IndexError):
                    continue
        return sorted(rows,key=lambda r:(-r['score'],r['reference_id'],r['template'].anchor_id))

    def select(self):
        f=self.frontend;a=f.actor;c=a.controller
        phase=('approach','press')[self.phase_index]
        if not f.ready(phase):
            return None
        candidates=self.candidates(phase)
        if not candidates:
            return None
        chosen=candidates[0]
        if phase=='approach':
            distance=float(np.linalg.norm(f.state.robot.pose.position-chosen['present_pose'].position))
            if distance<=.3:
                self.phase_index=1;c.phase_index=1
                c.committed_roles[chosen['template'].reference_role]=(chosen['reference_id'],)
                self.events.append(dict(time=float(f.state.robot.timestamp),
                    event='original_author_distance_guard_enter_press',distance_m=distance,
                    public_press_completion_claimed=False))
                candidates=self.candidates('press')
                if not candidates:
                    return None
                chosen=candidates[0];phase='press'
        chosen=dict(chosen,phase=phase)
        self.selected=chosen
        action=np.r_[chosen['pose'].xyzw(),chosen['opening']]
        self.events.append(dict(time=chosen['selected_at'],event='geometry_RML_pose_intent',
            phase=phase,reference_id=chosen['reference_id'],action=action.tolist(),
            predicted_reference_time=chosen['predicted_reference_time'],
            predictor=a.predictor,source_model_canonical_sha256=c.library.sha256,
            selected_by='public_predicted_TCP_distance_then_stable_identity',
            native_outcome_or_future_observation_used=False))
        f.issued(phase,chosen['opening'])
        return action

    def admit_pose(self, action):
        chosen=self.selected
        if chosen is None or self.frontend.state is None:
            return False,dict(reason='current_public_intent_required')
        now=float(self.frontend.state.robot.timestamp)
        accepted=(abs(now-chosen['selected_at'])<1e-8
                  and np.array_equal(action,np.r_[chosen['pose'].xyzw(),chosen['opening']]))
        return bool(accepted),dict(reason='same_current_public_pose_intent' if accepted else 'intent_mismatch',
            selected_at=chosen['selected_at'],predicted_reference_time=chosen['predicted_reference_time'])

    def admit_joint(self, joints):
        f=self.frontend;state,forecast=f.state,f.forecast;chosen=self.selected
        if state is None or chosen is None:
            return False,dict(reason='current_public_forecast_required')
        a=f.actor;c=a.controller;model=a.robot_model;now=float(state.robot.timestamp)
        q=np.asarray(joints,float)
        if q.shape!=state.robot.joints.shape or not np.isfinite(q).all():
            return False,dict(reason='invalid_RML_joint_target')
        if np.any(q<model.lower) or np.any(q>model.upper):
            return False,dict(reason='RML_target_outside_original_native_interval')
        cutoff=forecast.metadata.get('observed_until')
        if type(cutoff) not in (int,float) or abs(float(cutoff)-now)>1e-8:
            return False,dict(reason='current_public_forecast_required')
        c.generator.bind_candidate_scene(state,forecast,chosen['phase'],chosen['reference_id'],
            chosen['manipulated_ids'],now+.05)
        count=max(2,int(np.ceil(np.max(abs(q-state.robot.joints))/.03))+1)
        for fraction in np.linspace(0.,1.,count):
            sample=state.robot.joints+fraction*(q-state.robot.joints)
            if model.collision(sample,float(fraction)*.05):
                return False,dict(reason='public_forecast_collision_on_RML_target_sweep',samples=count,
                    failed_fraction=float(fraction),time=now,
                    sweep_is_not_an_actual_motor_trajectory_certificate=True)
        return True,dict(reason='public_forecast_RML_target_sweep_clear',samples=count,
            maximum_joint_sample_spacing_rad=.03,forecast_observed_until=float(cutoff),
            actual_acceleration_bound_not_claimed=True)

    def receipt(self):
        return dict(schema='r10_TRAIN_button_geometry_pose_intents_for_PhysMani_RML_v518',
            events=deepcopy(self.events),native_success_owned_by_parent=True,
            old_Hermite_reference_managers_executed=False,
            all_physical_motion_checked_separately=True,formal_VAL_TEST=False)
