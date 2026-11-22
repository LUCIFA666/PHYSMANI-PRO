"""Pure current-observation R10 actor and per-candidate collision binding.

Only the parent owns execution ticks, resets and native labels. This actor
receives RGB-D, public semantic templates and robot proprioception; it never
receives an environment, teacher, native outcome or reset/audit receipt.
"""
from collections import Counter
from dataclasses import asdict

import numpy as np

from ..contracts import Pose, RobotState
from ..reachability import TimedReachability
from .forecasting import ObservedAttachment
from .interaction import (CandidateGenerator, InteractionArtifactError,
    InteractionController, PolicyState, SE3InteractionMPC,
    TRAINUniformFirstCandidateSampler, continuation_fingerprint)
from .mode_observer import PublicCommandSegments, PublicModeObserver
from .public_roles import PublicRoleResolutionError
from .rotation_equivalence import CurrentRotationEquivalence


class PipelineInputError(ValueError):
    pass


def _plain(value):
    if isinstance(value,CurrentRotationEquivalence):
        return value.receipt()
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,dict):
        return {key:_plain(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):
        return [_plain(item) for item in value]
    return value


def _fresh_direct(state, body_id, maximum_age):
    body=state.bodies.get(body_id)
    meta=state.perception_metadata.get('bodies',{}).get(body_id,{})
    return bool(body is not None and body.visible and meta.get('directly_visible') is True
        and meta.get('registration_valid') is True
        and -1e-8 <= state.robot.timestamp-body.timestamp <= maximum_age)


class PhaseCollisionPort:
    """Full sampled robot/held-body checks against forecast public geometry.

    Allowed robot contacts disable target ghosts only for the arm check in the
    shared map backend. Held geometry still checks the complete world. No whole
    target body is globally ignored. Every candidate/active plan rebinds this
    port before any reachability probe can call robot_model.collision_query.
    """
    def __init__(self, task_spec, map_service, robot_model, *, maximum_observation_age,ring_affordance=None):
        self.spec,self.map,self.model=task_spec,map_service,robot_model
        self.required=tuple(map_service.required_dynamic_role_ids)
        self.maximum_observation_age=float(maximum_observation_age)
        self.ring_affordance=ring_affordance
        if not self.required or not callable(getattr(map_service,'check',None)):
            raise PipelineInputError('declare the shared static map and all dynamic body collision roles')
        self.frame,self.bound=None,None
        self.attachments,self.non_exogenous=set(),set()
        self.counts=Counter()
        self.last_geometry_modes,self.last_geometry_reasons={},{}
        self.query_callable=self.check_future
        robot_model.collision_query=self.query_callable

    def configure_frame(self, state, forecast, *, attachment_ids=(), non_exogenous_ids=()):
        if not isinstance(state,PolicyState) or abs(float(forecast.metadata.get('observed_until',state.robot.timestamp))-state.robot.timestamp) > 1e-7:
            raise PipelineInputError('bind current public robot/body state and causal forecast together')
        self.frame=(state,forecast)
        self.attachments,self.non_exogenous=set(attachment_ids),set(non_exogenous_ids)
        self.bound=None

    def _permissions(self, state, phase, reference_id, manipulated_ids):
        allowed=set()
        if phase == 'acquire':
            allowed.update(set(manipulated_ids)&set(self.spec.graspable_roles))
        elif phase == 'press':
            allowed.update(state.role_bindings.get('button',()))
            allowed.update(state.role_bindings.get('plate',()))
        elif phase == 'insert':
            pegs=state.role_bindings.get('selected_peg',())
            if len(pegs) == 1:
                allowed.update(pegs)
        elif phase == 'hang':
            spokes=state.role_bindings.get('selected_spoke',())
            if reference_id in spokes:
                allowed.add(reference_id)
        # In particular, a buzz/frame reference never becomes a contact bypass.
        return tuple(sorted(allowed&set(self.required)))

    def bind_candidate(self, *, state, forecast, phase, reference_id, manipulated_ids, arrival_time):
        if (self.frame is None or self.frame[1] is not forecast
                or state.robot.timestamp != self.frame[0].robot.timestamp
                or not np.isfinite(arrival_time) or arrival_time < state.robot.timestamp-1e-8):
            raise PipelineInputError('candidate collision scene must be rebound to this current frame')
        self.frame=(state,forecast)
        self.bound=dict(phase=phase,reference_id=reference_id,manipulated_ids=tuple(manipulated_ids),
            arrival_time=float(arrival_time),allowed_robot_contact_ids=self._permissions(state,phase,reference_id,manipulated_ids))
        self.counts['candidate_scene_bindings'] += 1

    def _noncolliding_geometry(self):
        declared=getattr(self.map,'noncolliding_geometry',{})
        if not isinstance(declared,dict) or set(declared)-set(self.required):
            raise PipelineInputError('noncolliding geometry must name declared task roles only')
        for body_id,record in declared.items():
            checksum=record.get('source_sha256') if isinstance(record,dict) else None
            if (not isinstance(record,dict) or record.get('provenance')!='TRAIN_geometry_model'
                    or record.get('geometry_rule')!='no owned collidable OR respondable component in TRAIN asset'
                    or not isinstance(checksum,str) or len(checksum)!=64
                    or any(c not in '0123456789abcdef' for c in checksum)):
                raise PipelineInputError('noncolliding role needs its frozen TRAIN geometry declaration: '+body_id)
        return declared

    def _poses(self, future_offset):
        if self.frame is None or self.bound is None or not np.isfinite(future_offset) or future_offset < -1e-8:
            raise PipelineInputError('an explicit candidate/phase bind precedes all collision probes')
        state,forecast=self.frame
        poses,bookkeeping={},[]
        noncolliding=self._noncolliding_geometry()
        for body_id in self.required:
            if body_id in noncolliding:
                continue # TRAIN asset proves no physical collision mesh exists.
            if body_id in self.attachments:
                continue # shared backend uses measured attachment + candidate robot FK
            if body_id in forecast.object_ids:
                poses[body_id]=forecast.at(body_id,state.robot.timestamp+max(0.,float(future_offset)))[0]
                continue
            permitted_plate=(body_id == 'plate' and body_id in self.non_exogenous
                and self.bound['phase'] == 'press' and body_id in self.bound['allowed_robot_contact_ids']
                and not self.attachments and not state.robot.contact
                and _fresh_direct(state,body_id,self.maximum_observation_age))
            if permitted_plate:
                # Only arm-vs-this-plate is explicitly permitted, and no held
                # object exists whose world sweep could depend on its future.
                # This is current controlled-body bookkeeping, not a forecast.
                poses[body_id]=state.bodies[body_id].pose
                bookkeeping.append(body_id)
                continue
            raise PipelineInputError('unknown or unavailable dynamic-body forecast: '+body_id)
        return poses,bookkeeping

    def readiness(self, state, forecast, phase):
        # This preliminary binding grants only the specific publicly known press
        # contacts. Candidate-specific acquisition/insertion permissions follow.
        self.bind_candidate(state=state,forecast=forecast,phase=phase,reference_id='',manipulated_ids=(),
                            arrival_time=state.robot.timestamp)
        try:
            _,bookkeeping=self._poses(0.)
            modes,reasons=self._geometry_modes()
            return dict(ready=True,controlled_body_bookkeeping_ids=bookkeeping,
                TRAIN_noncolliding_geometry_ids=sorted(self._noncolliding_geometry()),
                geometry_modes=modes,geometry_mode_evidence=reasons)
        except (PipelineInputError,ValueError,IndexError) as error:
            return dict(ready=False,reason=str(error),controlled_body_bookkeeping_ids=[])

    def _geometry_modes(self):
        """Select only immutable TRAIN meshes from current public evidence.

        Unknown orientation preserves the body with its all-rotation enclosure.
        Axis yaw sweeps are conditional on the declared 10deg/5mm engineering
        assumptions. A finite ring quotient requires its typed current pose
        and the same CAD certificate as the frozen map asset.
        """
        state,_=self.frame;now=float(state.robot.timestamp)
        entries=getattr(self.map,'estimated_bodies',{})
        modes,reasons={},{}
        for body_id,entry in entries.items():
            if body_id not in self.required:
                raise PipelineInputError('collision geometry names an undeclared actor role')
            receipt=entry.collision_geometry_receipt or {}
            kind=receipt.get('collision_geometry_kind')
            has_fallback=getattr(entry,'fallback_ghost',None) is not None
            body=state.bodies.get(body_id)
            meta=state.perception_metadata.get('bodies',{}).get(body_id,{})
            stamp=meta.get('pose_evidence_until')
            fresh=(body is not None and meta.get('registration_valid') is True
                and isinstance(stamp,(int,float)) and np.isfinite(stamp)
                and abs(float(stamp)-float(body.timestamp))<=1e-7
                and -1e-8<=now-float(stamp)<=self.maximum_observation_age)
            selected=False;reason='current orientation unavailable: retain all-rotation TRAIN enclosure'
            if kind=='rotation_uncertainty_cube':
                selected=True;reason='primary mesh already encloses every body rotation'
            elif kind=='exact_TRAIN_triangles':
                if fresh and meta.get('orientation_observable') is True:
                    selected=True;reason='current full-orientation RGBD or evidenced rigid-group pose'
                elif fresh and body_id=='ring' and self.ring_affordance is not None:
                    from .ring_affordance import VerifiedRingAffordance
                    evidence=meta.get('rotation_equivalence_evidence')
                    same_CAD=(isinstance(evidence,CurrentRotationEquivalence) and
                        evidence.symmetry.certificate['CAD_asset_sha256']==getattr(entry,'source_sha256',None))
                    if same_CAD and isinstance(self.ring_affordance,VerifiedRingAffordance) and self.ring_affordance.permits_geometry(
                            evidence,body_id=body_id,observed_pose=body.pose,observed_timestamp=body.timestamp,
                            now=now,maximum_age=self.maximum_observation_age):
                        selected=True;reason='current typed D4 quotient of the exact frozen TRAIN ring CAD; full R remains false'
            elif kind=='directed_axis_yaw_uncertainty_sweep':
                assumptions=(receipt.get('engineering_pose_error_assumption') is True
                    and receipt.get('engineering_tilt_bound_degrees')==10.
                    and receipt.get('engineering_translation_bound_m')==.005
                    and receipt.get('calibrated_safety_guarantee') is False
                    and receipt.get('current_directed_axis_readiness_required_for_every_collision_query') is True)
                if (fresh and body is not None and body.visible and meta.get('directly_visible') is True
                        and meta.get('axis_observable') is True and meta.get('axis_polarity')=='directed' and assumptions):
                    selected=True;reason='current directed RGBD axis; yaw swept under fixed conditional 10deg/5mm inflation'
            elif not has_fallback:
                # Legacy mock/fixture maps never registered two geometry modes.
                selected=True;reason='single frozen geometry; no runtime geometry variant available'
            if not selected and not has_fallback:
                raise PipelineInputError('required conservative rotation-uncertainty geometry is missing: '+body_id)
            modes[body_id]='primary' if selected else 'rotation_uncertainty'
            reasons[body_id]=dict(reason=reason,primary_geometry_kind=kind,
                current_pose_evidence=bool(fresh),full_orientation_promoted=False,
                engineering_assumption_used=selected and kind=='directed_axis_yaw_uncertainty_sweep')
        self.last_geometry_modes,self.last_geometry_reasons=modes,reasons
        return modes,reasons

    def check_future(self, future_offset):
        self.counts['sampled_collision_queries'] += 1
        try:
            poses,bookkeeping=self._poses(future_offset)
            modes,reasons=self._geometry_modes()
        except (PipelineInputError,ValueError,IndexError,KeyError):
            self.counts['unknown_scene_fail_closed_queries'] += 1
            return True
        self.counts['controlled_plate_bookkeeping_queries'] += bool(bookkeeping)
        # Clock/backend exceptions are infrastructure failures, not an ordinary
        # rejected policy candidate. Only unavailable public predictions above
        # are converted to a conservative collision result.
        kwargs=dict(ignored_object_ids=(),allowed_robot_contact_ids=self.bound['allowed_robot_contact_ids'])
        if modes:kwargs['geometry_modes']=modes
        result=self.map.check(poses,**kwargs)
        for body_id,mode in modes.items():self.counts['geometry_mode_queries:'+body_id+':'+mode]+=1
        if type(result) not in (bool,np.bool_):
            raise PipelineInputError('shared collision backend must return a boolean collision result')
        return bool(result)

    def __call__(self, *, pose, time, phase, observed_bodies, forecast, manipulated_ids):
        if (self.frame is None or forecast is not self.frame[1] or self.bound is None
                or phase != self.bound['phase'] or tuple(manipulated_ids) != self.bound['manipulated_ids']):
            raise PipelineInputError('endpoint collision readiness belongs to a different candidate')
        try:
            _,bookkeeping=self._poses(float(time)-self.frame[0].robot.timestamp)
            modes,reasons=self._geometry_modes()
            return dict(feasible=True,reason='full_path_collision_is_checked_by_timed_reachability',
                evidence=dict(allowed_robot_contact_ids=list(self.bound['allowed_robot_contact_ids']),
                    TRAIN_noncolliding_geometry_ids=sorted(self._noncolliding_geometry()),
                    whole_body_ignored_ids=[],held_world_collision_targets_preserved=True,
                    controlled_body_bookkeeping_ids=bookkeeping,controlled_body_bookkeeping_is_prediction=False,
                    geometry_modes=modes,geometry_mode_evidence=reasons))
        except (PipelineInputError,ValueError,IndexError,KeyError) as error:
            return dict(feasible=False,reason=str(error),evidence={})

    def receipt(self):
        return _plain(dict(**self.counts,required_dynamic_role_ids=self.required,last_binding=self.bound,
            last_geometry_modes=self.last_geometry_modes,last_geometry_mode_evidence=self.last_geometry_reasons,
            TRAIN_noncolliding_geometry=dict(self._noncolliding_geometry()),
            noncolliding_roles_keep_actor_identity_but_require_no_collision_forecast=True,
            per_candidate_bind_required=True,whole_dynamic_body_ignored=False,
            held_vs_world_contact_targets_are_checked=True,live_native_body_pose_read=False,
            controlled_plate_bookkeeping_is_prediction=False))


class PipelineActor:
    def __init__(self, task_spec, templates, tracker, forecast_bridge, library,
                 robot_model, map_service, candidate_config, mpc_config, *, predictor,
                 ranking, descriptions, role_resolver=None, success_model=None,
                 attachment_association_distance_m, mode_observer=None,command_segments=None,
                 continuation_sha256=None,purpose='evaluation',first_candidate_sampler=None,
                 allow_synthetic=False,gripper_timing_config=None,joint_limit_recovery_margin_rad=None,
                 ring_affordance=None,allow_development_ring_affordance=False):
        if (task_spec.task_name != library.task_spec.task_name or predictor not in ('cv','context')
                or ranking not in ('geometry','value') or forecast_bridge.method != predictor
                or not np.isfinite(attachment_association_distance_m) or attachment_association_distance_m <= 0
                or purpose not in ('evaluation','train_collection','validation_collection')):
            raise PipelineInputError('matching fitted task/predictor/control configuration required')
        if first_candidate_sampler is not None and purpose != 'train_collection':
            raise PipelineInputError('evaluation/VAL cannot inject TRAIN candidate exploration')
        if (joint_limit_recovery_margin_rad is None and not allow_synthetic) or (
                joint_limit_recovery_margin_rad is not None and
                (not np.isfinite(joint_limit_recovery_margin_rad) or joint_limit_recovery_margin_rad<=0)):
            raise PipelineInputError('production requires an explicit positive public joint recovery margin')
        self.robot_model=robot_model
        self.joint_limit_recovery_margin_rad=joint_limit_recovery_margin_rad
        lower,upper=np.asarray(robot_model.lower,float),np.asarray(robot_model.upper,float)
        if (lower.shape!=upper.shape or lower.ndim!=1 or np.isnan(lower).any() or np.isnan(upper).any()
                or np.any(lower>=upper) or
                (joint_limit_recovery_margin_rad is not None and
                 np.any(upper-lower<=2*joint_limit_recovery_margin_rad))):
            raise PipelineInputError('public recovery margin must fit inside the native joint intervals')
        self.spec,self.templates,self.tracker,self.bridge=task_spec,tuple(templates),tracker,forecast_bridge
        self.body_ids=tuple(template.role_id for template in self.templates)
        if (len(set(self.body_ids)) != len(self.body_ids)
                or set(self.body_ids) != set(forecast_bridge.body_ids)
                or set(self.body_ids) != set(map_service.required_dynamic_role_ids)):
            raise PipelineInputError('tracker/forecast/map must share all unique declared physical body IDs')
        if task_spec.family in ('insert_onto_rotating_peg','push_moving_button') and role_resolver is None:
            raise InteractionArtifactError('public language/colour target resolver is required for this task')
        self.descriptions=(descriptions,) if isinstance(descriptions,str) else tuple(descriptions)
        if not self.descriptions or any(not isinstance(text,str) for text in self.descriptions):
            raise PipelineInputError('pass public task descriptions, not a native variation selector')
        self.role_resolver,self.map=role_resolver,map_service
        self.mode_observer=mode_observer or PublicModeObserver(self.body_ids,task_spec.graspable_roles,
                                                               attachment_association_distance_m)
        self.command_segments=command_segments or PublicCommandSegments(self.body_ids)
        self.collision_port=PhaseCollisionPort(task_spec,map_service,robot_model,
            maximum_observation_age=candidate_config.maximum_observation_age,ring_affordance=ring_affordance)
        generator=CandidateGenerator(library,TimedReachability(robot_model),self.collision_port,
            candidate_config,allow_synthetic=allow_synthetic,ring_affordance=ring_affordance,
            allow_development_ring_affordance=allow_development_ring_affordance)
        digest=continuation_sha256 or continuation_fingerprint(library,candidate_config,mpc_config,
            gripper_timing_config,joint_limit_recovery_margin_rad)
        self.controller=InteractionController(library,generator,SE3InteractionMPC(mpc_config),
            predictor=predictor,ranking=ranking,continuation_sha256=digest,success_model=success_model,
            allow_synthetic=allow_synthetic,first_candidate_sampler=first_candidate_sampler,
            collection_purpose='train' if purpose == 'train_collection' else None,
            gripper_timing_config=gripper_timing_config)
        self.purpose,self.predictor,self.ranking=purpose,predictor,ranking
        self.previous_states,self.attachments,self.bound_attachments=[],{},set()
        self.last_time,self.episode_start=None,None
        self.last_command=dict(command='hold',public_command_kind='hold')
        self.events,self.last_diagnostics=[],{}
        self.requested_opening=None
        self.joint_limit_recovery_events=[]
        self.recovery_counted_ticks_observed=0
        self._pending_recovery_timestamp=None

    @property
    def first_selection_policy(self):
        return self.controller.first_selection_policy

    def _hold(self,reason):
        phase=(self.controller.library.phase_order[self.controller.phase_index]
               if self.controller.phase_index < len(self.controller.library.phase_order) else 'verify')
        self.last_command=dict(command='hold',phase=phase,reason=reason,public_command_kind='hold')
        self.command_segments.issue('hold')
        return dict(self.last_command)

    def _observation(self,observation,robot,remaining_seconds):
        if not isinstance(observation,dict) or not isinstance(robot,RobotState):
            raise PipelineInputError('actor receives current RGBD dict and public RobotState only')
        if robot.task_success:
            raise PipelineInputError('legacy native task-success channel must remain disabled for policy')
        forbidden=('native','teacher','expert','audit','reset_receipt','success','target_state','variation')
        if any(any(term in str(key).lower() for term in forbidden) for key in observation):
            raise PipelineInputError('native/teacher/reset/outcome fields cannot enter policy observations')
        if (not np.isfinite([robot.timestamp,remaining_seconds]).all() or remaining_seconds < 0
                or abs(float(observation['timestamp'])-robot.timestamp) > 1e-7):
            raise PipelineInputError('actor and current sensor/proprioception clocks disagree')
        if self.last_time is not None and robot.timestamp <= self.last_time+1e-8:
            raise PipelineInputError('act must receive one new counted sensor frame per invocation')
        self.last_time=float(robot.timestamp)
        if self.episode_start is None:
            self.episode_start=self.last_time

    def _roles(self,state):
        visible={body for body in self.body_ids if _fresh_direct(state,body,
            self.controller.generator.config.maximum_observation_age)}
        bindings={body:(body,) for body in self.body_ids}
        cups=tuple(sorted(body for body in visible if body.startswith('cup') and body[3:].isdigit()))
        spokes=tuple(sorted(body for body in visible if body.startswith('spoke') and body[5:].isdigit()))
        if cups:
            bindings['selected_cup']=cups
        if spokes:
            bindings['selected_spoke']=spokes
        return bindings

    def _attachments(self,state,modes,invalid):
        now=float(state.robot.timestamp)
        for body_id in self.body_ids:
            if modes.get(body_id) != 'attached' or body_id in invalid or not state.robot.contact:
                self.attachments.pop(body_id,None)
                continue
            if body_id not in self.attachments and _fresh_direct(state,body_id,
                    self.controller.generator.config.maximum_observation_age):
                body=state.bodies[body_id]
                local=Pose(state.robot.pose.rotation.T@(body.pose.position-state.robot.pose.position),
                           state.robot.pose.rotation.T@body.pose.rotation)
                self.attachments[body_id]=ObservedAttachment(body_id,local,now)

    def _synchronise_attachments(self,state):
        current=self.bridge.attachment_updates()
        for body_id in self.bound_attachments-set(current):
            self.map.bind_estimated_attachment(body_id,None,provenance='current_rgbd_robot_proprioception')
            self.controller.measured_attachments.pop(body_id,None)
        for body_id,pose in current.items():
            self.map.bind_estimated_attachment(body_id,pose,provenance='current_rgbd_robot_proprioception')
            item=self.attachments.get(body_id)
            if item is None:
                raise PipelineInputError('bridge cannot create an attachment without a causal actor measurement')
            metadata=state.perception_metadata.get('bodies',{}).get(body_id,{})
            self.controller.measured_attachments[body_id]=dict(local_pose_xyzw=pose.xyzw().tolist(),
                observed_at=float(item.observed_at),
                source='current_RGBD_body_pose_and_robot_proprioception_at_public_contact',
                orientation_observable=metadata.get('orientation_observable') is True,
                rotation_gauge=metadata.get('rotation_gauge'))
        self.bound_attachments=set(current)

    def act(self,observation,robot,remaining_seconds):
        self._observation(observation,robot,remaining_seconds)
        if self.requested_opening is None: self.requested_opening=float(robot.gripper_open)
        if self._pending_recovery_timestamp is not None:
            elapsed=float(robot.timestamp-self._pending_recovery_timestamp)
            if elapsed>=self.controller.generator.config.dt-1e-7:
                self.recovery_counted_ticks_observed+=1
                self.joint_limit_recovery_events[-1]['next_observed_timestamp']=float(robot.timestamp)
                self.joint_limit_recovery_events[-1]['next_observed_elapsed_seconds']=elapsed
                self._pending_recovery_timestamp=None
        modes,invalid=self.mode_observer.observe(self.previous_states,robot)
        states,metadata=self.tracker.observe(observation,modes=modes)
        bodies={body.object_id:body for body in states}
        if len(bodies) != len(states) or set(bodies)-set(self.body_ids):
            raise PipelineInputError('tracker returned unknown/duplicate semantic body identities')
        state=PolicyState(robot,bodies,metadata,{})
        state=PolicyState(robot,bodies,metadata,self._roles(state))
        self.previous_states=list(states)
        self._attachments(state,modes,set(invalid))
        observed=self.bridge.observe(states,metadata,robot,modes=modes,invalid_mode_ids=invalid,
            non_exogenous_ids=self.command_segments.non_exogenous,attachments=self.attachments)
        self._synchronise_attachments(state)
        forecast=self.bridge.predict(robot,horizon=self.bridge.config.maximum_forecast_seconds,
                                     dt=self.controller.generator.config.dt)
        self.last_diagnostics=dict(observed_until=float(robot.timestamp),mode_estimates=dict(modes),
            invalid_mode_ids=sorted(invalid),forecast_observation=observed,
            forecast_metadata=forecast.metadata,estimated_attachment_ids=sorted(self.bound_attachments))
        margin=self.joint_limit_recovery_margin_rad
        if margin is not None:
            lower,upper=np.asarray(self.robot_model.lower,float),np.asarray(self.robot_model.upper,float)
            measured=np.asarray(robot.joints,float)
            if measured.shape!=lower.shape or not np.isfinite(measured).all():
                raise PipelineInputError('finite public measured joint positions matching the native bounds required')
            low,high=measured<lower,measured>upper
            indices=np.flatnonzero(low|high)
            if len(indices):
                if remaining_seconds<self.controller.generator.config.dt-1e-8:
                    return self._hold('episode_budget_exhausted_joint_recovery_unexecuted')
                reference=measured.copy();reference[low]=lower[low]+margin;reference[high]=upper[high]-margin
                event=dict(time=float(robot.timestamp),violating_joint_indices=indices.tolist(),
                    measured_joints=measured.tolist(),requested_joints=reference.tolist(),
                    margin_rad=float(margin),preserved_requested_opening=float(self.requested_opening),
                    remaining_episode_seconds=float(remaining_seconds),native_state_reset=False,
                    actual_motor_tracking_guaranteed=False,trajectory_collision_verified=False,
                    planning_probe_called=False,candidate_or_critic_selection_called=False,
                    each_command_requires_one_counted_tick=True)
                self.joint_limit_recovery_events.append(event);self._pending_recovery_timestamp=float(robot.timestamp)
                self.last_diagnostics['joint_limit_recovery']=event
                self.command_segments.issue('joint_trajectory')
                self.last_command=dict(command='joint_trajectory',joints=reference,opening=float(self.requested_opening),
                    phase='joint_limit_recovery',public_command_kind='joint_trajectory',
                    reason='public_measured_joint_outside_native_interval',joint_limit_recovery=True,
                    bounded_reference_recovery=True,trajectory_collision_verified=False,
                    native_state_reset=False,each_command_requires_one_counted_tick=True)
                self.events.append(dict(time=float(robot.timestamp),phase='joint_limit_recovery',
                    command='joint_trajectory',public_command_kind='joint_trajectory',
                    reason=self.last_command['reason'],native_outcome_used_by_actor=False))
                return dict(self.last_command)
        if invalid or any(modes.get(body) not in ('external','attached','released') for body in self.body_ids):
            return self._hold('unknown_or_ambiguous_public_motion_mode')
        if self.role_resolver is not None:
            try:
                bindings,role_report=self.role_resolver.resolve(self.descriptions,state,observation,self.templates)
                state=PolicyState(robot,bodies,metadata,bindings)
                self.last_diagnostics['public_role_resolution']=role_report
            except PublicRoleResolutionError as error:
                self.last_diagnostics['public_role_resolution_error']=str(error)
                return self._hold('public_target_role_unavailable_or_ambiguous')
        # Committed cup/spoke decisions survive phase changes and are never
        # replaced by a newly visible neighbour merely because it is closer.
        state=PolicyState(robot,bodies,metadata,{**state.role_bindings,**self.controller.committed_roles})
        self.collision_port.configure_frame(state,forecast,attachment_ids=self.bound_attachments,
            non_exogenous_ids=self.command_segments.non_exogenous)
        if self.controller.phase_index >= len(self.controller.library.phase_order):
            return self._hold('trained_skills_completed_parent_owns_native_outcome')
        phase=self.controller.library.phase_order[self.controller.phase_index]
        ready=self.collision_port.readiness(state,forecast,phase)
        self.last_diagnostics['collision_scene_readiness']=ready
        if not ready['ready']:
            return self._hold('common_warmup_or_missing_public_collision_forecast')
        command=self.controller.step(state,forecast,remaining_seconds)
        kind='press' if command['command'] == 'joint_trajectory' and command.get('phase') == 'press' else command['command']
        if kind not in ('press','joint_trajectory','hold'):
            raise PipelineInputError('controller emitted an unsupported public actuator command')
        self.command_segments.issue(kind)
        self.last_command={**command,'public_command_kind':kind}
        if command['command']=='joint_trajectory': self.requested_opening=float(command['opening'])
        self.last_diagnostics['controller']=dict(self.controller.last_diagnostics)
        self.events.append(dict(time=float(robot.timestamp),phase=command.get('phase'),
            command=command['command'],public_command_kind=kind,
            history_frames=dict(forecast.metadata.get('history_frames',{})),
            reason=command.get('reason'),native_outcome_used_by_actor=False))
        return dict(self.last_command)

    def metrics_report(self):
        return _plain(dict(predictor=self.predictor,ranking=self.ranking,purpose=self.purpose,
            first_selection_policy=self.first_selection_policy,
            TRAIN_first_candidate_sampler=(self.controller.first_candidate_sampler.metadata
                if self.controller.first_candidate_sampler is not None else None),
            continuation_selection_policy='single_value_selection_then_frozen_geometry',
            continuation_sha256=self.controller.continuation_sha256,
            interaction_library_sha256=self.controller.library.sha256,
            TRAIN_interaction_semantic_priors=[dict(phase=template.phase,reference_role=template.reference_role,
                manipulated_role=template.manipulated_role,anchor_id=template.anchor_id)
                for template in self.controller.library.templates],
            public_events=self.events,observable_skill_events=self.controller.events,
            candidate_selection_events=self.controller.selection_events,
            continuation_attempts=self.controller.attempts.to_report(),
            gripper_timing_configuration=(asdict(self.controller.gripper_timing_config)
                if self.controller.gripper_timing_config is not None else None),
            gripper_timing_events=self.controller.gripper_timing_events,
            observed_gripper_open_is_boolean_not_continuous_width=True,
            joint_limit_recovery_margin_rad=self.joint_limit_recovery_margin_rad,
            joint_limit_recovery_events=self.joint_limit_recovery_events,
            recovery_commands_emitted=len(self.joint_limit_recovery_events),
            recovery_counted_ticks_confirmed_by_next_public_observation=self.recovery_counted_ticks_observed,
            recovery_final_command_may_require_parent_dispatch_receipt=self._pending_recovery_timestamp is not None,
            recovery_is_physical_counted_control_not_state_reset=True,
            latest_requested_opening=self.requested_opening,
            last_diagnostics=self.last_diagnostics,collision=self.collision_port.receipt(),
            native_state_is_policy_input=False,actor_advances_physics=False))
