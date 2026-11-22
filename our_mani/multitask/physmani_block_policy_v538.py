"""Public block perception/Context and skill goals for the common author RML.

The historical R09 executor/results remain immutable. This new policy uses
the same RGBD box prior, TRAIN marker calibration and grasp/lift geometry.
It never invokes the old one-tick Hermite/MPC actuator or a learned value.
"""
from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
from ..contracts import Pose
from ..benchmark_perception import RedBoxTracker,box_surface
from ..goal_perception import RedGoalMarker
from ..local_capture import LocalConfig
from ..dynamics import ConstantVelocityDynamics
from .physmani_skill_policy_v533 import errors
from .block_held_support_v298 import supported_box,held_collision
from .safe_native_mesh import read_native_mesh
from .supported_sphere_attachment_v10 import horizontal_rectangle
from .joint16_block_context_v97 import Joint16BlockContext


class TerminalBlockView:
    """Parent-only terminal interface; remove the legacy outcome policy field."""
    def __init__(self,env):object.__setattr__(self,'env',env)
    def __getattr__(self,key):return getattr(self.env,key)
    def __setattr__(self,key,value):setattr(self.env,key,value)
    @property
    def _native_terminal_latched(self):return bool(self.env._native_success_latched)
    @property
    def _native_failure_latched(self):return False
    @property
    def _native_terminal_failure_latched(self):return False
    def _before_action(self):
        if self._native_terminal_latched:raise RuntimeError('native terminal block cannot accept another action')
        self.env.ledger.before()
    def observe(self):
        observation,robot=self.env.observe()
        return observation,replace(robot,task_success=False)


class PublicBlockCollision:
    """Original immutable map and supported RGBD held transform, no live poses."""
    def __init__(self,env):
        from pyrep.const import ObjectType
        self.env=env;self.service=env.map_service;self.held_local=None
        self.attempted=False;self.events=[];self.queries=0
        matches=[g for g,info in zip(self.service.ghosts,self.service.map) if info['name']=='workspace']
        if len(matches)!=1:raise ValueError('one immutable block workspace support required')
        v,f,_=read_native_mesh(matches[0]);self.plane=horizontal_rectangle(v,f,matches[0].get_matrix())
        self.vertices=np.asarray(read_native_mesh(self.service.target)[0],float).reshape(-1,3)
        prior=RedBoxTracker().prior;half=np.asarray(prior.dimensions)/2
        if not (np.allclose(self.vertices.min(0),-half,atol=1e-5,rtol=0)
                and np.allclose(self.vertices.max(0),half,atol=1e-5,rtol=0)):
            raise ValueError('native map prototype must match the original public5cm box prior')
        self.gripper_handles={x.get_handle() for x in env.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE,exclude_base=False)}
        self.arm_shapes=[x for x in env.arm.get_objects_in_tree(object_type=ObjectType.SHAPE,exclude_base=False) if x.is_collidable()]
        if not self.gripper_handles or not self.arm_shapes:raise ValueError('complete arm/gripper map ownership required')

    def observe_contact(self,robot,raw,prior,own_close):
        if not robot.contact:
            self.held_local=None
            return
        if self.attempted or not own_close:return
        self.attempted=True
        if raw is None or raw.object_id!='target' or abs(raw.timestamp-robot.timestamp)>1e-8:
            self.events.append(dict(time=float(robot.timestamp),accepted=False,reason='first_contact_current_RGBD_unavailable'))
            return
        pose,proof=supported_box(raw.pose,raw.points_world,self.vertices,self.plane,prior)
        if proof['accepted']:
            self.held_local=Pose(robot.pose.inverse_apply(pose.position),robot.pose.rotation.T@pose.rotation)
        self.events.append(dict(proof,time=float(robot.timestamp),source='first_own_close_contact_current_RGBD'))

    def check(self,joints,future_offset,frontend):
        self.queries+=1
        model=self.env.robot_model;previous=model.collision_query
        if frontend.robot.contact and self.held_local is not None:
            def query(offset):
                tool=Pose.from_xyzw(self.env.arm.get_tip().get_pose());local=self.held_local
                return held_collision(self.service,Pose(tool.apply(local.position),tool.rotation@local.rotation),
                                      self.arm_shapes,self.gripper_handles)
        elif frontend.forecast is not None:
            def query(offset):
                pose=frontend.forecast.at('target',frontend.robot.timestamp+offset)[0]
                return self.service.check(pose)
        else:return True
        model.collision_query=query
        try:return bool(model.collision(joints,future_offset))
        finally:model.collision_query=previous

    def receipt(self):
        return dict(first_contact_support_events=deepcopy(self.events),collision_queries=self.queries,
            held_target_vs_arm_and_static_obstacles_checked=True,gripper_only_contact_exemption=True,
            original_held_support_geometry_reused=True)


class BlockPublicFrontend:
    def __init__(self,root,config,task,method,model,collision):
        self.tracker=RedBoxTracker(complete_geometry=False);self.model=model;self.collision=collision
        self.config=config;self.method=method;self.task=task
        self.marker=RedGoalMarker(json.loads((root/config['goal_calibration']).read_text())['offset_train_calibrated_m'])
        self.dynamics=(Joint16BlockContext(root/config['context_checkpoint']['path'],task=task,root=root)
                       if method=='context_geometry' else ConstantVelocityDynamics())
        self.robot=self.object=self.forecast=self.goal_pose=None
        self.close_confirmed=False;self.frames=0;self.events=[];self.last_time=None

    def observe(self,observation,robot):
        if robot.task_success or abs(float(observation['timestamp'])-robot.timestamp)>1e-8:
            raise ValueError('current public block RGBD/proprioception only')
        if self.last_time is not None and robot.timestamp<=self.last_time+1e-8:
            raise ValueError('one new counted public frame required')
        self.last_time=float(robot.timestamp);self.robot=robot;self.frames+=1
        if self.goal_pose is None:
            try:self.goal_pose,_=self.marker.observe(observation)
            except ValueError:pass
        raw=None
        try:raw,diagnostic=self.tracker.observe(observation)
        except ValueError as error:diagnostic=dict(public_tracker_unavailable=str(error))
        self.collision.observe_contact(robot,raw,self.tracker.prior,self.close_confirmed)
        self.object=None if raw is None else replace(raw,points_world=box_surface(raw.pose,self.tracker.prior.dimensions))
        if robot.contact and self.collision.held_local is not None:
            local=self.collision.held_local
            from ..contracts import ObjectState
            pose=Pose(robot.pose.apply(local.position),robot.pose.rotation@local.rotation)
            self.object=ObjectState('target',robot.timestamp,pose,box_surface(pose,self.tracker.prior.dimensions),
                robot.linear_velocity,robot.angular_velocity,1.,True)
        self.forecast=None
        if isinstance(self.dynamics,Joint16BlockContext):
            self.dynamics.set_public_motion_state(timestamp=robot.timestamp,
                gripper_open=bool(robot.gripper_open),contact=bool(robot.contact))
        if raw is not None and not robot.contact:
            obj=replace(raw,points_world=box_surface(raw.pose,self.tracker.prior.dimensions))
            if hasattr(self.dynamics,'observe'):self.dynamics.observe([obj])
            if self.frames>=self.config['warmup_ticks']:
                try:self.forecast=self.dynamics.predict([obj],self.config['forecast_seconds'],self.config['dt'])
                except ValueError as error:diagnostic['forecast_unavailable']=str(error)
        self.events.append(dict(time=float(robot.timestamp),public_contact=bool(robot.contact),
            target_observed=raw is not None,held_transform=self.collision.held_local is not None,
            causal_forecast_available=self.forecast is not None,diagnostic=diagnostic))


class BlockGeometryIntents:
    def __init__(self,frontend):
        self.frontend=frontend;self.local=LocalConfig(dt=frontend.config['dt'],
            max_speed=frontend.config['local_max_speed'],max_acceleration=frontend.config['local_max_acceleration'])
        self.grasp=Pose.from_xyzw([0.,0.,self.local.grasp_offset_z,0.,1.,0.,0.])
        self.phase='approach';self.selected=None;self.close_witness=None;self.lift_goal=None
        self.transport_goal=None;self.events=[];self.gripper_events=[]

    def current_grasp(self):
        f=self.frontend;obj=f.object
        if obj is None or abs(obj.timestamp-f.robot.timestamp)>1e-8:return None
        return Pose(obj.pose.apply(self.grasp.position),obj.pose.rotation@self.grasp.rotation)

    def select(self):
        f=self.frontend;r=f.robot
        if r is None or f.goal_pose is None or f.frames<f.config['warmup_ticks']:return None
        if self.close_witness is not None and r.contact and f.collision.held_local is not None and self.phase in ('approach','acquire'):
            self.phase='lift';self.lift_goal=Pose(r.pose.position+[0,0,self.local.lift_height],r.pose.rotation)
            local=f.collision.held_local
            self.transport_goal=Pose(f.goal_pose.position-f.goal_pose.rotation@local.position,f.goal_pose.rotation)
            self.events.append(dict(time=float(r.timestamp),event='own_close_and_supported_public_capture_enter_lift',native_success_claimed=False))
        if self.phase in ('lift','transport'):
            if not r.contact or f.collision.held_local is None:return None
            if self.phase=='lift' and np.linalg.norm(r.pose.position-self.lift_goal.position)<=.015:
                self.phase='transport'
            target=self.lift_goal if self.phase=='lift' else self.transport_goal;opening=0.;query=None
        else:
            if f.forecast is None:return None
            query=float(r.timestamp)+.5
            if query>f.forecast.times[-1]+1e-8:return None
            pose,_,_=f.forecast.at('target',query)
            target=Pose(pose.apply(self.grasp.position),pose.rotation@self.grasp.rotation)
            if self.phase=='approach':
                current=self.current_grasp()
                high=target.position.copy();high[2]=f.config['preposition_height']
                preposition=Pose(high,target.rotation)
                present=None if current is None else current.position.copy()
                if present is not None:present[2]=f.config['preposition_height']
                if present is not None and np.linalg.norm(r.pose.position-present)<=.1:self.phase='acquire'
                else:target=preposition
            opening=1. if self.phase=='approach' else 0.
        action=np.r_[target.xyzw(),opening]
        self.selected=dict(pose=target,opening=opening,selected_at=float(r.timestamp),phase=self.phase,action=action.copy())
        self.events.append(dict(time=float(r.timestamp),phase=self.phase,event='public_block_RML_pose_intent',
            action=action.tolist(),predicted_reference_time=query,method=f.method,native_state_or_outcome_read=False))
        return action

    def admit_pose(self,action):
        g=self.selected;f=self.frontend
        accepted=g is not None and abs(g['selected_at']-f.robot.timestamp)<1e-8 and np.array_equal(action,g['action'])
        return bool(accepted),dict(reason='same_current_public_block_pose_intent')

    def admit_joint(self,q):
        f=self.frontend;r=f.robot;model=f.model;q=np.asarray(q,float)
        if q.shape!=r.joints.shape or not np.isfinite(q).all() or np.any(q<model.lower) or np.any(q>model.upper):
            return False,dict(reason='original_native_joint_interval')
        if not r.contact and (f.forecast is None or abs(float(f.forecast.times[0])-r.timestamp)>1e-8):
            return False,dict(reason='current_public_block_forecast_required')
        count=max(2,int(np.ceil(np.max(abs(q-r.joints))/.03))+1)
        for fraction in np.linspace(0.,1.,count):
            if f.collision.check(r.joints+fraction*(q-r.joints),float(fraction)*.05,f):
                return False,dict(reason='public_block_RML_target_sweep_collision',samples=count,failed_fraction=float(fraction))
        return True,dict(reason='public_block_RML_target_sweep_clear',samples=count,actual_motion_bound_not_claimed=True)

    def qualify_opening(self,requested,current_requested):
        f=self.frontend;r=f.robot;desired=float(requested>=.5);command=desired;proposed=None;distance=angle=None
        reason='trained_block_phase_opening'
        if self.phase=='acquire' and desired<.5:
            if self.close_witness is not None:command=0.;reason='preserve_single_actual_close_edge'
            else:
                target=self.current_grasp()
                if target is not None:distance,angle=errors(r,target)
                if target is not None and distance<=.02 and angle<=.1 and r.gripper_open:
                    command=0.;reason='current_public_block_physical_SE3'
                    if current_requested>=.5:proposed=dict(issued_at=float(r.timestamp),distance_m=distance,angle_rad=angle)
                else:command=1.;reason='defer_close_until_current_public_block_physical_SE3'
        return command,dict(time=float(r.timestamp),phase=self.phase,qualified_opening=command,
            desired_opening=desired,reason=reason,distance_m=distance,angle_rad=angle),proposed

    def confirm_opening(self,opening,proof,proposed,*,confirmed_after):
        if proposed is not None:
            self.close_witness=dict(proposed,confirmed_after=float(confirmed_after));self.frontend.close_confirmed=True
        self.gripper_events.append(dict(proof,actually_dispatched=True,confirmed_after=float(confirmed_after),own_close_edge=proposed is not None))

    def receipt(self):
        f=self.frontend
        return dict(schema='r10_public_block_intents_on_common_author_RML_v538',phase=self.phase,
            events=deepcopy(self.events),gripper_events=deepcopy(self.gripper_events),
            public_frames=deepcopy(f.events),own_close_witness=deepcopy(self.close_witness),
            held_geometry=f.collision.receipt(),shared_context=None if not hasattr(f.dynamics,'receipt') else f.dynamics.receipt(),
            historical_R09_executor_or_TEST_results_modified=False,Value_NGFF_FASTER_used=False,formal_VAL_TEST=False)
