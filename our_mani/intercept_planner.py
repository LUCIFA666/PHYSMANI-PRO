"""Explicit candidate interception with a tracked, verified joint trajectory.

FASTER is absent. Nominal feasibility and execution use the same trajectory;
native position-servo tracking errors are logged rather than assumed away.
"""
from dataclasses import dataclass,replace
import numpy as np

from .contracts import Pose,HandoverRegion
from .control import HandoverGate
from .local_capture import LocalCaptureController,LocalConfig
from .context_control import ContextCaptureController
from .planning import RegionSelector
from .reachability import TimedReachability


@dataclass
class JointTrajectory:
    start: float
    end: float
    q0: np.ndarray
    v0: np.ndarray
    q1: np.ndarray
    v1: np.ndarray

    def at(self,time):
        duration=self.end-self.start
        if duration<=0:raise ValueError('joint trajectory must have positive duration')
        s=float(np.clip((time-self.start)/duration,0.,1.))
        q=(2*s**3-3*s**2+1)*self.q0+(s**3-2*s**2+s)*duration*self.v0
        q+=(-2*s**3+3*s**2)*self.q1+(s**3-s**2)*duration*self.v1
        v=(6*s**2-6*s)/duration*self.q0+(3*s**2-4*s+1)*self.v0
        v+=(-6*s**2+6*s)/duration*self.q1+(3*s**2-2*s)*self.v1
        return q,v


class CandidateInterception:
    def __init__(self,dynamics,model,collision_scene,goal_pose,config,*,predictor,ranking,value=None):
        self.dynamics,self.model,self.collision_scene=dynamics,model,collision_scene
        self.config,self.predictor,self.ranking=config,predictor,ranking
        self.goal_pose,self.value=goal_pose,value
        self.reachability=TimedReachability(model)
        self.regions=[HandoverRegion(f'candidate_{i}','target',Pose.from_xyzw([*offset,0,1,0,0]),config['region_radii'],
            max_relative_speed=config['relative_speed_limit'],max_orientation_error=config['orientation_limit'],
            value=1.,source='fixed_TRAIN_interception_templates') for i,offset in enumerate(config['candidate_regions'])]
        self.selector=RegionSelector(self.reachability,latency_reserve=0.,time_penalty=0. if value else .10,effort_penalty=0. if value else .02,
            success_value=value,minimum_success_probability=0.,
            object_position_bounds={'target':(np.array([-.03,-.53,.74]),np.array([.43,.53,.81]))})
        self.gate=HandoverGate()
        self.local_config=LocalConfig(max_speed=config['local_max_speed'],max_acceleration=config['local_max_acceleration'])
        self.stage='preposition'
        self.plan=self.goal=self.local=self.preposition_pose=None
        self.last_refresh=-np.inf
        self.events=[]
        self.handovers=0
        self.diagnostics={}

    def transition(self,stage,now,reason):
        self.events.append({'time':float(now),'from':self.stage,'to':stage,'reason':reason})
        self.stage=stage

    @staticmethod
    def trajectory(robot,check,end):
        return JointTrajectory(robot.timestamp,end,robot.joints.copy(),robot.joint_velocities.copy(),
            check.joints.copy(),check.joint_velocities.copy())

    def joint_command(self,robot):
        next_q,velocity=self.plan.at(robot.timestamp+self.config['dt'])
        error=next_q-robot.joints
        # Position setpoints compensate servo lag; native dynamics still run.
        delta=self.config['joint_servo_lead']*error
        delta=np.clip(delta,-self.model.velocity_limits*self.config['dt'],self.model.velocity_limits*self.config['dt'])
        command=np.clip(robot.joints+delta,self.model.lower,self.model.upper)
        if self.model.collision(command,self.config['dt']):
            self.plan=None
            return {'command':'hold','reason':'next_joint_target_collision'}
        return {'command':'joint_trajectory','joints':command,'opening':1.,
            'reference_joints':next_q,'reference_velocities':velocity,'servo_reference_error_rad':float(np.linalg.norm(error)),
            'trajectory_end':self.plan.end,'nominal_constraints_verified':True,
            'actual_tracking_guaranteed':False}

    def brake(self,robot,reason):
        duration=max(.2,1.5*float(np.max(np.abs(robot.joint_velocities)/self.model.acceleration_limits)))
        stop=robot.joints+.5*robot.joint_velocities*duration
        check=self.reachability.check_joints(robot,stop,np.zeros_like(stop),duration)
        if not check.feasible:return {'command':'hold','reason':reason,'braking_rejection':check.reason}
        self.plan=self.trajectory(robot,check,robot.timestamp+duration)
        command=self.joint_command(robot)
        command['reason']=reason
        command['braking_trajectory']=True
        return command

    def preposition(self,robot,obj):
        self.model.collision_query=None
        if self.preposition_pose is None:
            xy=np.clip(obj.pose.position[:2],[.08,-.28],[.32,.28])
            self.preposition_pose=Pose.from_xyzw([*xy,self.config['preposition_height'],0,1,0,0])
        distance=np.linalg.norm(robot.pose.position-self.preposition_pose.position)
        from scipy.spatial.transform import Rotation
        angle=Rotation.from_matrix(robot.pose.rotation@self.preposition_pose.rotation.T).magnitude()
        if distance<.035 and angle<.20:
            self.plan=None
            self.transition('seek',robot.timestamp,'observed_preposition_reached')
            return None
        if self.plan is None or robot.timestamp>=self.plan.end:
            reasons=[]
            for duration in (.8,1.2,1.6,2.,2.5,3.):
                check=self.reachability.check(robot,self.preposition_pose,np.zeros(3),np.zeros(3),duration)
                if check.feasible:
                    self.plan=self.trajectory(robot,check,robot.timestamp+duration)
                    break
                reasons.append(check.reason)
            if self.plan is None:
                return {'command':'hold','reason':'preposition_not_feasible','rejections':reasons}
        return self.joint_command(robot)

    def forecast(self,robot,obj):
        forecast=self.dynamics.predict([obj],self.config['forecast_seconds'],self.config['dt'])
        self.collision_scene.bind(forecast,robot.timestamp)
        self.model.collision_query=self.collision_scene.check
        return forecast

    def step(self,robot,obj,remaining_seconds):
        now=robot.timestamp
        if obj is None or not obj.visible or now-obj.timestamp>.1 or obj.timestamp>now+1e-8:
            self.goal=self.plan=None
            return {'command':'hold','reason':'fresh_visual_target_required'}
        if hasattr(self.dynamics,'observe'):self.dynamics.observe([obj])
        if self.stage=='preposition':
            command=self.preposition(robot,obj)
            if command is not None:return command
        if self.stage=='local':
            if self.capture_start+7.5<now:
                self.transition('failed',now,'local_capture_transport_timeout')
                return {'command':'hold','reason':'local_capture_transport_timeout'}
            try:
                pose,opening,v,w,details=self.local.command(robot,obj)
            except ValueError as error:
                if 'context' not in str(error) and 'history' not in str(error):raise
                return {'command':'hold','reason':'local_context_history_rebuilding','error':str(error)}
            except RuntimeError as error:
                self.goal=self.plan=self.local=None
                self.transition('seek',now,str(error))
                return {'command':'hold','reason':str(error)}
            return {'command':'local_capture','pose':pose,'opening':opening,'velocity':v,'angular_velocity':w,**details}
        if self.stage=='failed':return {'command':'hold','reason':'failed'}
        if remaining_seconds<1.2:
            self.transition('failed',now,'insufficient_remaining_capture_transport_budget')
            return {'command':'hold','reason':'insufficient_remaining_capture_transport_budget'}
        if self.goal is not None:
            gate=self.gate.check(self.goal,obj,robot,now)
            self.diagnostics['handover']=gate.__dict__
            if gate.allowed:
                if self.predictor=='context':
                    try:self.dynamics.context_snapshot(obj.object_id)
                    except ValueError as error:
                        return {'command':'hold','reason':'handover_context_history_rebuilding','error':str(error)}
                if self.value is not None:
                    p=self.value(region=self.goal.region,object_pose=obj.pose,object_velocity=obj.linear_velocity,
                        object_angular_velocity=obj.angular_velocity,robot_pose=robot.pose,robot_velocity=robot.linear_velocity,
                        robot_angular_velocity=robot.angular_velocity)
                    self.diagnostics['fresh_local_success_probability']=float(p) if np.isfinite(p) else None
                    if not np.isfinite(p):
                        self.goal=self.plan=None
                        self.transition('seek',now,'fresh_handover_outside_value_support')
                        return {'command':'hold','reason':'fresh_handover_outside_value_support'}
                self.local=(ContextCaptureController(self.goal_pose,self.local_config,self.dynamics) if self.predictor=='context' else
                            LocalCaptureController(self.goal_pose,self.local_config))
                self.handovers+=1;self.capture_start=now
                self.transition('local',now,'fresh_position_orientation_velocity_time_passed')
                return self.step(robot,obj,remaining_seconds)
            if now>self.goal.arrival_time+.2:
                self.goal=self.plan=None
                self.transition('seek',now,'missed_interception')
        if now-self.last_refresh>=self.config['replan_interval_seconds']-1e-8 or self.goal is None:
            self.last_refresh=now
            try:forecast=self.forecast(robot,obj)
            except ValueError as error:return {'command':'hold','reason':'forecast_unavailable','error':str(error)}
            # Refresh the existing interception rather than switching targets
            # every tick. A feasible fixed arrival is retained with fresh vision.
            retained=False
            if self.goal is not None and self.goal.arrival_time-now>=.15:
                pose,v,w=forecast.at(obj.object_id,self.goal.arrival_time)
                target,velocity=self.goal.region.in_world(pose,v,w)
                check=self.reachability.check(robot,target,velocity,w,self.goal.arrival_time-now)
                if check.feasible:
                    self.goal=replace(self.goal,observation_time=now,pose=target,linear_velocity=velocity,
                        angular_velocity=w,forecast=forecast,reachability=check)
                    self.plan=self.trajectory(robot,check,self.goal.arrival_time)
                    retained=True
                elif (self.plan is not None and now<self.plan.end
                      and np.linalg.norm(target.position-self.goal.pose.position)<=self.config['maximum_forecast_deviation']):
                    # Re-solving from a noisy servo state can reject a small
                    # correction. Keep the verified plan's absolute deadline
                    # instead of perpetually moving interception into future.
                    retained=True
                    self.diagnostics['arrival_commitment']='keep_verified_plan_for_small_forecast_change'
            if not retained and (self.goal is None or self.goal.arrival_time-now>=.15):
                self.goal=self.selector.select(forecast,self.regions,robot,now,min_horizon=.2,stride=self.config['candidate_stride'])
                self.diagnostics['selector']=dict(self.selector.last_diagnostics)
                if self.ranking=='geometry':
                    self.diagnostics['selector'].pop('selected_success_probability',None)
                    self.diagnostics['selector']['score_kind']='time_and_joint_effort_after_hard_feasibility'
                if self.goal is not None:
                    self.plan=self.trajectory(robot,self.goal.reachability,self.goal.arrival_time)
                    if self.stage!='intercept':self.transition('intercept',now,'feasible_candidate_selected')
                else:self.plan=None
        if self.plan is None:return self.brake(robot,'no_feasible_interception_candidate')
        self.collision_scene.bind(self.goal.forecast,now)
        return self.joint_command(robot)
