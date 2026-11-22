"""TRAIN-only committed waypoint supervision and counted source-style RML.

The follower core is CPU-testable. Native imports occur only in the explicitly
TRAIN-only planner/executor. Neither paths nor native teacher state are actor
observations. No function here calls pyrep.step or a blocking gripper helper.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

from ..contracts import Pose
from .protocol import MultitaskProtocolError,MultitaskInfrastructureError


class TeacherPathPlanningFailure(Exception):
    """A real teacher command has no path; retain failure without resampling."""


@dataclass(frozen=True)
class FollowerConfig:
    close_position_tolerance_m: float
    close_orientation_tolerance_rad: float
    joint_limit_recovery_margin_rad: float
    release_settle_ticks: int

    def __post_init__(self):
        values=(self.close_position_tolerance_m,self.close_orientation_tolerance_rad,
                self.joint_limit_recovery_margin_rad)
        if (not np.isfinite(values).all() or min(values)<=0 or type(self.release_settle_ticks) is not int
                or not 0<=self.release_settle_ticks<=200):
            raise ValueError('explicit finite TRAIN follower timing/recovery/settling parameters required')


def joint_backoff(joints,lower,upper,margin):
    q,lo,hi=map(lambda value:np.asarray(value,float),(joints,lower,upper))
    if (q.shape!=lo.shape or q.shape!=hi.shape or not np.isfinite([q,lo,hi]).all()
            or np.any(hi-lo<=2*margin)):
        raise MultitaskProtocolError('invalid public robot limits for counted recovery')
    return np.clip(q,lo+margin,hi-margin) if np.any((q<lo)|(q>hi)) else None


class CommittedWaypointFollower:
    """One raw teacher query per committed command, with explicit execution."""
    def __init__(self,teacher,path_planner,configuration,*,initial_opening):
        self.teacher,self.planner,self.config=teacher,path_planner,configuration
        self.opening=float(initial_opening>=.5)
        self.raw=None;self.path=None;self.reference=None;self.command_id=-1
        self.generated_at=None;self.state='query';self.path_done=False
        self.gripper_issued_at=None;self.gripper_done=False;self.release_edge=False
        self.settle_remaining=0;self.last_settle_time=None;self.failed=None
        self.commands=[];self.completed=False

    def _result(self,robot,*,new=False,recovery=False,arrived=False):
        return dict(label=self.raw,joint_reference=np.asarray(self.reference if self.reference is not None else robot.joints).copy(),
            opening=self.opening,command_id=self.command_id,command_generated_at=self.generated_at or 0.,
            command_age_seconds=0. if self.generated_at is None else float(robot.timestamp-self.generated_at),
            command_is_new=bool(new),execution_state=self.state,path_done=bool(self.path_done),
            measured_waypoint_arrival=bool(arrived),teacher_execution_valid=self.raw is not None and self.failed is None,
            joint_limit_recovery_action=bool(recovery),command_completed=False,failure=self.failed)

    def command(self,robot,lower,upper):
        recovery=joint_backoff(robot.joints,lower,upper,self.config.joint_limit_recovery_margin_rad)
        if recovery is not None:
            result=self._result(robot,recovery=True);result['joint_reference']=recovery
            result['teacher_execution_valid']=False
            result['execution_state']='public_joint_limit_recovery';return result
        if self.failed is not None:
            result=self._result(robot);result['joint_reference']=np.asarray(robot.joints).copy();return result
        now=float(robot.timestamp);new=False
        if self.completed:self.state='query'
        if self.state=='query':
            self.close_path()
            self.raw=dict(self.teacher());self.command_id+=1;self.generated_at=now;new=True
            self.path_done=False;self.gripper_done=False;self.gripper_issued_at=None;self.release_edge=False;self.completed=False
            self.reference=np.asarray(robot.joints).copy()
            goal=Pose.from_xyzw(self.raw['tcp_xyzw'])
            self.commands.append(dict(command_id=self.command_id,generated_at=now,
                raw_waypoint_xyzw=goal.xyzw().tolist(),desired_opening=float(self.raw['opening']),
                raw_stage=str(self.raw['stage']),raw_phase=str(self.raw['phase']),
                reference_body=str(self.raw['reference_body']),manipulated_body=str(self.raw['manipulated_body']),
                reference_role=str(self.raw['reference_role']),reference_kind=str(self.raw['reference_kind']),
                anchor_id=str(self.raw['anchor_id']),
                manipulated_role=str(self.raw.get('manipulated_role',self.raw['manipulated_body'])),
                issued_measured_tcp_xyzw=robot.pose.xyzw().tolist(),executed_tick_count=0,
                completion_target_type='committed_world_waypoint',
                command_completion_is_native_phase_success=False))
            try:self.path=self.planner(goal)
            except TeacherPathPlanningFailure as error:
                self.failed=str(error);self.state='failed_no_resample'
                self.commands[-1]['planning_failure']=self.failed
                return self._result(robot,new=new)
            self.state='follow_path'
        if self.state=='follow_path':
            self.path_done=bool(self.path.step())
            reference=self.path.get_executed_joint_position_action()
            if reference is not None:self.reference=np.asarray(reference,float).copy()
            if self.path_done:self.state='wait_arrival'
        arrived=False
        if self.state=='wait_arrival':
            goal=Pose.from_xyzw(self.raw['tcp_xyzw'])
            p=float(np.linalg.norm(robot.pose.position-goal.position))
            r=float(Rotation.from_matrix(goal.rotation@robot.pose.rotation.T).magnitude())
            arrived=p<=self.config.close_position_tolerance_m and r<=self.config.close_orientation_tolerance_rad
            if arrived:
                desired=float(float(self.raw['opening'])>=.5)
                self.release_edge=self.opening<.5 and desired>=.5
                self.opening=desired;self.gripper_issued_at=now;self.gripper_done=False
                self.state='actuate'
                self.commands[-1].update(measured_arrival_at=now,applied_opening=desired,
                    arrival_position_error_m=p,arrival_orientation_error_rad=r)
        result=self._result(robot,new=new,arrived=arrived)
        if self.state=='release_settle':result['joint_reference']=np.asarray(robot.joints).copy()
        return result

    def after_tick(self,feedback):
        """Complete only actual counted motor steps, never a raw request."""
        finished=False
        if feedback.get('joint_limit_recovery_action',False):return False
        if self.raw is not None and self.failed is None:
            self.commands[-1]['executed_tick_count']+=1
            self.commands[-1]['last_executed_at']=float(feedback['timestamp'])
        if self.state=='actuate':
            self.gripper_done=bool(feedback['gripper_actuation_done'])
            if self.gripper_done:
                self.commands[-1].setdefault('gripper_completed_at',float(feedback['timestamp']))
                if self.release_edge and self.config.release_settle_ticks:
                    self.state='release_settle';self.settle_remaining=self.config.release_settle_ticks
                else:finished=True
        elif self.state=='release_settle':
            self.settle_remaining-=1
            finished=self.settle_remaining==0
        if finished:
            terminal=Pose.from_xyzw(feedback['measured_tcp_xyzw'])
            target=Pose.from_xyzw(self.raw['tcp_xyzw'])
            self.commands[-1].update(command_completed_at=float(feedback['timestamp']),
                command_terminal_tcp_xyzw=terminal.xyzw().tolist(),
                command_terminal_target_xyzw=target.xyzw().tolist(),
                command_terminal_position_error_m=float(np.linalg.norm(terminal.position-target.position)),
                command_terminal_orientation_error_rad=float(Rotation.from_matrix(target.rotation@terminal.rotation.T).magnitude()))
            self.completed=True
        return finished

    def finish(self,reason,timestamp):
        if self.commands and not self.completed:
            self.commands[-1].update(cutoff_reason=str(reason),cutoff_at=float(timestamp),
                command_incomplete_at_cutoff=True)
        self.close_path()

    def close_path(self):
        if self.path is not None:self.path.close()
        self.path=None

    def close(self):self.close_path()


class NativeRMLPath:
    def __init__(self,path):self.path=path;self.done=False
    def step(self):
        if self.done:raise MultitaskProtocolError('committed RML path was stepped after completion')
        # Native RML cannot represent a path with no segment. An already-at-goal
        # single point is still subject to measured SE3 arrival before gripper.
        if len(self.path._path_points)<=self.path._num_joints:
            self.path._joint_position_action=np.asarray(self.path._path_points[-self.path._num_joints:])
            self.done=True
        else:self.done=bool(self.path.step())
        return self.done
    def get_executed_joint_position_action(self):return self.path.get_executed_joint_position_action()
    def close(self):
        if self.path._rml_handle is not None and not self.path._path_done:
            from pyrep.backend import sim
            sim.simRMLRemove(self.path._rml_handle)
            self.path._rml_handle=None


def plan_arm_path(arm,goal,*,family,algorithm,configuration_path_error):
    """Author family's pure arm-planning parameters; never task wrappers.

    Some author task.get_path wrappers run hidden physics while resampling.
    TRAIN retains the complete unmodified arm path instead.
    """
    initial_ignore={'moving_basketball_in_hoop':True,'push_moving_button':True,
        'insert_onto_rotating_peg':True,'put_rubbish_in_moving_bin':True,
        'beat_the_rotating_buzz':False,'place_cups_on_rotating_frame':False,
        'remove_cups_from_rotating_frame':False}
    if family not in initial_ignore:raise MultitaskProtocolError('unregistered TRAIN planner family')
    parameters=dict(quaternion=goal.xyzw()[3:],relative_to=None,trials=100,max_configs=10,
        max_time_ms=10,trials_per_goal=5,algorithm=algorithm)
    try:return arm.get_path(goal.position,ignore_collisions=initial_ignore[family],**parameters)
    except configuration_path_error:
        if initial_ignore[family]:raise
        return arm.get_path(goal.position,ignore_collisions=True,**parameters)


def native_path_planner(env):
    """Owned pure arm path is privileged TRAIN supervision only."""
    if getattr(env,'development_split',None)!='train':raise MultitaskProtocolError('oracle RML is TRAIN only')
    def plan(goal):
        from pyrep.errors import ConfigurationPathError,IKError
        from pyrep.const import ConfigurationPathAlgorithms as Algos
        from rlbench.backend.exceptions import InvalidActionError
        before=(env.physics_time(),float(env.task.t),env.ledger.ticks)
        q=np.asarray(env.arm.get_joint_positions());targets=np.asarray(env.arm.get_joint_target_positions())
        tip=Pose.from_xyzw(env.arm.get_tip().get_pose())
        if joint_backoff(q,env.robot_model.lower,env.robot_model.upper,.005) is not None:
            raise MultitaskProtocolError('recover public q bounds before native teacher planning')
        error=None;path=None
        try:path=plan_arm_path(env.arm,goal,family=env.spec.family,algorithm=Algos.RRTConnect,
            configuration_path_error=ConfigurationPathError)
        except (ConfigurationPathError,IKError,InvalidActionError) as failure:error=failure
        if before!=(env.physics_time(),float(env.task.t),env.ledger.ticks):
            raise MultitaskProtocolError('native teacher planning advanced physics/task clock')
        actual_tip=Pose.from_xyzw(env.arm.get_tip().get_pose())
        if (not np.allclose(q,env.arm.get_joint_positions(),atol=1e-6,rtol=0)
                or not np.allclose(targets,env.arm.get_joint_target_positions(),atol=1e-6,rtol=0)
                or np.linalg.norm(tip.position-actual_tip.position)>1e-6
                or Rotation.from_matrix(tip.rotation@actual_tip.rotation.T).magnitude()>1e-6):
            raise MultitaskProtocolError('native teacher path planning did not restore robot state')
        if error is not None:raise TeacherPathPlanningFailure(repr(error)) from error
        return NativeRMLPath(path)
    return plan


def execute_train_reference(env,reference,opening,*,phase):
    """One TRAIN-only direct reference tick, matching V7 physical actuation.

    Joint positions are bounded; velocity reference increments are not clipped.
    Native motor velocity limits and common physics/gripper remain unchanged.
    """
    if getattr(env,'development_split',None)!='train':raise MultitaskProtocolError('direct teacher reference is TRAIN only')
    env._before_action()
    q=np.asarray(reference,float);current=np.asarray(env.arm.get_joint_positions())
    if q.shape!=current.shape or not np.isfinite(q).all():raise MultitaskProtocolError('invalid TRAIN direct joint reference')
    applied=np.clip(q,env.robot_model.lower,env.robot_model.upper)
    before=np.asarray(env.gripper.get_open_amount(),float)
    try:
        env.arm.set_joint_target_positions(applied.tolist())
        desired=float(opening>=.5)
        if desired!=env._requested_opening:
            if desired:env.gripper.release();env.ledger.release_requests+=1
            else:
                for obj in env.task.get_graspable_objects():env.gripper.grasp(obj)
                env.ledger.grasp_requests+=1
            env._requested_opening=desired
        done=bool(env.gripper.actuate(desired,velocity=env.protocol['gripper_velocity']))
    except Exception as error:raise MultitaskInfrastructureError('TRAIN direct motor/gripper reference failed') from error
    env._tick(phase)
    after=np.asarray(env.gripper.get_open_amount(),float)
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise MultitaskProtocolError('nonfinite measured native gripper opening')
    return dict(applied_joint_reference=applied,gripper_actuation_done=done,
        gripper_open_amount_before=before,gripper_open_amount_after=np.asarray(env.gripper.get_open_amount(),float),
        timestamp=float(env.time),measured_tcp_xyzw=Pose.from_xyzw(env.arm.get_tip().get_pose()).xyzw(),
        native_physics_ticks=1,velocity_reference_increment_clipped=False,
        native_motor_velocity_limits_unchanged=True)
