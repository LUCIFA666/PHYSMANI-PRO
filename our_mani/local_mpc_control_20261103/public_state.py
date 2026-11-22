"""Assemble MPC state from public estimates and fresh robot/controller data.

Unknown manipulation state produces no policy state. A closing request cannot
set held=True. Native task labels, object handles and success are not inputs.
This is an input contract, not qualification of public attachment estimates.
"""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation

from .public_motion_bridge import PublicRobotSensors
from ..contracts import ObjectState, Pose


ACTUATOR_KEYS = frozenset(('reference_valid','reference_positions','reference_velocities',
    'reference_accelerations','target_joints','reference_clock','gripper_target',
    'gripper_done','reference_is_measured_state'))
PHASE_KEYS = frozenset(('phase','dwell','lost_hold','updated_at','metrics','held_is_measured',
    'close_entered_at','lift_goal_position','hold_tcp_quaternion','reopening'))


def _phase_memory(phase, now):
    if phase is None:
        return None
    if not isinstance(phase,dict) or not set(phase)<=PHASE_KEYS or phase.get('phase') not in ('approach','align','close','hold'):
        raise ValueError('only public controller phase memory is accepted')
    out = deepcopy(phase)
    for key in ('dwell','lost_hold','close_entered_at'):
        if key in out and (not np.isfinite(out[key]) or out[key]<0):
            raise ValueError('finite causal public phase timing required')
    if 'close_entered_at' in out and out['close_entered_at']>now+1e-8:
        raise ValueError('public close entry cannot be in the future')
    for key in ('reopening','held_is_measured'):
        if key in out and not isinstance(out[key],(bool,np.bool_)):
            raise ValueError('explicit public phase boolean required')
    if out.get('updated_at') is not None and (not np.isfinite(out['updated_at']) or out['updated_at']>now+1e-8):
        raise ValueError('public phase time cannot be in the future')
    metrics = out.get('metrics',{})
    if not isinstance(metrics,dict) or not set(metrics)<= {'distance','pregrasp_distance','angle','relative_speed'}:
        raise ValueError('only public controller phase metrics are accepted')
    if any(not np.isfinite(v) or v<0 for v in metrics.values()):
        raise ValueError('finite public phase metrics required')
    for key,size in (('lift_goal_position',3),('hold_tcp_quaternion',4)):
        if key in out:
            arr = np.asarray(out[key],float)
            if arr.shape!=(size,) or not np.isfinite(arr).all():
                raise ValueError('finite public '+key+' required')
            if key=='hold_tcp_quaternion' and not np.isclose(np.linalg.norm(arr),1.,atol=1e-5,rtol=0):
                raise ValueError('unit public hold TCP quaternion required')
    if 'held_is_measured' in out:
        out['held_is_measured'] = False  # Current held is an explicit public estimate.
    return out


class PublicStateAssembler:
    def __init__(self, *, allow_test_doubles=False):
        self.allow_test_doubles = bool(allow_test_doubles)
        self.last_robot = None
        self.robot_twist = None

    def observe_robot(self, robot):
        if not isinstance(robot, PublicRobotSensors):
            robot = PublicRobotSensors.from_mapping(robot)
        if self.last_robot is None:
            velocity, omega, valid = np.zeros(3), np.zeros(3), False
        else:
            dt = robot.timestamp - self.last_robot.timestamp
            if dt <= 0:
                raise ValueError('unique causal robot sensor times required')
            valid = bool(np.isclose(dt, .05, atol=1e-6, rtol=0))
            if valid:
                velocity = (robot.tcp_position-self.last_robot.tcp_position)/dt
                previous = Rotation.from_quat(self.last_robot.tcp_quaternion)
                omega = (Rotation.from_quat(robot.tcp_quaternion)*previous.inv()).as_rotvec()/dt
            else:
                velocity, omega = np.zeros(3), np.zeros(3)
        self.last_robot = robot
        self.robot_twist = (velocity, omega, valid)
        return valid

    def build(self, result, robot, actuator_state, *, phase_state=None):
        if not isinstance(robot, PublicRobotSensors):
            robot = PublicRobotSensors.from_mapping(robot)
        if self.last_robot is None or abs(robot.timestamp-self.last_robot.timestamp)>1e-9:
            raise ValueError('observe each actual 50ms robot frame before state assembly')
        for key in ('tcp_position','tcp_quaternion','q','dq','finger_open_amounts'):
            if not np.array_equal(getattr(robot,key),getattr(self.last_robot,key)):
                raise ValueError('build must use the exact observed robot sensor frame')
        if not isinstance(actuator_state,dict) or set(actuator_state)!=ACTUATOR_KEYS:
            raise ValueError('fresh whitelisted executor reference snapshot required; no task channels')
        if actuator_state['reference_is_measured_state'] is not False:
            raise ValueError('controller references are not robot measurements')
        if type(actuator_state['reference_valid']) is not bool:
            raise ValueError('explicit actual reference validity required')
        if actuator_state['gripper_done'] is not None and not isinstance(actuator_state['gripper_done'],(bool,np.bool_)):
            raise ValueError('actual gripper_done must be boolean or uninitialized')
        initialized = any(actuator_state[k] is not None for k in
            ('reference_positions','reference_velocities','reference_accelerations','target_joints'))
        if initialized:
            clock = actuator_state['reference_clock']
            if clock is None or not np.isfinite(clock) or abs(clock-robot.timestamp)>1e-8:
                raise ValueError('fresh executor reference_clock must equal the decision timestamp')
            for key in ('reference_positions','reference_velocities','reference_accelerations'):
                arr = np.asarray(actuator_state[key],float)
                if arr.shape!=(7,) or not np.isfinite(arr).all():
                    raise ValueError('finite initialized executor '+key+'[7] required')
            target = actuator_state['target_joints']
            if actuator_state['reference_valid'] or target is not None:
                arr = np.asarray(target,float)
                if arr.shape!=(7,) or not np.isfinite(arr).all():
                    raise ValueError('finite actual target_joints[7] required')
        elif actuator_state['reference_valid'] or actuator_state['reference_clock'] is not None:
            raise ValueError('uninitialized executor cannot claim a valid/current reference')
        requested = float(actuator_state['gripper_target'])
        if requested not in (0.,1.):
            raise ValueError('actual known binary requested gripper opening required')
        if not isinstance(result,dict) or abs(float(result['timestamp'])-robot.timestamp)>1e-8:
            raise ValueError('timestamp-aligned current public bridge result required')
        if (result.get('Native_inputs_used') is not False or result.get('future_observations_used') is not False
                or result.get('observations_fabricated') is not False):
            raise ValueError('only causal public motion estimates are accepted')
        if not self.allow_test_doubles and (result.get('forecaster_is_injected_test_double')
                or result.get('tracker_dependency_injected') or result.get('goal_marker_dependency_injected')):
            raise ValueError('test doubles cannot enter a Native policy canary')
        phase = _phase_memory(phase_state,robot.timestamp)
        if not result.get('available') or not self.robot_twist[2]:
            return None
        obj = result.get('object_observation')
        if not isinstance(obj,ObjectState) or abs(obj.timestamp-robot.timestamp)>1e-8 or not obj.visible:
            return None
        goal = result.get('goal_pose')
        if not isinstance(goal,Pose):
            return None
        if phase and 'lift_goal_position' in phase and not np.allclose(phase['lift_goal_position'],goal.position,atol=1e-8,rtol=0):
            raise ValueError('phase lift target must match the public static goal estimate')
        velocity, omega, _ = self.robot_twist
        if result.get('mode')=='FREE' and result.get('free_forecast_available'):
            pose = obj.pose
            object_v, object_w = obj.linear_velocity, obj.angular_velocity
            held = False
            held_source = 'public_free_motion_estimate_not_Native_held'
        elif result.get('mode')=='ATTACHED_PUBLIC' and result.get('controlled_mode_available'):
            pose = result['controlled']['estimated_world_pose']
            if not isinstance(pose,Pose):
                raise ValueError('public controlled world pose required')
            object_v = velocity + np.cross(omega,pose.position-robot.tcp_position)
            object_w = omega
            held = True
            held_source = 'current_public_attachment_evidence_not_Native_held'
        else:
            return None
        state = dict(time=robot.timestamp,tcp_position=robot.tcp_position.copy(),
            tcp_quaternion=robot.tcp_quaternion.copy(),tcp_velocity=velocity.copy(),
            tcp_angular_velocity=omega.copy(),q=robot.q.copy(),dq=robot.dq.copy(),
            finger_open_amounts=robot.finger_open_amounts.copy(),
            opening=float(robot.finger_open_amounts.mean()),requested_opening=requested,
            object_position=pose.position.copy(),object_quaternion=pose.xyzw()[3:],
            object_linear_velocity=np.asarray(object_v).copy(),object_angular_velocity=np.asarray(object_w).copy(),
            held=held,held_source=held_source,lift_goal_position=goal.position.copy(),
            actuator_state=deepcopy(actuator_state),phase_state=phase,
            information_setting='public_estimated_state_development_canary',
            Native_object_state_used=False,Native_held_used=False,Native_success_used=False,
            object_orientation_observable=bool(result['rotation_observable']))
        return state
