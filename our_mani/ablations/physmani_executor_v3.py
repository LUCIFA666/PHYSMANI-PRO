"""Preserve the released path algorithm without uncounted physics steps.

PhysMani Mover appends ignore_collisions=1. The released executor discards
the first ten extracted waypoints and keeps Cartesian displacements >1 cm,
then executes nine RML arm ticks and one gripper tick per decision. Network
weights, sensor/world-model code and the common r05 actuator are unchanged.
"""
import numpy as np

from ..aligned.physmani import PhysManiAdapter as OriginalAdapter
from ..aligned.simulation import physics_time


def path_poses_without_physics(path):
    arm=path._arm
    clock=physics_time()
    joints_before=arm.get_joint_positions()
    targets_before=arm.get_joint_target_positions()
    ik_before=arm._ik_target.get_pose()
    joint_waypoints=np.asarray(path._path_points,dtype=float).reshape(-1,arm.get_joint_count())[1:].copy()
    poses=[]
    try:
        for joints in joint_waypoints:
            arm.set_joint_positions(joints.tolist())
            arm.set_joint_target_positions(joints.tolist())
            poses.append(arm.get_tip().get_pose())
    finally:
        arm.set_joint_positions(joints_before)
        arm.set_joint_target_positions(targets_before)
        arm._ik_target.set_pose(ik_before)
    if abs(physics_time()-clock)>1e-6:
        raise RuntimeError('waypoint extraction advanced physics')
    if not np.allclose(arm.get_joint_positions(),joints_before,rtol=0,atol=1e-6):
        raise RuntimeError('waypoint extraction did not restore robot joints')
    if not np.allclose(arm.get_joint_target_positions(),targets_before,rtol=0,atol=1e-6):
        raise RuntimeError('waypoint extraction did not restore servo targets')
    return np.asarray(poses,dtype=float).reshape(-1,7),joint_waypoints


def released_waypoint_indices(poses,*,discard=10,min_dist=.01):
    selected=[]
    for i in range(discard,len(poses)):
        if not selected or np.linalg.norm(poses[i,:3]-poses[selected[-1],:3])>min_dist:
            selected.append(i)
    return selected


def modify_released_path(path):
    poses,joints=path_poses_without_physics(path)
    selected=released_waypoint_indices(poses)
    raw=len(joints)+1
    path._path_points=joints[selected].reshape(-1)
    return dict(raw_joint_waypoints=raw,extracted_waypoints=len(joints),
                discarded_initial_waypoints=10,cartesian_spacing_m=.01,
                selected_waypoints=len(selected),selected_extracted_indices=selected,
                extraction_advance_physics=False)


class PhysManiAdapter(OriginalAdapter):
    def execute(self,env,action,*,observer=None):
        from pyrep.const import ConfigurationPathAlgorithms as Algos
        from pyrep.errors import ConfigurationPathError,IKError
        action=np.asarray(action,dtype=float)
        if action.shape!=(8,) or not np.isfinite(action).all():
            raise ValueError('invalid PhysMani action')
        clock=physics_time();path=None;error=None;modification=None
        # Original quick workspace/pose checks fail only the arm segment;
        # MoveArmThenGripper still rests and applies the gripper command.
        if not np.isclose(np.linalg.norm(action[3:7]),1.,rtol=0,atol=1e-3):
            error='nonunit_action_quaternion'
        elif not env.scene.check_target_in_workspace(action[:3]):
            error='outside_workspace'
        else:
            try:
                # Published Mover explicitly requests ignore_collisions=True.
                path=env.map_service.plan_path(action[:3].tolist(),quaternion=action[3:7].tolist(),
                    ignore_collisions=True,trials=100,max_configs=10,max_time_ms=10,
                    trials_per_goal=5,algorithm=Algos.RRTConnect)
                modification=modify_released_path(path)
                if len(path._path_points)<2*env.arm.get_joint_count():
                    # Native RML needs two points; the author catches an arm
                    # execution exception and proceeds with rest/gripper.
                    error='released_modified_path_too_short';path=None
            except (ConfigurationPathError,IKError) as exc:
                error=type(exc).__name__;path=None
        if abs(physics_time()-clock)>1e-6:
            raise RuntimeError('baseline planning/extraction advanced physics')
        done=False;arm_error=None;last_target=np.asarray(env.arm.get_joint_target_positions())
        ticks=int(self.config['baseline_decision_ticks'])
        executed=0
        try:
            for tick in range(ticks):
                if env.ledger.remaining<env.ledger.dt-1e-8 or env._native_success_latched:break
                if tick<ticks-1 and path is not None and not done and arm_error is None:
                    try:
                        done=path.step()
                        if path._joint_position_action is not None:
                            last_target=np.asarray(path._joint_position_action)
                    except RuntimeError as exc:
                        arm_error=repr(exc)
                opening=float(np.round(action[7])) if tick==ticks-1 else env._requested_opening
                env.command_joints(last_target,opening=opening,
                    phase='physmani_gripper' if tick==ticks-1 else 'physmani_path')
                executed+=1
                observation,robot=env.observe()
                if observer:observer(observation,robot)
                if env._native_success_latched:break
                self.observe(observation,robot)
        finally:
            # Receding-horizon decisions abandon unfinished RML handles.
            if path is not None and path._rml_handle is not None and not path._path_done:
                from pyrep.backend import sim
                sim.simRMLRemove(path._rml_handle)
                path._rml_handle=None
        return dict(path_found=path is not None,error=error,arm_error=arm_error,
                    ignore_collisions=True,waypoint_modification=modification,
                    executor='released_waypoints_no_hidden_physics_v3',ticks=executed)
