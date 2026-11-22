"""Unwired, single declared12mm TCP-translation hold nominal experiment.

Only the hold-kind IK destination differs. Original world goal, full stage and
terminal cost, actions, model, constraints, H8 and 256-sample math are inherited.
Research cold anchors only: not admitted for reuse of an old warm basis.
"""
import time
import numpy as np
from .joint_goal_cem import JointGoalCEMPlanner,NOMINAL_BASIS_VERSION,reference
from .public_grasp_probe import VerificationProbeConfig
from .hold_orientation_guard_prototype_20261103 import bounded_object_nominal_destination

VERSION='unwired_current_TCP_plus_object_goal_direction_probe12mm_hold_nominal_20261103_v1'


class LocalHoldNominalPlanner(JointGoalCEMPlanner):
    def _prepare_nominals(self,initial,candidates,old):
        if old.get('phase')!='hold' or 'hold_tcp_quaternion' not in old:
            raise ValueError('This standalone experiment accepts existing hold feedback only')
        maximum=VerificationProbeConfig().lift_m
        if maximum!=.012:raise ValueError('The one predeclared old-probe12mm reference changed')
        world_goal=np.asarray(old.get('lift_goal_position',initial.get('lift_goal_position')),float)
        object_now=np.asarray(initial['object_position'],float)
        object_local=bounded_object_nominal_destination(object_now,world_goal,maximum_step_m=maximum)
        tcp_local=np.asarray(initial['tcp_position'])+(object_local-object_now)
        started=time.perf_counter();positions=[];quaternions=[];times=[]
        for kind in range(3):
            for candidate in candidates:
                at=max(initial['time'],candidate['deadline'])
                if kind<2:
                    target=reference.contact_target(initial,candidate,at,pregrasp=kind==0)
                    p,q=target['position'],target['quaternion']
                else:
                    p=tcp_local.copy()
                    q=old['hold_tcp_quaternion']
                    at=initial['time']
                positions.append(p);quaternions.append(q);times.append(at)
        poses=np.c_[positions,quaternions]
        seeds=np.broadcast_to(initial['q'],(24,7)).copy()
        result=self.goal_helper.adapter.solve(seeds,np.asarray(positions),np.asarray(quaternions))
        self._nominals=dict(joint_goal=result['joint_goal'].reshape(3,8,7),feasible=result['joint_goal_feasible'].reshape(3,8),
            pose=poses.reshape(3,8,7),time=np.asarray(times).reshape(3,8))
        self.nominal_receipt=dict(DLS_calls=1,poses=24,pregrasp_grasp_lift_per_candidate=3,
            feasible_by_kind=self._nominals['feasible'].copy(),wall_seconds=time.perf_counter()-started,
            deadlines=[candidate['deadline'] for candidate in candidates],
            basis_version=NOMINAL_BASIS_VERSION,object_prediction='current_anchor_CV_world_omega_fixed_absolute_deadline',
            frozen_current_project_modified=False,unwired_hold_nominal_version=VERSION,
            maximum_nominal_translation_m=maximum,existing_hold_quaternion_used=True,
            TCP_destination=tcp_local.copy(),original_world_object_goal=world_goal.copy(),
            rule='currentTCP+clip_norm(originalWorldObjectGoal-currentObject,12mm)',
            candidate_r_not_used_for_hold_nominal=True,original_global_cost_goal_unchanged=True,
            production_integrated=False,incompatible_old_warm_not_qualified=True)
