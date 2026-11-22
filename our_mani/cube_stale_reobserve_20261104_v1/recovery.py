"""Bounded reobservation proposals, separate from measured centres and goals.

This new protocol is not enabled in any frozen producer. A prediction is only
an object_roi_hint. Only current real measurements restore original policy
eligibility. Retargets never introduce a close edge or reset proof owners.
"""
from copy import deepcopy
import numpy as np
from ..contracts import ObjectState
from ..local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors
from ..generic_reference_mpc_20261104_v1.reference_mpc import qualify_first_action


class StaleReobserveSupervisorV1:
    def __init__(self,*,maximum_reobserve_cycles=3,joint_retreat_cap_rad=.03,hint_horizon_seconds=.15):
        if type(maximum_reobserve_cycles)is not int or maximum_reobserve_cycles!=3:
            raise ValueError('Explicit bounded three100ms reobservation attempts required')
        if joint_retreat_cap_rad!=.03 or hint_horizon_seconds!=.15:
            raise ValueError('Declared0.03rad proposal bound /0.15s hint horizon required')
        self.maximum_cycles=maximum_reobserve_cycles;self.cap=joint_retreat_cap_rad;self.hint_horizon=hint_horizon_seconds
        self._now=None;self._robot=None;self._history=[];self._last_q=None
        self._stale=False;self._eligible=False;self._cycles=0;self._epoch=0
        self._ready=False
        self._pending=None;self._serial=0

    def observe(self,result,robot,*,close_requested,attempt_id=None,previously_confirmed=False):
        if type(robot)is not PublicRobotSensors or type(close_requested)is not bool or type(previously_confirmed)is not bool:
            raise TypeError('Actual typed public robot and explicit command/proof state required')
        if type(result)is not dict or any(result.get(k)is not False for k in ('Native_inputs_used','future_observations_used','observations_fabricated')):
            raise ValueError('Original public provenance required')
        now=float(robot.timestamp)
        if self._now is not None:
            if now<=self._now:raise ValueError('Only one actual current frame per timestamp')
            if abs(now-self._now-.05)>1e-6:
                self._history=[];self._last_q=None;self._stale=True
        if abs(float(result.get('timestamp',-1))-now)>1e-9:raise ValueError('Same-frame public result/robot required')
        self._now=now;self._robot=robot
        obj=result.get('object_observation')
        measured=(result.get('observed_target_available')is True and isinstance(obj,ObjectState)
                  and obj.visible and abs(obj.timestamp-now)<=1e-9)
        self._eligible=not close_requested and attempt_id is None and not previously_confirmed
        if measured:
            self._history.append((now,obj.pose.position.copy()))
            self._history=self._history[-5:];self._last_q=robot.q.copy()
        ready=measured and result.get('free_forecast_available')is True
        self._ready=bool(ready)
        if ready:
            self._stale=False;self._cycles=0
        elif not measured and not self._stale:
            self._stale=True;self._epoch+=1;self._cycles=0
        return dict(timestamp=now,current_measurement_available=measured,
            original_free_policy_ready=bool(ready),stale=self._stale,
            forbid_new_close=not ready,existing_closed_request_must_be_preserved=close_requested,
            original_probe_or_maintenance_reset=False,goal_sphere_or_crop_anchor_rewritten=False)

    def _hint(self):
        if not self._history or not self._eligible:return None
        t,p=self._history[-1];age=self._now-t
        if age<0 or age>self.hint_horizon+1e-9:return None
        velocity=np.zeros(3)
        if len(self._history)>=2:
            t0,p0=self._history[-2]
            if abs(t-t0-.05)<=1e-6:velocity=(p-p0)/(t-t0)
        return dict(name='object_roi_hint',position_world=(p+age*velocity).tolist(),
            current_time=self._now,last_actual_measurement_time=t,age_seconds=age,
            measurement=False,forecast_for_control=False,purpose='search_hint_only',
            replaces_public_sphere_center=False,changes_certified_crop_or_route=False)

    def propose(self):
        if self._robot is None:raise ValueError('Observe actual current public feedback first')
        if self._pending is not None:return deepcopy(self._pending)
        retreat=self._stale and self._eligible and self._last_q is not None and self._cycles<self.maximum_cycles
        q=self._robot.q.copy()
        if retreat:q=q+np.clip(self._last_q-q,-self.cap,self.cap)
        self._serial+=1
        proposal=dict(proposal_id=self._serial,epoch=self._epoch,timestamp=self._now,
            kind='REOBSERVE_JOINT'if retreat else 'KEEP_ORIGINAL_HOLD',
            joint_goal=q.tolist(),actual_q_at_proposal=self._robot.q.tolist(),
            opening_command=None,forbid_new_close=not self._ready,
            object_roi_hint=self._hint(),attempt_index=self._cycles,
            Native_qualification_required=bool(retreat),physical_or_visibility_guarantee=False,
            original_goal_or_observed_centre_replaced=False,original_proof_owner_reset=False)
        self._pending=proposal
        return deepcopy(proposal)

    def commit(self,proposal_id,*,executed_ticks):
        if self._pending is None or proposal_id!=self._pending['proposal_id']:
            raise ValueError('Only exact pending proposal may commit once')
        if type(executed_ticks)is not int or executed_ticks not in (0,1,2):
            raise ValueError('Actual original executor0..2ticks required')
        proposal=self._pending;self._pending=None
        if proposal['kind']=='REOBSERVE_JOINT':self._cycles+=1
        return dict(proposal_id=proposal_id,actual_executed_ticks=executed_ticks,
            reobservation_attempts=self._cycles,maximum_attempts=self.maximum_cycles,
            prediction_promoted_to_measurement=False)


def execute_reobserve_cycle(supervisor,backend,executor,proposal):
    """Optional future runner hook. Original8queries, then original executor.

    This never issues a command for KEEP_ORIGINAL_HOLD: the existing V72 hold /
    proof path owns that behavior. No frozen runner currently invokes this.
    """
    if type(supervisor)is not StaleReobserveSupervisorV1 or executor.backend is not backend:
        raise TypeError('Exact new supervisor and current original executor backend required')
    if supervisor._pending!=proposal:raise ValueError('Exact fresh whole proposal required')
    if proposal['kind']!='REOBSERVE_JOINT':
        return dict(executed=False,reason='original_hold_path_retained',commit=supervisor.commit(proposal['proposal_id'],executed_ticks=0))
    if (executor.state()['gripper_target']!=1.
            or getattr(backend,'requested_opening',1.)!=1.
            or abs(backend.clock()-proposal['timestamp'])>1e-9):
        raise ValueError('Reobservation is pre-close/current-time only; no borrowed closed or old state')
    q=np.asarray(backend.get_joint_positions(),float)
    if not np.allclose(q,proposal['actual_q_at_proposal'],atol=1e-6,rtol=0):
        raise ValueError('Physical q changed since public proposal')
    allowed,qualification=qualify_first_action(backend,dict(time=backend.clock(),q=q),proposal)
    if not allowed:
        return dict(executed=False,reason='original_configuration_checks_rejected',qualification=qualification,
            commit=supervisor.commit(proposal['proposal_id'],executed_ticks=0))
    receipt=executor.execute_joint_target(proposal['joint_goal'],opening=None,phase='STALE_safe_public_reobservation')
    return dict(executed=True,qualification=qualification,executor=receipt.to_dict(),
        commit=supervisor.commit(proposal['proposal_id'],executed_ticks=receipt.executed_ticks),
        opening_command_changed=False,centre_or_goal_replaced=False)
