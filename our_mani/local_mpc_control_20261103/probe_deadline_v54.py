"""Probe terminal cost at its admitted absolute deadline, not a rolling H8.

Only the location of terminal phi changes. All eight original running costs,
command/predicted bounds, model transitions, CEM256/H8 and cached destinations
are retained. No action is clipped/replaced after optimization. This is an
optional development protocol, not an assertion of physical grasp success.
"""
import numpy as np

from .joint_goal_cem import norm, qlog, qmul, qinv
from .probe_nominal_cache_v53 import CachedNominalVerificationProbePlanner


SCHEMA='fixed_absolute_probe_terminal_phi_v54_20261103_v1'


class FixedDeadlineVerificationProbePlanner(CachedNominalVerificationProbePlanner):
    """Same constructor as v53; spec must carry the original trigger/deadline.

    Native decisions/trigger/deadline must lie on the actual 100ms grid. A
    deadline is never moved or interpolated. Warm proposals are still rescored
    using fresh actual state/h through the inherited v53 validation path.
    """

    def _terminal_index(self, now):
        values={name:self.probe_spec.get(name) for name in ('trigger_time','deadline')}
        values['decision_time']=now
        for name,value in values.items():
            if (isinstance(value,(bool,np.bool_))
                    or not isinstance(value,(float,int,np.floating,np.integer))
                    or not np.isfinite(value) or value<0):
                raise ValueError('finite absolute '+name+' required for fixed probe deadline')
            if abs(float(value)-round(float(value)/self.config.dt)*self.config.dt)>1e-8:
                raise ValueError('trigger/decision/deadline must align to actual100ms; no interpolation')
        trigger=float(values['trigger_time']);deadline=float(values['deadline']);now=float(now)
        if abs(deadline-trigger-.8)>1e-8:
            raise ValueError('preserve the original fixed .8s probe admission budget')
        if now<trigger-1e-8:raise ValueError('probe decision precedes its actual admitted trigger')
        remaining=deadline-now;index=int(round(remaining/self.config.dt))
        if not 1<=index<=8 or abs(remaining-index*self.config.dt)>1e-8:
            raise ValueError('fixed probe deadline is expired or outside the current H8 domain')
        return index

    def _rollout_batch(self,initial,candidates,modes,controls,old,committed,fixed_sequence):
        index=self._terminal_index(initial['time'])
        cost,parts,states,actions=super()._rollout_batch(
            initial,candidates,modes,controls,old,committed,fixed_sequence)
        old_phi=parts['probe_terminal_phi']
        if index==8:
            # Preserve exact first-plan numerical equivalence and tie ranking.
            new_phi=old_phi.copy()
        else:
            position=states['tcp_position'][:,index]
            quaternion=states['tcp_quaternion'][:,index]
            error=position-self.probe_spec['target_position']
            angle=norm(qlog(qmul(quaternion,qinv(self.probe_spec['target_quaternion']))))
            new_phi=self.probe_config.terminal_weight*(
                np.sum((error/self.probe_config.position_scale_m)**2,-1)
                +.2*(angle/self.probe_config.maximum_rotation_rad)**2)
            cost=cost-old_phi+new_phi
        parts=dict(parts)
        parts.update(probe_terminal_original_phi=old_phi.copy(),probe_terminal_phi=new_phi,
            probe_terminal_phi_delta=new_phi-old_phi,
            probe_terminal_original_index=np.full(len(modes),8),
            probe_terminal_index=np.full(len(modes),index),
            probe_terminal_original_time=np.full(len(modes),initial['time']+.8),
            probe_terminal_time=np.full(len(modes),self.probe_spec['deadline']))
        return cost,parts,states,actions

    def plan(self,state,candidates,*,warm_start=None):
        index=self._terminal_index(state['time'])
        result=super().plan(state,candidates,warm_start=warm_start)
        result['diagnostics'].update(probe_terminal_protocol=SCHEMA,
            probe_terminal_original_index=8,probe_terminal_index=index,
            probe_terminal_original_time=float(state['time'])+.8,
            probe_terminal_time=float(self.probe_spec['deadline']),
            fixed_probe_trigger_time=float(self.probe_spec['trigger_time']),
            fixed_probe_deadline_unchanged=True,deadline_interpolated=False,
            terminal_weight_unchanged=True,eight_running_costs_unchanged=True,
            hard_constraints_and_model_transitions_unchanged=True,
            terminal_annotation_fields_are_diagnostics_not_additive_costs=True,
            selected_action_postprocessed=False)
        return result
