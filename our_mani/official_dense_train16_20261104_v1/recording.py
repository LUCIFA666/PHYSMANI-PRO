"""One public frame/private label pair per actually observed physics step.

A PyRep-only gripper step remains distinguishable from a Scene+Task step.
No MPC h, velocity/acceleration command, contact force or action timing is
invented. Missed low-level simulator steps fault this preliminary observer.
"""
from copy import deepcopy
import math


class UnobservedPhysicalAdvance(RuntimeError):
    pass


def measured_clocks(value):
    if type(value)is not dict or set(value)!={'physics','task'}:
        raise ValueError('Exact independently read physics/task clocks required')
    out={}
    for key,x in value.items():
        if isinstance(x,bool)or not isinstance(x,(int,float))or not math.isfinite(x):
            raise ValueError('Finite nonboolean actual Native clock readback required')
        out[key]=float(x)
    return out


class DenseTrainingLedger:
    """Callbacks are an observation boundary, never a policy/qualification API."""
    def __init__(self,read_clocks,read_public,read_private,sink,*,maximum_physics_steps=400):
        if not all(callable(v)for v in (read_clocks,read_public,read_private,sink)):
            raise TypeError('Separate explicit current public/private readers and sink required')
        if type(maximum_physics_steps)is not int or maximum_physics_steps<1:
            raise ValueError('Fixed finite capture budget required')
        self.read_clocks,self.read_public,self.read_private,self.sink=read_clocks,read_public,read_private,sink
        self.maximum_physics_steps=maximum_physics_steps
        self.origin=None;self.last_physics=None;self.pending=None;self.rows=0;self.physics_steps=0
        self.fault=None;self.command=None;self.task_step_calls=0

    def set_command(self,label_only_command):
        # The official expert may be privileged TRAIN teacher. Its command,
        # stage/target identities are never merged into read_public().
        if type(label_only_command)is not dict:raise TypeError('Command label metadata required')
        self.command=deepcopy(label_only_command)

    def _live(self):
        if self.fault:raise RuntimeError('Dense observer permanently faulted: '+self.fault)

    def _fail(self,message):
        self.fault=message;raise UnobservedPhysicalAdvance(message)

    def initial(self):
        self._live()
        if self.origin is not None:raise ValueError('One initial row and one attempt only')
        self.origin=measured_clocks(self.read_clocks());self.last_physics=self.origin['physics']
        return self._emit('initial',None)

    def before_physics(self):
        self._live()
        if self.origin is None:raise ValueError('Record initial row first')
        self.flush('next_physics_before_taskless_row')
        now=measured_clocks(self.read_clocks())
        if abs(now['physics']-self.last_physics)>1e-9:
            self._fail('Unobserved low-level simulator advance; planning/resampling cannot be motion data')
        if self.physics_steps>=self.maximum_physics_steps:
            raise RuntimeError('Fixed physical step budget reached; retain partial attempt')
        # Actual q-target readback before this step, not a synthetic derivative.
        return dict(clocks=now,private_before=deepcopy(self.read_private()),command=deepcopy(self.command))

    def after_physics(self,before):
        self._live();after=measured_clocks(self.read_clocks())
        dt=after['physics']-before['clocks']['physics']
        if self.pending is not None or dt<=0:
            self._fail('One positive observed Native physics step required; no duplicate or UI-only tick')
        self.physics_steps+=1;self.last_physics=after['physics']
        self.pending=dict(physics_step=self.physics_steps,before=before,after_pyrep=after,
                          task_driver_observed=False,task_driver_before=None,task_driver_after=None)

    def before_task_driver(self):
        self._live();now=measured_clocks(self.read_clocks())
        if self.pending is None or abs(now['physics']-self.pending['after_pyrep']['physics'])>1e-9:
            self._fail('Task driver without one matching pending physics step')
        return now

    def after_task_driver(self,before):
        self._live();after=measured_clocks(self.read_clocks())
        if after['physics']!=before['physics'] or self.pending is None:
            self._fail('Task driver changed physics outside the observer boundary')
        self.pending.update(task_driver_observed=True,task_driver_before=before,task_driver_after=after)
        self.task_step_calls+=1
        return self.flush('after_original_task_driver')

    def flush(self,boundary='phase_return'):
        self._live()
        now=measured_clocks(self.read_clocks())
        if self.origin is not None and abs(now['physics']-self.last_physics)>1e-9:
            self._fail('Unobserved low-level simulator advance at phase return')
        if self.pending is None:return None
        if abs(now['physics']-self.pending['after_pyrep']['physics'])>1e-9:
            self._fail('Unobserved physics while a pending row awaited task driver')
        pending=self.pending
        try:row=self._emit(boundary,pending)
        except BaseException as error:
            self.fault=type(error).__name__+': '+str(error);raise
        self.pending=None
        return row

    def _emit(self,boundary,transition):
        from ..generic_reference_mpc_20261104_v1.public_observation import validate_policy_packet
        clocks=measured_clocks(self.read_clocks())
        public=validate_policy_packet(self.read_public())
        private=self.read_private()
        after=measured_clocks(self.read_clocks())
        if clocks!=after:self._fail('RGBD/private read advanced Native clock')
        relative=clocks['physics']-self.origin['physics']
        if abs(public['timestamp']-relative)>1e-9:raise ValueError('Public RGBD must share actual physical timestamp')
        if not isinstance(private,dict):raise TypeError('Dedicated private all-body TRAIN label required')
        label=dict(schema='official_dense_TRAIN_private_row_20261104_v1',label_only=True,policy_use_allowed=False,
            physical_step=self.physics_steps,actual_clocks=clocks,clock_origins=self.origin,
            actual_relative_physics_time=relative,actual_relative_task_time=clocks['task']-self.origin['task'],
            boundary=boundary,transition=deepcopy(transition),snapshot=deepcopy(private),
            issued_expert_command_label_only=deepcopy(self.command if transition is None else transition['before']['command']),
            joint_velocity_reference=None,joint_acceleration_reference=None,actuator_h=None,
            reference_observed_mask=dict(q_target=private.get('actual_q_target_positions')is not None,vref=False,aref=False,h=False),
            force=None,force_observed=False,physical_dt=None if transition is None else
            transition['after_pyrep']['physics']-transition['before']['clocks']['physics'],
            task_clock_not_substituted_for_physics=True,initial_row=transition is None,
            qualified_motion_training=False,planning_gap_status='no_unobserved_gap_so_far_not_whole_attempt_certificate')
        self.sink(self.rows,public,label);self.rows+=1
        return label
