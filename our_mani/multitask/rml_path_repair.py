"""Owned, opt-in repair of native virtual RML velocity-limit probes.

No frozen collector imports this module. The future recovery adapter may pass
``NativeRMLHandleFactory`` to NativeRMLPath, explicitly assign its returned
handle before delegating the unchanged native ``path.step()``, and retain the
receipt. This module never steps physics or writes a robot joint/target. All
native limits, RML parameters and the accepted path's actual execution remain
the author's. Each virtual probe starts from the path's initial q; a prior
probe's final q cannot become the next probe's previous sample.
"""
import operator
import numpy as np

VERSION='r10_owned_RML_virtual_probe_previous_q_reset_v1'


class RMLVirtualProbeExhausted(RuntimeError):
    """Virtual initialization exceeded its fixed numerical resource budget."""


class RMLProbeBackendFailure(RuntimeError):
    """An invalid/failed native RML operation, not a policy outcome."""


class RMLProbeCleanupFailure(RuntimeError):
    """A probe handle could not be deleted; infrastructure is unverified."""


def _require(value,message):
    if not value:raise ValueError(message)


def _handle(value):
    try:result=operator.index(value)
    except TypeError as error:raise RMLProbeBackendFailure('native RML did not return an integer handle') from error
    if result<0:raise RMLProbeBackendFailure('native RML returned a negative handle')
    return result


def create_repaired_rml_handle(path,*,sim_backend,maximum_virtual_steps=10000,receipt=None):
    """Return one live execution handle after deleting every virtual probe.

    The returned handle belongs to the caller and has not been stepped. Probe
    failures always delete their locally owned handle before propagating. The
    backend's simRMLStep advances a virtual trajectory only; the actual native
    path's _step_motion and counted executor remain outside this factory.
    """
    _require(type(maximum_virtual_steps) is int and maximum_virtual_steps>0,'explicit positive virtual probe budget required')
    receipt={} if receipt is None else receipt
    _require(isinstance(receipt,dict),'mutable audit receipt mapping required')
    receipt.update(version=VERSION,complete=False,probes=[],deleted_probe_handles=[],
        cleanup_failures=[],maximum_virtual_steps=maximum_virtual_steps,
        previous_q_reset_before_every_probe=True,physics_step_called=False,
        arm_joint_or_target_setter_called=False,physical_parameters_modified=False,
        final_execution_handle_stepped_by_factory=False,live_execution_handle_transferred=False)
    arm=path._arm;n=path._num_joints
    points=np.asarray(path._path_points,float)
    _require(type(n) is int and n>0 and points.ndim==1 and len(points)>=2*n
        and len(points)%n==0 and np.isfinite(points).all(),'finite multi-point native joint path required')
    limits=np.asarray(arm.get_joint_upper_velocity_limits(),float)
    dt=float(sim_backend.simGetSimulationTimeStep())
    maximum_velocity=float(arm.max_velocity);maximum_acceleration=float(arm.max_acceleration);maximum_jerk=float(arm.max_jerk)
    _require(limits.shape==(n,) and np.isfinite(limits).all() and (limits>0).all()
        and np.isfinite([dt,maximum_velocity,maximum_acceleration,maximum_jerk]).all()
        and min(dt,maximum_velocity,maximum_acceleration,maximum_jerk)>0,'unchanged positive native dt/velocity/acceleration/jerk limits required')
    lengths=np.asarray(path._get_path_point_lengths(),float)
    _require(lengths.shape==(len(points)//n,) and np.isfinite(lengths).all()
        and lengths[0]==0. and (np.diff(lengths)>=0).all() and lengths[-1]>0.,
        'nonzero finite monotone original joint-path arc lengths required')
    receipt.update(dt=dt,joint_upper_velocity_limits=limits.tolist(),maximum_velocity=maximum_velocity,
        maximum_acceleration=maximum_acceleration,maximum_jerk=maximum_jerk,
        path_point_count=len(lengths),path_length=float(lengths[-1]),velocity_limit_acceptance_ratio=1.001,
        native_RML_parameters=dict(dofs=1,smallestTimeStep=.0001,flags=-1,currentPosVelAccel=[0,0,0],
            selection=[1],targetPosVel=[float(lengths[-1]),0]))
    correction=1.
    def allocate():
        return _handle(sim_backend.simRMLPos(1,.0001,-1,[0,0,0],
            [maximum_velocity*correction,maximum_acceleration,maximum_jerk],[1],[float(lengths[-1]),0]))
    while True:
        previous_q=points[:n].copy() # The only intended trajectory-algorithm repair.
        rmax=0.;state=0;steps=0;probe_handle=None
        row=dict(probe_index=len(receipt['probes']),velocity_correction=correction,
            previous_q_initial=previous_q.tolist(),virtual_steps=0,maximum_joint_velocity_ratio=0.,
            first_step_joint_velocity_ratio=None,final_state=None,probe_handle_deleted=False,
            original_segment_condition_unmatched_samples=[])
        receipt['probes'].append(row)
        try:
            probe_handle=allocate();row['probe_handle']=probe_handle
            while state==0:
                state,pva=sim_backend.simRMLStep(probe_handle,dt,1);steps+=1
                row.update(virtual_steps=steps,virtual_trajectory_seconds=steps*dt,final_state=int(state))
                if steps>maximum_virtual_steps:
                    row['failure_kind']='virtual_probe_step_budget_exhausted'
                    raise RMLVirtualProbeExhausted('owned virtual RML probe exceeded '+str(maximum_virtual_steps)+' steps')
                pva=np.asarray(pva,float)
                if state not in (0,1) or pva.shape!=(3,) or not np.isfinite(pva).all():
                    row['failure_kind']='invalid_native_RML_state'
                    raise RMLProbeBackendFailure('native virtual RML step returned an invalid state/value')
                position=float(pva[0]);found=False
                for i in range(len(lengths)-1):
                    if lengths[i]<=position<=lengths[i+1]:
                        if lengths[i+1]==lengths[i]:
                            row['failure_kind']='zero_length_native_segment'
                            raise RMLProbeBackendFailure('native virtual RML selected a zero-length path segment')
                        weight=(position-lengths[i])/(lengths[i+1]-lengths[i])
                        first,second=points[n*i:n*(i+1)],points[n*(i+1):n*(i+2)]
                        q=first+(second-first)*weight
                        ratio=float(np.max(np.abs((q-previous_q)/dt)/limits));previous_q=q
                        if not np.isfinite(ratio):raise RMLProbeBackendFailure('nonfinite native joint velocity ratio')
                        rmax=max(rmax,ratio)
                        if row['first_step_joint_velocity_ratio'] is None:row['first_step_joint_velocity_ratio']=ratio
                        row['maximum_joint_velocity_ratio']=rmax;found=True;break
                if not found:
                    # Preserve the original probe's inclusive segment test.
                    # The original actual _step_motion separately extrapolates
                    # its final segment on floating endpoint overshoot. Do not
                    # silently alter either behavior in this narrow repair.
                    row['original_segment_condition_unmatched_samples'].append(
                        dict(virtual_step=steps,position=position,state=int(state)))
        finally:
            if probe_handle is not None:
                try:sim_backend.simRMLRemove(probe_handle)
                except Exception as error:
                    receipt['cleanup_failures'].append(dict(handle=probe_handle,error=repr(error)))
                    raise RMLProbeCleanupFailure('owned virtual RML handle deletion failed') from error
                row['probe_handle_deleted']=True;receipt['deleted_probe_handles'].append(probe_handle)
        if rmax<=1.001:break
        corrected=correction/rmax
        if not np.isfinite(corrected) or not 0<corrected<correction:
            row['failure_kind']='invalid_velocity_correction'
            raise RMLProbeBackendFailure('native virtual RML velocity correction cannot decrease finitely')
        correction=corrected
    # A fresh handle starts the real path at zero, exactly as in the author
    # implementation. It is transferred and must be deleted by native path
    # completion or the caller's ordinary close/failure cleanup.
    handle=allocate()
    receipt.update(complete=True,accepted_velocity_correction=correction,
        live_execution_handle=handle,live_execution_handle_transferred=True)
    return handle


class NativeRMLHandleFactory:
    """Explicit future NativeRMLPath handle_factory callable; opt-in only."""
    def __init__(self,*,sim_backend=None,maximum_virtual_steps=10000):
        self.sim_backend=sim_backend;self.maximum_virtual_steps=maximum_virtual_steps;self.last_receipt=None
    def __call__(self,path):
        backend=self.sim_backend
        if backend is None:
            from pyrep.backend import sim
            backend=sim
        self.last_receipt={}
        return create_repaired_rml_handle(path,sim_backend=backend,
            maximum_virtual_steps=self.maximum_virtual_steps,receipt=self.last_receipt)
