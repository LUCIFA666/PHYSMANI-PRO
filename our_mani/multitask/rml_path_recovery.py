"""Explicit recovery-only adapter; frozen v11 collectors do not import it.

Reuse the existing NativeRMLPath and author _step_motion. Only initialize its
handle with the owned virtual-probe factory before the first native step.
The one real native diagnostic did NOT reproduce the historical fault; it
proved compatibility, handle cleanup and zero physical/robot-state change.
"""
import json
from pathlib import Path

from .evaluation import file_sha
from .protocol import MultitaskProtocolError
from .rml_path_repair import NativeRMLHandleFactory
from .train_teacher import NativeRMLPath,native_path_planner

ROOT=Path(__file__).resolve().parents[2]
VERSION='r10_explicit_owned_RML_recovery_adapter_v1'
NATIVE_DIAGNOSTIC=dict(path='artifacts/framework/multitask_r10/RML_single_source_native_diagnostic_v1/report.json',
    sha256='118a89e30d483fa4ee697f4731417292450b5ef82a4666c4ffb8fffda944eba9')


def recovery_diagnostic_evidence(root=ROOT,pointer=None):
    """Bind the actual limited native evidence without upgrading its claim."""
    root=Path(root).resolve();pointer=NATIVE_DIAGNOSTIC if pointer is None else pointer
    path=(root/pointer['path']).resolve()
    if not path.is_relative_to(root) or file_sha(path)!=pointer['sha256']:
        raise ValueError('owned RML recovery diagnostic report path/SHA changed')
    value=json.loads(path.read_text())
    if (value.get('complete') is not True or value.get('native_cleanup_complete') is not True
            or value.get('all_inputs_SHA_rechecked_at_end') is not True
            or value.get('original_failure_reproduced') is not False
            or value.get('repaired_handle_creation_succeeded') is not True
            or len(value.get('repair_receipt',{}).get('probes',[]))!=1
            or any(value.get('whole_diagnostic_delta',{}).values())
            or value.get('independent_native_processes')!=1 or value.get('dataset_or_fitting_input') is not False):
        raise ValueError('single native diagnostic scope/result differs from its disclosed evidence')
    return dict(diagnostic_report=dict(pointer),native_diagnostic_processes=1,
        historical_fault_reproduced_in_native_diagnostic=False,
        native_velocity_correction_second_round_observed=False,
        measured_native_evidence='same fresh path compatibility, all RML handles cleaned, zero physical/robot-state delta',
        velocity_spike_repair_evidence='CPU mocked-native cross-probe tests; real failed path was not saved',
        dataset_or_closed_loop_success_claimed=False)


class RecoveredNativeRMLPath(NativeRMLPath):
    """Tiny opt-in subclass, with unchanged real native path execution."""
    def __init__(self,path,*,handle_factory=None,root=ROOT,diagnostic_pointer=None):
        if path._rml_handle is not None or path._path_done:
            raise MultitaskProtocolError('recovery adapter requires one fresh uninitialized native path')
        super().__init__(path)
        self.handle_factory=handle_factory or NativeRMLHandleFactory()
        if not callable(self.handle_factory):raise TypeError('explicit callable owned RML handle factory required')
        self._closed=False
        self._execution_stopped=False
        self.recovery_receipt=dict(version=VERSION,**recovery_diagnostic_evidence(root,diagnostic_pointer),
            initialization_count=0,base_adapter_step_calls=0,native_step_calls=0,initialization_receipt=None,
            close_called=False,close_complete=False,close_errors=[],execution_stopped=False,
            native_execution_completion_observed=False,errors=[],
            physics_step_called_by_adapter=False,real_native_step_motion_replaced=False)

    def step(self):
        if self._closed or self._execution_stopped:raise MultitaskProtocolError('recovery path cannot execute after close/error')
        try:
            if (not self.done and len(self.path._path_points)>self.path._num_joints
                    and self.path._rml_handle is None):
                self.recovery_receipt['initialization_count']+=1
                self.path._rml_handle=self.handle_factory(self.path)
                self.recovery_receipt['initialization_receipt']=getattr(self.handle_factory,'last_receipt',None)
            self.recovery_receipt['base_adapter_step_calls']+=1
            if not self.done and len(self.path._path_points)>self.path._num_joints:
                self.recovery_receipt['native_step_calls']+=1
            done=super().step()
        except Exception as error:
            self.recovery_receipt['initialization_receipt']=getattr(self.handle_factory,'last_receipt',None)
            self.recovery_receipt['errors'].append(dict(type=type(error).__name__,message=str(error)))
            self.close()
            raise
        if done and self.path._path_done:
            # Author _step_motion already removed this handle on state==1.
            # Retire its stale Python pointer; do not remove it twice.
            self.path._rml_handle=None
            self.recovery_receipt['native_execution_completion_observed']=True
        return done

    def close(self):
        if self._closed:return
        self._execution_stopped=True;self.recovery_receipt['execution_stopped']=True
        self.recovery_receipt['close_called']=True
        try:super().close()
        except Exception as error:
            self.recovery_receipt['close_errors'].append(dict(type=type(error).__name__,message=str(error)))
            raise
        self._closed=True;self.recovery_receipt['close_complete']=True


def recover_native_path(path,*,handle_factory=None,root=ROOT,diagnostic_pointer=None):
    return RecoveredNativeRMLPath(path,handle_factory=handle_factory,root=root,diagnostic_pointer=diagnostic_pointer)


def recovery_native_path_planner(env,*,handle_factory=None,root=ROOT,diagnostic_pointer=None):
    """Use the existing pure arm planner unchanged; wrap its unstepped result.

    A future recovery-only collector can substitute this callable when it
    constructs CommittedWaypointFollower. No original core hook is needed.
    """
    original=native_path_planner(env)
    def plan(goal):
        owned=original(goal)
        if not isinstance(owned,NativeRMLPath) or owned.done or owned.path._rml_handle is not None:
            raise MultitaskProtocolError('original TRAIN planner did not return an unstepped native path')
        return recover_native_path(owned.path,handle_factory=handle_factory,root=root,diagnostic_pointer=diagnostic_pointer)
    return plan
