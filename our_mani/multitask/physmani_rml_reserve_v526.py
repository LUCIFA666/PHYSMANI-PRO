"""Keep the PhysMani RML generator with a local planning reserve and no tail pause.

Use the existing repaired native RML handle factory. Its path-local arm view
sets planning acceleration to4; the real arm and original baseline stay at
their original parameters. Advance RML on the gripper tick as well, avoiding
the original ninth-reference repetition at each ten-tick action boundary.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from .physmani_rml_start_v522 import PhysManiGeometryExecutor as PreviousExecutor,binding as previous_binding
from .physmani_geometry_executor_v517 import GuardedPath,GeometryTargetRejected
from .rml_path_repair import NativeRMLHandleFactory
from ..sensor_alignment import physmani_executor_v6 as direct
from .evaluation import file_sha

SOURCES=('our_mani/multitask/physmani_rml_reserve_v526.py',
         'our_mani/multitask/rml_path_repair.py')


def binding(root,native_repo):
    parent=previous_binding(root,native_repo)
    return dict(schema='r10_PhysMani_RML_local_reference4_and_continuous_tenth_tick_v526',
        parent_start_adapter=parent,
        sources_sha256={**parent['sources_sha256'],**{p:file_sha(Path(root)/p) for p in SOURCES}},
        native_sources_sha256=parent['native_sources_sha256'],
        path_local_planning_acceleration=4.,actual_measured_window_limit=6.,
        inherited_timing_parameters_explicitly_superseded_for_ours=True,
        native_arm_parameters_and_original_baseline_unchanged=True,
        existing_repaired_virtual_RML_probe_factory_used=True,
        original_native_RML_step_motion_and_gripper_execution_reused=True,
        one_RML_advance_per_counted_motion_tick_including_gripper_tick=True,
        extra_physics_ticks=0,actual_motion_bound_not_guaranteed=True,formal_VAL_TEST=False)


class PlanningArm:
    def __init__(self,original):self.original=original
    def __getattr__(self,name):return getattr(self.original,name)
    @property
    def max_acceleration(self):return 4.


class PlanningPath:
    def __init__(self,original):self.original=original;self._arm=PlanningArm(original._arm)
    def __getattr__(self,name):return getattr(self.original,name)


class PhysManiGeometryExecutor(PreviousExecutor):
    def __init__(self,*args,rml_factory_class=NativeRMLHandleFactory,**kwargs):
        super().__init__(*args,**kwargs)
        self.rml_factory_class=rml_factory_class
        self.planning_receipts=[]
        self.tenth_tick_events=[]

    def execute(self,env,action,*,observer=None):
        service=env.map_service;original_plan=service.plan_path
        original_apply=direct._apply_direct_target
        current={'path':None,'failed':False}
        factories=[]
        first_geometry_event=len(self.geometry_events)
        def plan_path(*args,**kwargs):
            path=original_plan(*args,**kwargs)
            current['path']=path
            factory=self.rml_factory_class(maximum_virtual_steps=10000)
            factories.append(factory)
            # Native path.step() still calls its own unchanged _step_motion.
            # Only its handle allocation is supplied by the existing factory.
            path._get_rml_handle=lambda:factory(PlanningPath(path))
            old_step=path.step
            def step():
                try:return old_step()
                except BaseException:
                    current['failed']=True
                    raise
            path.step=step
            return path
        def apply(owned,joints,*,opening,phase):
            if owned is not env:
                raise direct.ProtocolInvariantError('RML reserve escaped its owned environment')
            path=current['path']
            rejected=any(not e['accepted'] for e in self.geometry_events[first_geometry_event:])
            if (phase=='physmani_gripper' and path is not None and not current['failed'] and not rejected
                    and path._rml_handle is not None and not path._path_done):
                try:
                    GuardedPath(path,env,self.admit_joint_target,self.geometry_events).step()
                    if path._joint_position_action is not None:
                        joints=np.asarray(path._joint_position_action,float).copy()
                    self.tenth_tick_events.append(dict(time=float(env.time),RML_advanced=True,
                        same_counted_gripper_tick=True,additional_physics_ticks=0))
                except GeometryTargetRejected:
                    self.tenth_tick_events.append(dict(time=float(env.time),RML_advanced=False,
                        reason='fresh_geometry_rejected_final_RML_target',additional_physics_ticks=0))
            return original_apply(owned,joints,opening=opening,phase=phase)
        service.plan_path=plan_path
        direct._apply_direct_target=apply
        try:return super().execute(env,action,observer=observer)
        finally:
            direct._apply_direct_target=original_apply
            service.plan_path=original_plan
            self.planning_receipts.extend(deepcopy(f.last_receipt) for f in factories if f.last_receipt is not None)

    def geometry_receipt(self):
        return dict(super().geometry_receipt(),RML_reference_reserve_v526=dict(
            local_planning_acceleration=4.,native_arm_parameters_modified=False,
            virtual_factory_receipts=deepcopy(self.planning_receipts),
            tenth_tick_events=deepcopy(self.tenth_tick_events),
            source_counterfactual_clipping_counts_can_include_final_RML_advance=True,
            actual_motion_bound_not_guaranteed=True))
