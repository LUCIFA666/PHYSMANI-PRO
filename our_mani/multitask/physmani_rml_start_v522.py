"""Start the author's filtered RML path at the current measured robot joints.

The original waypoint filter is still executed. Prepending its omitted start
avoids submitting the eleventh raw waypoint as a discontinuous first target.
RRT, RML, gripper and counted stepping remain the original implementations.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from .physmani_geometry_executor_v517 import (
    PhysManiGeometryExecutor as PreviousExecutor,binding as previous_binding)
from .evaluation import file_sha

SOURCES=('our_mani/multitask/physmani_rml_start_v522.py',)


def binding(root,native_repo):
    original=previous_binding(root,native_repo)
    return dict(schema='r10_PhysMani_filtered_RML_with_measured_start_v522',
        original_geometry_adapter=original,
        sources_sha256={**original['sources_sha256'],**{p:file_sha(Path(root)/p) for p in SOURCES}},
        native_sources_sha256=original['native_sources_sha256'],
        author_waypoint_filter_retained=True,measured_actual_start_prepended_after_filter=True,
        original_RRT_RML_timing_gripper_counted_tick_and_motor_parameters_unchanged=True,
        actual_motion_bound_not_guaranteed=True,formal_VAL_TEST=False)


class ContinuousStartPath:
    def __init__(self,path,start,events):
        object.__setattr__(self,'_source_path',path)
        object.__setattr__(self,'_measured_start',np.asarray(start,float).copy())
        object.__setattr__(self,'_events',events)

    def __getattr__(self,name):return getattr(self._source_path,name)

    def __setattr__(self,name,value):
        if name=='_path_points':
            points=np.asarray(value,float)
            count=len(self._measured_start)
            if points.ndim!=1 or len(points)%count or not np.isfinite(points).all():
                raise ValueError('finite source RML path with original joint count required')
            rows=points.reshape(-1,count)
            if len(rows) and not np.allclose(rows[0],self._measured_start,rtol=0,atol=1e-7):
                self._events.append(dict(author_filtered_waypoints=len(rows),
                    measured_start=self._measured_start.tolist(),
                    original_first_filtered_waypoint=rows[0].tolist(),
                    removed_initial_reference_gap_rad=float(np.max(abs(rows[0]-self._measured_start))),
                    native_or_dynamic_object_state_reset=False))
                points=np.vstack([self._measured_start,rows]).reshape(-1)
            setattr(self._source_path,name,points)
        else:setattr(self._source_path,name,value)

    def step(self):return self._source_path.step()


class PhysManiGeometryExecutor(PreviousExecutor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.measured_start_events=[]

    def execute(self,env,action,*,observer=None):
        service=env.map_service;original=service.plan_path
        def plan_path(*args,**kwargs):
            path=original(*args,**kwargs)
            return ContinuousStartPath(path,env.arm.get_joint_positions(),self.measured_start_events)
        service.plan_path=plan_path
        try:return super().execute(env,action,observer=observer)
        finally:service.plan_path=original

    def geometry_receipt(self):
        return dict(super().geometry_receipt(),measured_RML_start_v522=dict(
            events=deepcopy(self.measured_start_events),native_robot_state_reset=False,
            original_RML_generator_reused=True,actual_motion_bound_not_guaranteed=True))
