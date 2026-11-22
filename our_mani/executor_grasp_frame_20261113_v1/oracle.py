"""Observe actual forecast helper calls, including function-local imports."""
from pathlib import Path
import sys
from ..executor_timed_tasks_20261112_v1.oracle import proxy

def instrument(original):
    root=Path(original.__code__.co_filename).resolve().parent
    def expert(task,bool_return_path=False):
        if bool_return_path:raise ValueError('Read-only oracle query only')
        task.begin();calls=[]
        previous=sys.getprofile()
        def observe(frame,event,arg):
            if previous is not None:previous(frame,event,arg)
            if event!='call'or Path(frame.f_code.co_filename).resolve().parent!=root:return
            values=frame.f_locals;name=frame.f_code.co_name
            if name=='compute_target_position'and't'in values:
                calls.append(dict(lookahead_seconds=float(values['t']-task.t),timing_source='observed_original_position_forecast_call'))
            elif name=='compute_target_pose'and't_delay'in values:
                declared=float(values['t_delay']);base=values.get('frame_base');waypoint=values.get('waypoint')
                effective=declared;linked=None
                if hasattr(base,'get_handle')and hasattr(waypoint,'get_parent'):
                    node=waypoint;linked=False
                    while node is not None:
                        if node.get_handle()==base.get_handle():linked=True;break
                        node=node.get_parent()
                    if not linked:effective=0.
                calls.append(dict(lookahead_seconds=effective,declared_helper_delay_seconds=declared,
                    waypoint_descends_from_rotating_frame=linked,timing_source='observed_original_pose_forecast_call'))
        sys.setprofile(observe)
        try:result=original(task,bool_return_path=False)
        finally:sys.setprofile(previous)
        timing=calls[-1]if calls else dict(lookahead_seconds=0.,timing_source='original_current_waypoint_no_forecast_call')
        expert.last_timing=dict(timing,lookahead_seconds=round(max(0.,timing['lookahead_seconds']),9),
            all_forecast_calls=calls,observes_local_import_helpers=True,shadow_scene=task.scene.verify())
        return result
    expert.last_timing=None
    return expert
