import types
import numpy as np
from .shadow import Scene,TaskProxy

def proxy(task):
    from pyrep.objects.object import Object
    from pyrep.backend import sim
    def rotation(angles):
        position=sim.ffi.new('float[3]',[0.,0.,0.]);euler=sim.ffi.new('float[3]',list(map(float,angles)))
        output=sim.ffi.new('float[12]')
        result=sim.lib.simBuildMatrix(position,euler,output)
        if result<0:raise RuntimeError('Original matrix-only Euler conversion failed')
        return np.asarray(list(output)).reshape(3,4)[:,:3]
    def euler(matrix):return sim.simGetEulerAnglesFromMatrix(np.asarray(matrix[:3]).reshape(12).tolist())
    return TaskProxy(task,lambda:Scene(Object,rotation,euler))

def instrument(original):
    """Capture the forecast time actually used, independently for every call."""
    context={};namespace=dict(original.__globals__)
    if 'compute_target_position'in namespace:
        position_function=namespace['compute_target_position']
        def position(*args,**kwargs):
            t=kwargs.get('t',args[0]if args else None)
            if t is not None:
                context['lookahead']=float(t-context['task'].t);context['source']='original_compute_target_position_time'
            return position_function(*args,**kwargs)
        namespace['compute_target_position']=position
    if 'compute_target_pose'in namespace:
        pose_function=namespace['compute_target_pose']
        def pose(*args,**kwargs):
            delay=kwargs.get('t_delay',args[2]if len(args)>2 else None)
            if delay is not None:context['lookahead']=float(delay);context['source']='original_compute_target_pose_time'
            return pose_function(*args,**kwargs)
        namespace['compute_target_pose']=pose
    function=types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    def expert(task,bool_return_path=False):
        if bool_return_path:raise ValueError('Read-only source command query only')
        task.begin();context.update(task=task,lookahead=0.,source='original_current_waypoint_no_forecast_call')
        result=function(task,bool_return_path=False)
        verification=task.scene.verify()
        expert.last_timing=dict(lookahead_seconds=round(max(0.,context['lookahead']),9),timing_source=context['source'],
                                shadow_scene=verification)
        return result
    expert.last_timing=None
    return expert
