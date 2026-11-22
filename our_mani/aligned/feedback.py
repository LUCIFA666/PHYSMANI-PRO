"""Camera feedback ablation: refresh predictions at 2 Hz, keep 20 Hz robot control.

The history encoder can observe the same 20 Hz stream as the baseline's online
world model. Only the action-producing pipeline's forecast is frozen between
decision boundaries. Existing joint trajectories and MPC continue using
proprioception and the last forecast; no 50 ms pose increment is held for 500 ms.
"""
import numpy as np
from ..contracts import Forecast,ObjectState
from ..benchmark_perception import box_surface
from ..intercept_planner import CandidateInterception


class CachedVisualDynamics:
    def __init__(self,background):
        self.background=background
        self.cache=None
        self.context=None
        self.source_object=None
        self.ignore_observe=False

    def __getattr__(self,name):return getattr(self.background,name)

    def observe(self,objects):
        if not self.ignore_observe and hasattr(self.background,'observe'):
            self.background.observe(objects)

    def refresh(self,obj):
        self.cache=self.background.predict([obj],1.1,.05)
        self.source_object=obj
        self.context=self.background.context_snapshot(obj.object_id) if hasattr(self.background,'context_snapshot') else None

    def context_snapshot(self,object_id='target'):
        if self.context is None:raise ValueError('context cache not ready')
        return dict(self.context)

    def object_at(self,time):
        if self.cache is None:raise ValueError('visual cache not ready')
        pose,velocity,angular=self.cache.at(self.source_object.object_id,time)
        return ObjectState(self.source_object.object_id,time,pose,box_surface(pose,(.05,.05,.05)),
                           velocity,angular,self.source_object.confidence)

    def predict(self,objects,horizon,dt):
        if self.cache is None:raise ValueError('visual cache not ready')
        obj=objects[0]
        times=obj.timestamp+np.arange(round(horizon/dt)+1)*dt
        rows=[self.cache.at(obj.object_id,t) for t in times]
        return Forecast((obj.object_id,),times,np.array([p.position for p,_,_ in rows])[:,None],
            np.array([p.rotation for p,_,_ in rows])[:,None],np.array([v for _,v,_ in rows])[:,None],
            np.array([w for _,_,w in rows])[:,None],metadata={**self.cache.metadata,
                'visual_feedback_frozen_at':float(self.cache.times[0]),'camera_feedback_period_seconds':.5})


class CachedFeedbackInterception:
    def __init__(self,*args,**kwargs):
        self.inner=CandidateInterception(*args,**kwargs)
        self.last_visual_refresh=-np.inf

    def __getattr__(self,name):return getattr(self.inner,name)

    def step(self,robot,obj,remaining_seconds):
        now=robot.timestamp
        dynamics=self.inner.dynamics
        if now-self.last_visual_refresh>=.5-1e-8:
            self.last_visual_refresh=now
            if obj is None:return {'command':'hold','reason':'2Hz_camera_frame_unavailable'}
            try:dynamics.refresh(obj)
            except ValueError as exc:return {'command':'hold','reason':'2Hz_context_unavailable','error':str(exc)}
            return self.inner.step(robot,obj,remaining_seconds)
        if dynamics.cache is None:return {'command':'hold','reason':'awaiting_first_2Hz_forecast'}
        if self.inner.stage=='local':
            predicted=dynamics.object_at(now)
            dynamics.ignore_observe=True
            try:return self.inner.step(robot,predicted,remaining_seconds)
            finally:dynamics.ignore_observe=False
        if self.inner.stage=='failed':return {'command':'hold','reason':'failed'}
        if self.inner.plan is not None:
            if self.inner.goal is not None:
                self.inner.collision_scene.bind(dynamics.cache,now)
                self.inner.model.collision_query=self.inner.collision_scene.check
            return self.inner.joint_command(robot)
        return {'command':'hold','reason':'awaiting_next_2Hz_visual_decision'}
