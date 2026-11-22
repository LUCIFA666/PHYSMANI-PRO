"""Causal public-motion residual for repeated reflections missed by the GRU.

Bounds/speeds are estimated from already observed RGB-D positions only. No task
area, Native body pose, schedule, future observation or label is accepted.
Until repeated extrema are observed consistently, the original GRU forecast is
retained. Neural candidate inputs continue to use their frozen GRU forecast.
"""
from collections import deque
from copy import deepcopy
import numpy as np
from ..contracts import Forecast


class PublicReflectionObserver:
    def __init__(self):
        self.history=deque(maxlen=96);self.direction=[0,0];self.events=[[],[]]

    def reset(self):self.history.clear();self.direction=[0,0];self.events=[[],[]]

    def observe(self,time,position):
        p=np.asarray(position,float)
        if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError('current public xyz required')
        if self.history:
            dt=time-self.history[-1][0]
            if dt<=0:raise ValueError('public motion observations must have increasing actual timestamps')
            if abs(dt-.05)>1e-6:self.reset()
        if self.history:
            delta=p-self.history[-1][1]
            for axis in range(2):
                if abs(delta[axis])<.004:continue
                sign=1 if delta[axis]>0 else -1
                if self.direction[axis] and sign!=self.direction[axis]:
                    old=self.history[-1]
                    self.events[axis].append(dict(time=old[0],position=float(old[1][axis]),kind='maximum' if sign<0 else 'minimum'))
                    self.events[axis]=self.events[axis][-12:]
                self.direction[axis]=sign
        self.history.append((float(time),p.copy()))

    def fuse(self,forecast):
        if not self.history or abs(self.history[-1][0]-forecast.times[0])>1e-8:
            raise ValueError('same-frame current public position and GRU forecast required')
        p=forecast.positions.copy();v=forecast.linear_velocities.copy();axes=[];evidence=[]
        elapsed=forecast.times-forecast.times[0]
        for axis in range(2):
            events=self.events[axis]
            minima=[e['position'] for e in events if e['kind']=='minimum']
            maxima=[e['position'] for e in events if e['kind']=='maximum']
            if len(events)<3 or not minima or not maxima or len(self.history)<12:continue
            lo,hi=float(np.median(minima)),float(np.median(maxima))
            if hi-lo<.1 or np.std(minima)>.015 or np.std(maxima)>.015:continue
            history=list(self.history)
            speeds=np.abs(np.diff([r[1][axis] for r in history[-32:]]))/.05
            speeds=speeds[speeds>.08]
            if len(speeds)<8:continue
            speed=float(np.median(speeds))
            if np.median(np.abs(speeds-speed))>.15*speed:continue
            x=float(forecast.positions[0,0,axis])
            lo=min(lo,x);hi=max(hi,x);width=hi-lo
            phase=(x-lo) if self.direction[axis]>0 else 2*width-(x-lo)
            phase=(phase+speed*elapsed)%(2*width)
            predicted=lo+np.where(phase<=width,phase,2*width-phase)
            predicted[0]=x
            p[:,0,axis]=predicted
            v[0,0,axis]=self.direction[axis]*speed
            v[1:,0,axis]=np.diff(predicted)/.05
            axes.append(axis);evidence.append(dict(axis=axis,observed_extrema=len(events),lo=lo,hi=hi,speed=speed))
        return Forecast(forecast.object_ids,forecast.times,p,forecast.rotations,v,forecast.angular_velocities,
            metadata=dict(source='causal_public_reflection_residual_plus_frozen_GRU',calibrated_axes=axes,
                calibration=evidence,Native_inputs_used=False,future_observations_used=False,
                observation_count=len(self.history),GRU_retained_on_uncalibrated_axes=True)),evidence
