"""Keep previously observed reflection evidence across measurement gaps.

No gap is interpolated or treated as a measured velocity/turn. The current
anchor still has to come from the original five-frame public GRU pipeline.
"""
import numpy as np
from ..contracts import Forecast
from .public_motion import PublicReflectionObserver


class EarlyPublicReflectionObserver(PublicReflectionObserver):
    def observe(self,time,position):
        p=np.asarray(position,float)
        if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError('current public xyz required')
        if self.history:
            dt=time-self.history[-1][0]
            if dt<=0:raise ValueError('increasing actual public timestamps required')
            if abs(dt-.05)<=1e-6:
                delta=p-self.history[-1][1]
                for axis in range(2):
                    if abs(delta[axis])<.004:continue
                    sign=1 if delta[axis]>0 else -1
                    if self.direction[axis] and sign!=self.direction[axis]:
                        old=self.history[-1]
                        self.events[axis].append(dict(time=old[0],position=float(old[1][axis]),kind='maximum' if sign<0 else 'minimum'))
                        self.events[axis]=self.events[axis][-12:]
                    self.direction[axis]=sign
            else:self.direction=[0,0]
        self.history.append((float(time),p.copy()))

    def fuse(self,forecast):
        if not self.history or abs(self.history[-1][0]-forecast.times[0])>1e-8:raise ValueError('current public anchor required')
        p=forecast.positions.copy();v=forecast.linear_velocities.copy();axes=[];evidence=[]
        history=list(self.history);elapsed=forecast.times-forecast.times[0]
        times=np.asarray([r[0] for r in history]);positions=np.asarray([r[1] for r in history])
        consecutive=np.abs(np.diff(times)-.05)<1e-6
        for axis in range(2):
            events=[e for e in self.events[axis] if forecast.times[0]-e['time']<=6.]
            minima=[e['position'] for e in events if e['kind']=='minimum'];maxima=[e['position'] for e in events if e['kind']=='maximum']
            if not self.direction[axis] or len(events)<2 or not minima or not maxima or len(history)<12:continue
            lo,hi=float(np.median(minima)),float(np.median(maxima))
            if hi-lo<.1 or np.std(minima)>.015 or np.std(maxima)>.015:continue
            speeds=np.abs(np.diff(positions[:,axis])[consecutive])/.05
            speeds=speeds[speeds>.08][-32:]
            if len(speeds)<8:continue
            speed=float(np.median(speeds))
            if np.median(np.abs(speeds-speed))>.15*speed:continue
            x=float(p[0,0,axis])
            if x<lo-.04 or x>hi+.04:continue
            lo=min(lo,x);hi=max(hi,x);width=hi-lo
            phase=(x-lo) if self.direction[axis]>0 else 2*width-(x-lo)
            phase=(phase+speed*elapsed)%(2*width)
            predicted=lo+np.where(phase<=width,phase,2*width-phase);predicted[0]=x
            p[:,0,axis]=predicted;v[0,0,axis]=self.direction[axis]*speed
            v[1:,0,axis]=np.diff(predicted)/.05
            axes.append(axis);evidence.append(dict(axis=axis,observed_extrema=len(events),lo=lo,hi=hi,speed=speed,
                measured_consecutive_velocity_pairs=int(consecutive.sum()),gap_interpolation=False,reflection_model_is_hypothesis=True))
        return Forecast(forecast.object_ids,forecast.times,p,forecast.rotations,v,forecast.angular_velocities,
            metadata=dict(source='observed_extrema_pair_public_reflection_residual_plus_frozen_GRU',calibrated_axes=axes,
                calibration=evidence,Native_inputs_used=False,future_observations_used=False,
                observation_count=len(history),GRU_retained_on_uncalibrated_axes=True)),evidence
