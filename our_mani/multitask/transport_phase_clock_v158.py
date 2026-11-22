"""Fixed transport deadline beginning at the first confirmed partial motion.

Elapsed partial transport is already part of the same physical operation. A
later endpoint candidate retains that deadline instead of reserving the full
learned duration a second time. No unissued proposal starts this clock.
"""
import math


class TransportPhaseClock:
    def __init__(self,library,candidate_config):
        self.dt=float(candidate_config.dt);self.epochs={};self.events=[]
        self.budgets={i:max(max(candidate_config.arrival_offsets_seconds),
            max(t.duration_budget_seconds for t in library.for_phase(phase)))+self.dt
            for i,phase in enumerate(library.phase_order) if phase=='transport'}

    def provisional_deadline(self,index,now,global_deadline):
        if index not in self.budgets:raise ValueError('only actual fitted transport phases')
        epoch=self.epochs.get(index)
        return min(float(global_deadline),epoch['deadline'] if epoch else float(now)+self.budgets[index])

    def confirm(self,index,issued_at,after,planned_deadline):
        if index not in self.budgets:raise ValueError('only actual fitted transport phases')
        values=(issued_at,after,planned_deadline)
        if not all(math.isfinite(float(x)) for x in values) or abs(after-issued_at-self.dt)>1e-8:
            raise ValueError('one actual counted transport dispatch required')
        if index not in self.epochs:
            deadline=min(float(planned_deadline),float(issued_at)+self.budgets[index])
            if not 0<=issued_at<after<=deadline+1e-8<=10.+1e-8:raise ValueError('transport must fit original global/phase deadline')
            self.epochs[index]=dict(issued_at=float(issued_at),confirmed_after=float(after),deadline=deadline,
                original_maximum_phase_budget=self.budgets[index])
            self.events.append(dict(event='transport_clock_started_after_actual_prefix_dispatch',phase_index=index,**self.epochs[index]))
        else:
            if after>self.epochs[index]['deadline']+1e-8:raise ValueError('confirmed transport exceeded original deadline')
            self.events.append(dict(event='transport_clock_not_reset',phase_index=index,time=float(after),deadline=self.epochs[index]['deadline']))

    def remaining(self,index,now):
        epoch=self.epochs.get(index)
        return None if epoch is None else max(0.,epoch['deadline']-float(now))
