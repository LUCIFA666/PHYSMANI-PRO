"""Read-only witnesses for an explicitly selected task phase's existing queries."""
from copy import deepcopy
from types import MethodType
import numpy as np
from .public_collision_diagnostic_v6 import CollisionObserver as OriginalObserver


class FocusedCollisionObserver(OriginalObserver):
    def __init__(self,actor,env,*,phases=('insert',)):
        self.phases=tuple(phases)
        super().__init__(actor,env)

    def _instrument(self):
        service=self.service
        original_check=service.check; original_collision=self.actor.robot_model.collision
        original_arm=self.env.arm.check_arm_collision
        ghosts=list(service.static_ghosts)
        for body in service.estimated_bodies.values():ghosts.extend(body.ghosts())
        def sample(instance,q,offset):
            previous=self.sample
            self.sample=dict(future_offset=float(offset),sampled_joints=np.asarray(q).tolist())
            try:return original_collision(q,offset)
            finally:self.sample=previous
        self.actor.robot_model.collision=MethodType(sample,self.actor.robot_model)
        def check(instance,*a,**kw):
            port=self.actor.collision_port;record=None
            if port.bound is not None and port.bound['phase'] in self.phases and len(self.events)<512:
                record=dict(time=float(port.frame[0].robot.timestamp),sample=deepcopy(self.sample),
                    binding=deepcopy(port.bound),attachments=sorted(service.estimated_attachments),
                    geometry_modes=deepcopy(kw.get('geometry_modes')),arm=None,held_world=[])
            previous=self.current;self.current=record
            try:
                result=original_check(*a,**kw)
                if record is not None and bool(result):
                    record['collision']=True;self.events.append(record)
                return result
            finally:self.current=previous
        service.check=MethodType(check,service)
        def arm(*a,**kw):
            result=original_arm(*a,**kw)
            if self.current is not None:self.current['arm']=dict(collision=bool(result),extra_partner_queries=False)
            return result
        self.env.arm.check_arm_collision=arm
        for ghost in ghosts:
            original=ghost.check_collision
            def pair(other=None,original=original,ghost=ghost):
                result=original(other)
                if self.current is not None and bool(result):
                    self.current['held_world'].append(dict(held=self.geometry(ghost),obstacle=self.geometry(other)))
                return result
            ghost.check_collision=pair

    def receipt(self):
        return dict(super().receipt(),schema='r10_focused_existing_collision_queries_v1',
            phases=list(self.phases),maximum_rejection_records=512,
            diagnostic_window='explicit phases only; record existing query results',
            extra_read_only_arm_ghost_queries=0,policy_commands_modified=False)
