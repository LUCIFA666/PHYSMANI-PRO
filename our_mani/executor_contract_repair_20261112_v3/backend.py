"""Unmodified Native stepping with separate robot/table contact diagnostics."""
from ..multitask_executor_cycle_20261112_v6.backend import TaskExecutionBackend as ContinuousBackend


class TaskExecutionBackend(ContinuousBackend):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from pyrep.objects.shape import Shape
        from .clearance import HardwareTableClearance
        self._table_handle=Shape('diningTable').get_handle()
        self._hardware_handles={Shape(name).get_handle():name for name in HardwareTableClearance.names}
        self.prohibited_contact_frames=0
        self.prohibited_contact_records=0
        self.first_prohibited_contact_time=None
        self.on_hardware_contact=None

    def tick(self,phase='hardware_clearance_execution'):
        super().tick(phase)
        from pyrep.backend import sim
        before=self._clock_tuple();events=[]
        for contact in sim.simGetContactInfo(sim.sim_handle_all,True):
            handles=contact['contact_handles']
            if self._table_handle not in handles:continue
            for handle in handles:
                if handle in self._hardware_handles:
                    events.append(dict(robot_body=self._hardware_handles[handle],contact=contact['contact']))
        if events:
            self.prohibited_contact_frames+=1;self.prohibited_contact_records+=len(events)
            if self.first_prohibited_contact_time is None:self.first_prohibited_contact_time=self.clock()
            if self.on_hardware_contact is not None:
                self.on_hardware_contact(dict(clock=self.clock(),tick=self._ticks,events=events))
        if self._clock_tuple()!=before:raise RuntimeError('Contact diagnostics advanced physics')

    def hardware_contact_audit(self):
        return dict(actual_frames_checked=self._ticks,prohibited_robot_table_contact_frames=self.prohibited_contact_frames,
                    prohibited_contact_records=self.prohibited_contact_records,first_contact_time=self.first_prohibited_contact_time,
                    finger_contacts_permitted=True,contacts_not_used_by_NN=True,original_success_predicate_unchanged=True)
