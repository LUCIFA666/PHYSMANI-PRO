"""A gripper delay counts physical ticks, independently of pose replanning."""
class GripperSchedule:
    def __init__(self, delay_ticks=10):
        if type(delay_ticks) is not int or delay_ticks < 1:
            raise ValueError('Positive physical delay required')
        self.delay_ticks=delay_ticks
        self.pending=None
        self.start_tick=None

    def cancel(self):
        self.pending=self.start_tick=None

    def request(self, opening, actual_request, tick):
        if opening not in (0.,1.):raise ValueError('Binary opening required')
        if opening==actual_request:
            self.cancel()
        elif opening!=self.pending:
            self.pending=opening
            self.start_tick=tick

    def due(self, tick):
        return self.pending if self.pending is not None and tick-self.start_tick>=self.delay_ticks-1 else None

    def acknowledge(self, issued):
        if issued is not None:
            if issued!=self.pending:raise ValueError('Unscheduled gripper write')
            self.cancel()
