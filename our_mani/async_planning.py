"""Nonblocking dynamics forecasting; simulator IK stays on the control thread."""
from concurrent.futures import ThreadPoolExecutor

from .pipeline import Decision, Stage


class AsyncPlanning:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='our-mani-dynamics')
        self.future = None

    def step(self, objects, robot, now):
        if self.pipeline.stage != Stage.SEEK:
            return self.pipeline.step(objects, robot, now)
        if self.future is None:
            if not objects:
                return Decision(Stage.SEEK, 'hold', details={'reason': 'waiting_for_target'})
            self.future = self.executor.submit(self.pipeline.dynamics.predict, tuple(objects),
                                               self.pipeline.horizon, self.pipeline.forecast_dt)
            return Decision(Stage.SEEK, 'hold', details={'reason': 'forecast_in_progress'})
        if not self.future.done():
            return Decision(Stage.SEEK, 'hold', details={'reason': 'forecast_in_progress'})
        forecast = self.future.result()
        self.future = None
        # Current, fresh observations enter all feasibility and handover checks.
        return self.pipeline.step(objects, robot, now, forecast=forecast)

    def close(self):
        self.pipeline.invalidate_actions()
        self.executor.shutdown(wait=True)
