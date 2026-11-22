"""Nominal moving-object collision probes; restore the live scene after each."""


class ForecastCollisionScene:
    def __init__(self, arm, tracked_shapes):
        self.arm, self.shapes = arm, dict(tracked_shapes)
        self.forecast, self.origin = None, None

    def bind(self, forecast, robot_time):
        self.forecast, self.origin = forecast, float(robot_time)

    def check(self, future_offset):
        if self.forecast is None:
            return True  # Dynamic feasibility cannot silently become static.
        saved = []
        try:
            poses = {identity:self.forecast.at(identity, self.origin+future_offset)[0]
                     for identity in self.shapes}
        except (ValueError, IndexError):
            return True
        try:
            for identity, shape in self.shapes.items():
                saved.append((shape, shape.get_pose()))
                # No physics step or dynamic reset occurs during a query.
                shape.set_pose(poses[identity].xyzw().tolist(), reset_dynamics=False)
            return bool(self.arm.check_arm_collision())
        finally:
            for shape, pose in reversed(saved):
                shape.set_pose(pose, reset_dynamics=False)
