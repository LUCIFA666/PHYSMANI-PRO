class HeldDescent:
    """Detect a commanded lowering segment after lifting a held object."""
    def __init__(self):self.peak=None

    def observe(self,*,held,goal_z):
        if not held:
            self.peak=None
            return False
        self.peak=goal_z if self.peak is None else max(self.peak,goal_z)
        return goal_z<self.peak-.05
