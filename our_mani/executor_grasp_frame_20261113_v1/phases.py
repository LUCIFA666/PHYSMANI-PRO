class LatchedHeldContact:
    def __init__(self,drop=.05):self.drop=drop;self.peak=None;self.active=False
    def observe(self,*,held,goal_z):
        if not held:
            self.peak=None;self.active=False;return False
        self.peak=goal_z if self.peak is None else max(self.peak,goal_z)
        self.active=self.active or goal_z<self.peak-self.drop
        return self.active
