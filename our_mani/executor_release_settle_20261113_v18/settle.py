import math

class VerticalReleaseSettle:
    def __init__(self):self.pending=False;self.since=None
    def start(self,height):self.pending=True;self.height=float(height);self.since=None
    def reset(self):self.pending=False;self.since=None
    def qualify(self,clock,measured_vz,reference_vz,reference_height_gap):
        stable=(all(math.isfinite(x)for x in (measured_vz,reference_vz,reference_height_gap))
            and abs(measured_vz)<=.01 and abs(reference_vz)<=.01 and abs(reference_height_gap)<=.002)
        if not stable:self.since=None
        elif self.since is None:self.since=clock
        return bool(self.pending and self.since is not None and clock-self.since>=.10-1e-6)
