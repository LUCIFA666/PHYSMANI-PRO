"""Oracle-only view of the template grasp frame; real robot is unchanged."""
import numpy as np
from scipy.spatial.transform import Rotation

class Tip:
    def __init__(self,actual,delta):self.actual=actual;self.delta=delta
    def get_matrix(self):
        matrix=np.asarray(self.actual.get_matrix(),float)
        delta=self.delta()
        return matrix if delta is None else matrix@np.linalg.inv(delta)
    def get_pose(self):
        matrix=self.get_matrix();return np.r_[matrix[:3,3],Rotation.from_matrix(matrix[:3,:3]).as_quat()]
    def get_position(self):return self.get_matrix()[:3,3].copy()

class Arm:
    def __init__(self,actual,delta):self.actual=actual;self.delta=delta
    def get_tip(self):return Tip(self.actual.get_tip(),self.delta)
    def __getattr__(self,name):
        if name.startswith('get_'):return getattr(self.actual,name)
        raise AttributeError('Oracle arm mutations unavailable: '+name)

class Robot:
    def __init__(self,actual,delta):self.arm=Arm(actual.arm,delta);self.gripper=actual.gripper
