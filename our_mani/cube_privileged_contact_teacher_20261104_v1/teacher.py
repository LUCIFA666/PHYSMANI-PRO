"""Planner-only Cube symmetry frame; raw Native labels remain unmodified."""
from copy import deepcopy
import numpy as np
from ..local_mpc_20261102.worker import run_p1


class CanonicalCubeTeacherPolicyView:
    """A deliberately privileged policy view, not a public-perception adapter.

    The Native symmetric Cube may use Ry(pi) as its body coordinates. Original
    P1 already compensates its TCP orientation, but its local+Z approach would
    otherwise point down. A canonical I frame defines contact/approach for the
    teacher only. No Native object rotation/position or motion setter is called.
    """
    def __init__(self,backend):self.backend=backend
    def __getattr__(self,name):return getattr(self.backend,name)
    def read_state(self):
        original=self.backend.read_state();state=deepcopy(original)
        state['object_quaternion']=np.array([0.,0.,0.,1.])
        state['object_angular_velocity']=np.zeros(3)
        state['teacher_planning_reference']='Cube_symmetry_world_I_top_down_not_Native_body_attitude'
        state['teacher_privileged_current_Cube_centre_velocity_held_goal_used']=True
        state['formal_public_policy_result']=False
        return state


def run_privileged_teacher(backend,executor,output,config):
    """Original P1 CV, geometry, CEM, phase thresholds and action checks."""
    if type(backend)is not CanonicalCubeTeacherPolicyView or executor.backend is not backend:
        raise TypeError('Explicit private TRAIN teacher view must own original executor feedback')
    return run_p1(backend,executor,output,config,capture_rgbd=False)
