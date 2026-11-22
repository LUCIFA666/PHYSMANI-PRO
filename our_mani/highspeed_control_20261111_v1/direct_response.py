"""TRAIN-calibrated response for the actual position-only motor interface."""
from pathlib import Path
import hashlib,json
import numpy as np


class DirectPositionResponse:
    def __init__(self,path,expected_sha256):
        raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('Direct servo parameter SHA differs')
        value=json.loads(raw)
        if value.get('schema')!='TRAIN_direct_position_servo_diagonal_20261111_v1' or value.get('dt')!=.05 or value.get('fit_role')!='TRAIN_development':
            raise ValueError('Explicit TRAIN-only direct servo calibration required')
        if value.get('object_truth_used')is not False or value.get('future_object_states_used')is not False:
            raise ValueError('Robot-only servo calibration required')
        self.coefficients=np.asarray(value['coefficients'],float)
        if self.coefficients.shape!=(7,3) or not np.isfinite(self.coefficients).all():raise ValueError('Finite direct servo coefficients required')
        self.coefficients.setflags(write=False)
        self.evidence=value

    def predict_joint(self,q,dq,qref,vref,aref,*,dt=.05):
        if dt!=.05:raise ValueError('Direct servo response is fitted for 50ms')
        q,dq,qref=(np.asarray(v,float) for v in (q,dq,qref))
        if q.ndim!=2 or q.shape[1]!=7 or dq.shape!=q.shape or qref.shape!=q.shape or not all(np.isfinite(v).all() for v in (q,dq,qref)):
            raise ValueError('Finite q/dq/qref[N,7] required')
        # vref/aref are controller-side reference derivatives, not motor input.
        delta=self.coefficients[:,0]*(qref-q)+self.coefficients[:,1]*dq*dt+self.coefficients[:,2]
        return dict(q=q+delta,dq=delta/dt,feasible=np.ones(len(q),bool),velocity_reference_actuated=False)
