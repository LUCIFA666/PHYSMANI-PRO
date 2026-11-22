"""Frozen NumPy joint response B; independent gripper response is explicit.

No training, Torch, Native, FK/IK, clipping, or contact inference occurs here.
The basic callable fits JointResidualFKExecutionModel's joint-only interface.
CoupledWithFingerResponse supplies its rich interface only when a separate
finger predictor is explicitly injected. Position VAL evidence never qualifies
grasp/contact/gripper performance or formal generalization.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA='frozen_TRAIN_only_coupled_joint_delta_ridge_B_20261103_v1'
DT=.05
FEATURE_ORDER=[f'{block}_j{j+1}' for block in
    ('qref_minus_q','dq_dt','vref_dt','aref_dt_squared','sin_q','cos_q') for j in range(7)]+['bias']


def _array(value,shape,name):
    array=np.asarray(value,dtype=np.float64)
    if array.shape!=shape or not np.isfinite(array).all():
        raise ValueError('finite '+name+str(shape)+' required')
    return array.copy()


def _readonly(value,shape,name):
    result=_array(value,shape,name);result.setflags(write=False);return result


class FrozenCoupledResponse:
    """Joint-only response; callable dispatches no finger/attachment forecasts."""
    def __init__(self,parameters,*,parameter_sha256=None):
        if parameters.get('schema')!=SCHEMA or parameters.get('feature_order')!=FEATURE_ORDER:
            raise ValueError('exact frozen B schema and 43-feature order required')
        fit=parameters.get('fit_source',{});normal=parameters.get('normalization',{})
        hyper=parameters.get('hyperparameters',{})
        if (fit.get('parameter_fit_split')!='train' or normal.get('fit_split')!='train'
                or parameters.get('formal_TEST_used') is not False
                or hyper.get('ridge')!=1e-3 or hyper.get('dt')!=DT
                or hyper.get('input_dimension')!=43 or hyper.get('output_dimension')!=7):
            raise ValueError('fixed TRAIN-only B calibration and 50ms schema required')
        self.mean=_readonly(normal.get('mean'),(42,),'TRAIN feature mean')
        self.scale=_readonly(normal.get('scale'),(42,),'TRAIN feature scale')
        if np.any(self.scale<=0):raise ValueError('positive TRAIN feature scale required')
        self.coefficient=_readonly(parameters.get('coefficient_scaled'),(43,7),'scaled B coefficients')
        self._parameters=deepcopy(parameters)
        self.parameter_sha256=parameter_sha256
        self.evidence=dict(model_schema=SCHEMA,parameter_sha256=parameter_sha256,
            fit_split='train',normalization_fit_split='train',frozen_parameters=True,
            joint_response_only=True,dq_is_position_finite_difference=True,
            position_validation_only=True,formal_TEST_used=False,
            gripper_model_included=False,attachment_inferred=False,contact_model_trained=False,
            closed_loop_task_success_measured=False,full_known_reference_chain_qualified=False,
            globally_qualified=False)

    @classmethod
    def from_json(cls,path,*,expected_sha256):
        if (not isinstance(expected_sha256,str) or len(expected_sha256)!=64
                or any(c not in '0123456789abcdef' for c in expected_sha256.lower())):
            raise ValueError('explicit expected SHA256 of frozen B parameter JSON required')
        raw=Path(path).read_bytes();actual=hashlib.sha256(raw).hexdigest()
        if actual!=expected_sha256.lower():raise ValueError('frozen B parameter file SHA256 differs')
        return cls(json.loads(raw),parameter_sha256=actual)

    def parameters(self):return deepcopy(self._parameters)

    def predict_joint(self,q,dq,qref,vref,aref,*,dt=DT):
        if dt!=DT:raise ValueError('B was fitted for exactly one 50ms physical transition')
        q=np.asarray(q,dtype=np.float64)
        if q.ndim!=2 or q.shape[-1]!=7:raise ValueError('q[N,7] required')
        shape=q.shape;q=_array(q,shape,'q');dq=_array(dq,shape,'dq')
        qref=_array(qref,shape,'qref');vref=_array(vref,shape,'vref');aref=_array(aref,shape,'aref')
        continuous=np.concatenate((qref-q,dq*DT,vref*DT,aref*DT*DT,np.sin(q),np.cos(q)),axis=-1)
        design=np.c_[(continuous-self.mean)/self.scale,np.ones(len(q))]
        delta=design@self.coefficient
        next_q=q+delta;next_dq=delta/DT
        if not np.isfinite(next_q).all() or not np.isfinite(next_dq).all():
            raise ValueError('nonfinite B response; no substitute action was produced')
        return dict(q=next_q,dq=next_dq,feasible=np.ones(len(q),dtype=bool),
                    feasibility_scope='finite numerical response; outer joint bounds and Native collision still required',
                    position_validation_only=True)

    def __call__(self,q,dq,qref,vref,aref,latent=None,dt=DT):
        """Basic generic transition; existing outer finger model stays separate."""
        result=self.predict_joint(q,dq,qref,vref,aref,dt=dt)
        result['latent']=deepcopy(latent)
        return result


class CoupledWithFingerResponse:
    """Rich generic transition: B joints plus an explicitly separate finger model.

    finger_response.predict_execution has the original generic signature and
    must return finger_open_amounts[N,2] and feasible[N]. Its joint predictions
    are discarded. Its held/contact outputs, if any, are never forwarded.
    """
    def __init__(self,joint_response,finger_response):
        if not isinstance(joint_response,FrozenCoupledResponse):
            raise TypeError('frozen B joint response required')
        if not callable(getattr(finger_response,'predict_execution',None)):
            raise TypeError('explicit independent rich finger predictor required')
        self.joint_response,self.finger_response=joint_response,finger_response
        self.evidence=deepcopy(joint_response.evidence)
        self.evidence.update(gripper_model_included=True,gripper_quality_inherited_not_requalified=True,
            finger_response_evidence=deepcopy(getattr(finger_response,'evidence',{})),
            globally_qualified=False,full_known_reference_chain_qualified=False)

    def predict_execution(self,x,references,latent,dt,*,held_anchor):
        q=np.asarray(x.get('q'),dtype=np.float64)
        if q.ndim!=2 or q.shape[-1]!=7:raise ValueError('q[N,7] required')
        n=len(q)
        anchor=np.asarray(held_anchor)
        if anchor.shape!=(n,) or anchor.dtype!=np.bool_:raise ValueError('explicit measured held_anchor[N] bool required')
        fingers=_array(x.get('finger_open_amounts'),(n,2),'two actual finger apertures')
        if np.any((fingers<0)|(fingers>1)):raise ValueError('actual finger aperture outside[0,1]')
        joint=self.joint_response.predict_joint(q,x.get('dq'),references.get('q_ref'),
            references.get('dq_ref'),references.get('ddq_ref'),dt=dt)
        target=_array(references.get('gripper_target'),(n,),'known gripper target')
        if not np.isin(target,(0.,1.)).all():raise ValueError('binary known gripper command required')
        # Copies isolate the position forecast from an injected implementation.
        result=self.finger_response.predict_execution(deepcopy(x),deepcopy(references),
            deepcopy(latent),dt,held_anchor=anchor.copy())
        predicted_fingers=_array(result.get('finger_open_amounts'),(n,2),'predicted independent finger apertures')
        if np.any((predicted_fingers<0)|(predicted_fingers>1)):
            raise ValueError('independent finger response outside[0,1]; no clipping applied')
        finger_ok=np.asarray(result.get('feasible'))
        if finger_ok.shape!=(n,) or finger_ok.dtype!=np.bool_:
            raise ValueError('independent finger feasible[N] bool required')
        return dict(q=joint['q'],dq=joint['dq'],finger_open_amounts=predicted_fingers,
                    feasible=joint['feasible']&finger_ok,latent=deepcopy(result.get('latent',latent)),
                    position_validation_only=True,globally_qualified=False)
