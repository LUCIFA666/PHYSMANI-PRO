"""Small bootstrap ensemble and task-specific affine probability calibration."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from scipy.spatial import cKDTree
import torch
from ..local_learning import MLP

TASKS=('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed')


def fit_calibration(logits,labels,tasks,weights=None):
    logits=np.asarray(logits,dtype=float);labels=np.asarray(labels,dtype=float)
    indices=np.array([TASKS.index(t) for t in tasks])
    weights=np.ones(len(labels)) if weights is None else np.asarray(weights,dtype=float)
    weights=weights/weights.sum()
    def objective(parameters):
        z=parameters[0]*logits+parameters[1:][indices]
        nll=np.sum(weights*(np.logaddexp(0.,z)-labels*z))
        # A small calibration set cannot support extreme slopes/intercepts.
        return nll+.005*(parameters[0]-1.)**2+.001*np.sum(parameters[1:]**2)
    fitted=minimize(objective,[1.,0.,0.],method='L-BFGS-B',bounds=[(.2,5.),(-5.,5.),(-5.,5.)])
    if not fitted.success:raise ValueError('affine calibration optimization failed')
    return dict(slope=float(fitted.x[0]),task_intercepts=dict(zip(TASKS,map(float,fitted.x[1:]))),
        method='positive-slope affine logits with task intercepts; regularized on disjoint calibration sources')


def calibrated_probabilities(probabilities,tasks,calibration):
    z=logit(np.clip(np.asarray(probabilities),1e-6,1-1e-6))
    intercept=np.array([calibration['task_intercepts'][t] for t in tasks])
    return expit(calibration['slope']*z+intercept)


class CandidateProbability:
    def __init__(self,checkpoint,*,controller_sha256=None):
        path=Path(checkpoint)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        self.mean,self.scale=saved['feature_mean'],saved['feature_std']
        self.metadata=saved['metadata'];self.calibration=saved['calibration']
        if controller_sha256 and self.metadata['controller_sha256']!=controller_sha256:
            raise ValueError('candidate probability controller does not match')
        self.models=[]
        for state in saved['state_dicts']:
            model=MLP(len(self.mean),1,saved['hidden']);model.load_state_dict(state);model.eval()
            self.models.append(model)
        self.support=json.loads(path.with_name('value_support.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.support['value_checkpoint_sha256']:
            raise ValueError('candidate support belongs to another checkpoint')
        self.tree=cKDTree(self.support['normalized_fit_features'])

    @torch.inference_mode()
    def predict(self,features,task):
        x=np.asarray(features,dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=len(self.mean) or not np.isfinite(x).all():
            raise ValueError('invalid full-candidate probability input')
        tensor=torch.tensor((x-self.mean)/self.scale,dtype=torch.float32)
        members=np.stack([torch.sigmoid(model(tensor)).numpy()[:,0] for model in self.models])
        probability=calibrated_probabilities(members.mean(0),[task]*len(x),self.calibration)
        support_z=(x-np.asarray(self.support['mean']))/np.asarray(self.support['scale'])
        distance=self.tree.query(support_z)[0]
        supported=(distance<=self.support['maximum_nearest_fit_distance'])
        supported&=(x>=np.asarray(self.support['lower'])).all(1)&(x<=np.asarray(self.support['upper'])).all(1)
        return dict(probability=probability,supported=supported,ensemble_std=members.std(0),nearest_fit_distance=distance)
