"""CPU inference bridge for the P2 17-state / 22-reference residual model.

This does not fit weights, launch Native, call CUDA or infer contact. A loaded
model remains a recorded-reference diagnostic model until closed-chain and
Native qualification are recorded separately. Held input stays at the actual
planning anchor, exactly as in its offline free-rollout training contract.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _matrix(value,shape,name):
    value=np.asarray(value,dtype=np.float64)
    if value.shape!=shape or not np.isfinite(value).all():raise ValueError('finite '+name+str(shape)+' required')
    return value


class TorchServoTransition:
    """Strict CPU adapter; direct construction is explicitly an untrained test.

    Production callers use from_checkpoint, which rejects partial training and
    binds weights, normalization and the validated FK snapshot. The adapter
    exposes predict_execution to JointResidualFKExecutionModel; it never turns
    the scalar mean aperture into two invented finger measurements.
    """
    def __init__(self,model,*,checkpoint_evidence=None):
        import torch
        if any(parameter.device.type!='cpu' for parameter in model.parameters()):
            raise ValueError('P2 execution inference requires an explicit CPU model')
        if any(buffer.device.type!='cpu' for buffer in model.buffers()):
            raise ValueError('P2 execution inference requires all model buffers on CPU')
        self.model=model.eval()
        self.evidence=deepcopy(checkpoint_evidence or dict(trained=False,scope='untrained_CPU_contract_test_only'))
        self.metadata=dict(device='cpu',state_order='q7_dq7_finger_open_amounts2_held_anchor1',
                           reference_order='qref7_vref7_aref7_gripper_target1',dt=.05,
                           held_input='actual_anchor_fixed_no_future_contact_label',
                           full_known_reference_chain_qualified=False,Native_task_success_measured=False,
                           checkpoint=deepcopy(self.evidence))

    @classmethod
    def from_checkpoint(cls,path,*,expected_sha256=None):
        import torch
        from .servo import JointServoResidual
        path=Path(path).resolve(strict=True)
        before=_sha(path)
        if expected_sha256 is not None and before!=expected_sha256:raise ValueError('servo checkpoint SHA mismatch')
        payload=torch.load(path,map_location='cpu',weights_only=True)
        contract=payload.get('model_contract',{})
        status=payload.get('training_status',{})
        identity=payload.get('identity',{})
        if (payload.get('training_complete') is not True or contract.get('training_complete') is not True
                or contract.get('trained') is not True or status.get('training_complete') is not True
                or status.get('status')!='complete' or type(status.get('epochs_requested')) is not int
                or status.get('epochs_requested',0)<=0
                or status.get('epochs_completed')!=status.get('epochs_requested')
                or payload.get('optimization_steps',0)<=0
                or identity.get('formal_TEST_used') is not False or payload.get('formal_TEST_used') is not False
                or contract.get('formal_TEST_used') is not False):
            raise ValueError('complete source-bound trained servo checkpoint with no formal TEST required')
        fk_identity=identity.get('fk',{})
        source_hashes=identity.get('source_sha256',{})
        if not source_hashes:raise ValueError('servo training source identity is required')
        for source,digest in source_hashes.items():
            source=Path(source).resolve(strict=True)
            if source.suffix!='.py' or _sha(source)!=digest:raise ValueError('servo training source identity mismatch')
        snapshot_path=Path(fk_identity['snapshot_path']).resolve(strict=True)
        if _sha(snapshot_path)!=fk_identity.get('snapshot_sha256'):raise ValueError('servo FK snapshot SHA mismatch')
        snapshot=json.loads(snapshot_path.read_text())
        if snapshot.get('Native_FK_validation_passed') is not True:raise ValueError('validated FK snapshot required')
        stats=payload['normalization']
        if identity.get('normalization')!=stats:raise ValueError('normalization identity differs from trained payload')
        model=JointServoResidual(snapshot,**{key:stats[key] for key in
            ('residual_q_bound','residual_dq_bound','feature_mean','feature_std')})
        expected=model.state_dict()
        state_dict=payload['state_dict']
        # strict load alone would silently replace FK/normalization buffers;
        # these buffers must first match their independently bound metadata.
        for key,value in expected.items():
            if key.startswith('fk.') or key in ('residual_q_bound','residual_dq_bound','feature_mean','feature_std'):
                if key not in state_dict or not torch.equal(value,state_dict[key]):
                    raise ValueError('servo model buffer differs from FK/normalization identity: '+key)
        model.load_state_dict(state_dict,strict=True)
        if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
            raise ValueError('servo checkpoint contains nonfinite weights')
        if _sha(path)!=before:raise ValueError('servo checkpoint changed during loading')
        model.metadata=deepcopy(contract)
        evidence=dict(path=str(path),sha256=before,trained=True,training_complete=True,
                      selected_epoch=payload.get('epoch'),optimization_steps=payload['optimization_steps'],
                      training_status=deepcopy(status),identity=deepcopy(identity),
                      full_known_reference_chain_qualified=payload.get('full_known_reference_chain_qualified',False),
                      formal_TEST_used=False)
        return cls(model,checkpoint_evidence=evidence)

    def predict_execution(self,x,references,latent,dt,*,held_anchor):
        import torch
        if dt!=.05:raise ValueError('servo network is trained for exactly 50ms transitions')
        q=np.asarray(x.get('q'),dtype=np.float64)
        if q.ndim!=2 or q.shape[1]!=7:raise ValueError('actual or predicted q[N,7] required')
        n=len(q)
        q=_matrix(q,(n,7),'q');dq=_matrix(x.get('dq'),(n,7),'dq')
        fingers=_matrix(x.get('finger_open_amounts'),(n,2),'actual two finger openings')
        if np.any((fingers<0)|(fingers>1)):raise ValueError('two actual finger openings must be in [0,1]')
        anchor=np.asarray(held_anchor)
        if anchor.shape!=(n,) or anchor.dtype!=np.bool_:raise ValueError('explicit bool held_anchor[N] required')
        reference=np.concatenate([_matrix(references.get(key),(n,7),key) for key in
                                  ('q_ref','dq_ref','ddq_ref')],axis=-1)
        target=_matrix(references.get('gripper_target'),(n,),'gripper_target')
        if not np.isin(target,(0.,1.)).all():raise ValueError('binary dispatched gripper target required')
        reference=np.c_[reference,target]
        state=np.c_[q,dq,fingers,anchor.astype(np.float64)]
        with torch.inference_mode():
            prediction=self.model(torch.as_tensor(state,dtype=torch.float32),torch.as_tensor(reference,dtype=torch.float32))
        result={key:prediction[key].detach().cpu().numpy().astype(np.float64,copy=True) for key in ('q','dq')}
        result['finger_open_amounts']=prediction['opening'].detach().cpu().numpy().astype(np.float64,copy=True)
        if (not all(np.isfinite(value).all() for value in result.values())
                or result['finger_open_amounts'].shape!=(n,2)
                or np.any((result['finger_open_amounts']<0)|(result['finger_open_amounts']>1))):
            raise ValueError('nonfinite/out-of-range servo prediction')
        result.update(feasible=np.ones(n,dtype=bool),latent=deepcopy(latent))
        return result
