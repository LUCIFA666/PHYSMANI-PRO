"""Portable, no-CLI access to the byte-preserved B research mathematics.

Archived scripts retain historical paths as evidence. Compile only their
hash-checked pure function definitions, with explicit project dependencies;
do not execute their imports, main(), filesystem loaders, or real-data fit.
The tests below use only synthetic fixtures for regression/causality checks.
"""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from our_mani.local_mpc_p2_20261103_v2.reference import ReferenceLimits,reference_step


PROJECT=Path(__file__).resolve().parents[1]
ARCHIVE=PROJECT/'artifacts/framework/local_mpc_20261102/P2_coupled_response_v1'


def _source(relative):
    path=ARCHIVE/relative;raw=path.read_bytes()
    manifest=json.loads((ARCHIVE/'archive_manifest.json').read_text())
    key=str(path.relative_to(PROJECT))
    if hashlib.sha256(raw).hexdigest()!=manifest['files_sha256'][key]:
        raise RuntimeError('archived research source SHA differs: '+relative)
    return ast.parse(raw,filename=str(path)),path


def _constant(tree,name):
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id==name for x in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('missing archived constant '+name)


def _load_math():
    base,_=_source('research/local_mpc_servo_ridge_20261103.py')
    tree,path=_source('research/local_mpc_servo_coupled_ridge_B_20261103.py')
    dt,horizon,ridge=_constant(base,'DT'),_constant(base,'HORIZON'),_constant(tree,'RIDGE')
    assert (dt,horizon,ridge)==(.05,16,1e-3)
    wanted={'features','fit','predict','metric','rollouts'}
    definitions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in wanted]
    if {node.name for node in definitions}!=wanted:raise ValueError('archived pure B functions differ')
    namespace=dict(np=np,Counter=Counter,DT=dt,HORIZON=horizon,RIDGE=ridge,reference_step=reference_step)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(path),'exec'),namespace)
    return SimpleNamespace(**{name:namespace[name] for name in wanted})


RESEARCH=_load_math()


class ToyFK:
    def forward(self,q):
        q=np.asarray(q);result=np.broadcast_to(np.eye(4),(*q.shape[:-1],4,4)).copy()
        result[...,:3,3]=q[...,:3]
        return result


def episode(n=8,split='train'):
    q=np.broadcast_to(np.linspace(0.,.24,n+1)[:,None],(n+1,7)).copy()
    dq=np.r_[np.zeros((1,7)),np.diff(q,axis=0)/.05]
    refs=np.concatenate((q[:-1]+.02,np.full((n,7),.2),np.zeros((n,7))),axis=-1)
    goals=np.broadcast_to(np.linspace(.5,.8,n)[:,None],(n,7)).copy();parts=[]
    for i in range(n):
        if i:
            h=dict(q_ref=refs[i-1,:7],dq_ref=refs[i-1,7:14],ddq_ref=refs[i-1,14:21],
                   goal_q=goals[i-1],valid=True,initialized=True,gripper_target=1.)
        else:
            h=dict(q_ref=np.zeros(7),dq_ref=np.zeros(7),ddq_ref=np.zeros(7),goal_q=np.zeros(7),
                   valid=False,initialized=False,gripper_target=1.)
        parts.append(dict(h=h))
    return dict(q=q,dq=dq,refs=refs,goals=goals,parts=parts,split=split,
        episode_id='fixture_'+split,parent_id='parent_'+split,tcp=q[:,:3].copy(),
        held=np.zeros(n+1,bool),phase=np.array(['approach']*n),macro_tick=np.arange(n)%2,
        limits=ReferenceLimits(np.full(7,-2.),np.full(7,2.),np.full(7,3.)))
