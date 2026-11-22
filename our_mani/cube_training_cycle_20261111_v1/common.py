"""Prospective split/STOP and append-only evidence for this new scoped cycle."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'config/framework/cube_training_cycle_20261111_v1/plan.json'
OUT=ROOT/'artifacts/framework/cube_training_cycle_20261111_v1'
NATIVE_PYTHON='/home/ubuntu/physmani-conda/envs/physmani5090/bin/python'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open() as f: return [json.loads(line) for line in f if line.strip()]

def write(p,v):
    from ..local_mpc_20261102.records import write_new
    write_new(p,v)

def append(p,v):
    from ..local_mpc_20261102.records import append as add
    add(p,v)

def utc(): return datetime.now(timezone.utc).isoformat()

def pointer(p): return dict(path=str(Path(p).resolve(strict=True)),sha256=sha(p))

def identity(p):
    if not p.exists(): return None
    s=p.stat();return dict(sha256=sha(p),mtime_ns=s.st_mtime_ns,inode=s.st_ino)

def stopped(plan):
    if (identity(ROOT/'runtime/framework/STOP')!=plan['historical_global_stop'] or
        any(p.exists() for p in (OUT/'STOP',ROOT/'runtime/framework/cube_training_cycle_20261111_v1/STOP')) or
        any(p.exists() for p in OUT.glob('*/STOP'))):
        raise InterruptedError('New scoped Cube cycle STOP; preserve all partial data')

def verify(mapping):
    for name,digest in mapping.items():
        if sha(name)!=digest: raise RuntimeError('Frozen input/source changed: '+name)

def plan():
    p=read(PLAN)
    if p.get('schema')!='Cube_train_validation_test_cycle_20261111_v1' or p.get('training_authorized')is not True:
        raise ValueError('Explicit newly authorized Cube cycle required')
    cases=p['cases']
    for key in ('parent_id','episode_id','seed','id'):
        if len({c[key] for c in cases})!=len(cases): raise ValueError('Whole reset roles overlap: '+key)
    if set(c['role'] for c in cases)!={'fit','selection','validation','test'}: raise ValueError('Four fixed roles required')
    if any(c['task']not in p['tasks']or c['config']['task_time_limit_seconds']!=60 or c['config']['budget_ticks']!=1200 for c in cases):
        raise ValueError('Only scoped Cube tasks with actual60s budget required')
    stopped(p);return p

def parents(p,role): return [c['parent_id']for c in p['cases']if c['role']==role]
def heldout_parents(p): return [c['parent_id']for c in p['cases']if c['role']in('validation','test')]

def training_sources():
    from ..cube_motion_residual_20261104_v1.runtime import source_map as motion_sources
    from ..cube_matched_candidate_20261104_v1.model import model_source_map
    files=list((ROOT/'our_mani/cube_training_cycle_20261111_v1').glob('*.py'))
    files += [PLAN,ROOT/'scripts/train_cube_motion_residual_20261104_v1.py',ROOT/'scripts/train_cube_matched_candidate_20261104_v1.py',
              ROOT/'our_mani/current_cube_rgbd_centre_large_20261104_v2/training.py',
              ROOT/'our_mani/cube_candidate_teacher_20261104_v5/labels.py']
    m={str((ROOT/k).resolve()):v for k,v in motion_sources().items()}
    m.update(model_source_map());m.update({str(p.resolve()):sha(p)for p in files})
    return m

def status(stage,**kw):
    v=dict(stage=stage,updated_utc=utc(),**kw)
    tmp=OUT/'status.tmp';tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,OUT/'status.json');append(OUT/'events.jsonl',v)
