"""Package new completed traces losslessly after native cleanup.

The full original evaluator JSON is stored byte-for-byte in gzip with hashes.
The main case retains outcomes, identities and every actual motion window.
Large diagnostic lists become explicit pointers into the full trace. Existing
completed outputs are never processed or changed by this module.
"""
from pathlib import Path
import gzip
import hashlib
import shutil
import numpy as np
from .evaluation import atomic_json,file_sha


def json_pointer(parent,key):
    return parent+'/'+str(key).replace('~','~0').replace('/','~1')


def shell(value,pointer=''):
    if isinstance(value,np.ndarray):return shell(value.tolist(),pointer)
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:shell(v,json_pointer(pointer,k)) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        if len(value)<=16 and all(v is None or isinstance(v,(str,int,float,bool)) for v in value):return value
        return dict(archived_list=True,length=len(value),full_trace_JSON_pointer=pointer)
    return value


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value


def compact_record(record):
    result=shell(record)
    # These arrays are the complete independently measured execution evidence,
    # kept directly readable by existing motion and phase audits.
    actor=record.get('actor_metrics')
    if isinstance(actor,dict):
        out=result['actor_metrics']
        for key in ('public_events','observable_skill_events','candidate_selection_events','gripper_timing_events'):
            if key in actor:out[key]=plain(actor[key])
        for key in ('continuous_reference_feedforward','task_dispatch_v43'):
            if isinstance(actor.get(key),dict):
                for name in ('observed_events','actual_dispatch_confirmations'):
                    if name in actor[key]:out[key][name]=plain(actor[key][name])
        if isinstance(actor.get('executor_public_timeline'),dict):
            out['executor_public_timeline']=plain(actor['executor_public_timeline'])
    if 'continuation_attempts' in record:result['continuation_attempts']=plain(record['continuation_attempts'])
    # Baseline observers use this evidence instead of the geometry manager.
    if 'all_stage_counted_joint_motion' in record:
        result['all_stage_counted_joint_motion']=plain(record['all_stage_counted_joint_motion'])
    if isinstance(record.get('alignment_receipt'),dict):
        source=record['alignment_receipt'].get('all_stage_counted_joint_motion_v191')
        if source is not None:result['alignment_receipt']['all_stage_counted_joint_motion_v191']=plain(source)
    return result


def uncompressed_sha(path):
    digest=hashlib.sha256()
    with gzip.open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def run_archived_case(prepared,*,backend=None,runner=None):
    if runner is None:
        from scripts.evaluate_multitask_case_r10 import run_case
        runner=run_case
    out=Path(prepared['out']);archive=out/'full_evaluator_case.json.gz'
    temporary=out/'full_evaluator_case.json.gz.tmp';raw=out/'case.json';saved=out/'full_evaluator_case.uncompressed.tmp'
    if any(p.exists() for p in (archive,temporary,raw,saved)):
        raise FileExistsError('archive wrapper only accepts a fresh never-started case')
    record=runner(prepared,backend=backend)
    if record.get('complete') is not True or record.get('cleanup_completed') is not True:
        raise ValueError('archive only after completed native cleanup')
    original_sha=file_sha(raw);original_bytes=raw.stat().st_size
    with raw.open('rb') as source,temporary.open('xb') as target:
        with gzip.GzipFile(fileobj=target,mode='wb',filename='',mtime=0,compresslevel=1) as compressed:
            shutil.copyfileobj(source,compressed,length=1024*1024)
    if uncompressed_sha(temporary)!=original_sha:
        raise ValueError('full trace must round-trip byte-for-byte before packaging')
    temporary.rename(archive)
    compact=compact_record(record)
    compact['full_trace_archive']=dict(schema='r10_lossless_original_evaluator_trace_gzip_v1',
        path=str(archive.resolve()),sha256=file_sha(archive),uncompressed_sha256=original_sha,
        uncompressed_bytes=original_bytes,compressed_bytes=archive.stat().st_size,
        original_outcomes_identities_and_actual_motion_preserved=True,
        recorded_after_native_cleanup=True,policy_or_physics_changed=False,
        packaging_source_sha256=file_sha(__file__))
    # Only this just-created temporary uncompressed copy is removed. The full
    # raw evaluator bytes are already verified and preserved in the archive.
    raw.rename(saved)
    try:atomic_json(raw,compact)
    except BaseException:
        if not raw.exists():saved.rename(raw)
        raise
    saved.unlink()
    return compact
