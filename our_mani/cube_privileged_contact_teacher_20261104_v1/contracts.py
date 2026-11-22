"""Only the predeclared fit8/selection4 parents may run this private teacher."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]
SCHEMA='Cube_privileged_canonical_P1_contact_teacher_TRAIN12_20261104_v1'
PARTITION=ROOT/'artifacts/framework/local_mpc_20261102/cube_matched_motion_candidate_partition_20261104_v1/partition.json'
PARTITION_SHA='38ac76250d382d36b3c28b01cc8551a25cc95edb850a3c21cc9bf1b47d3ade6a'
BASE=ROOT/'artifacts/framework/local_mpc_20261102/cube_trueclock_motion_capture_preparation_v2/execution_source_freeze.json'
BASE_SHA='0f418fa42deea4befdf3eb9d0b6875de6005612e4f134aca043c8d79de51866e'
REFERENCE=ROOT/'config/framework/P4_position_dwell_v72_development_canaries_v1/00_val_static.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def validate_plan(path):
    v=read(path)
    if v.get('schema')!=SCHEMA or v.get('formal_public_task_score')is not False:
        raise ValueError('Explicit privileged TRAIN-only contact teacher required')
    if sha(PARTITION)!=PARTITION_SHA or v.get('partition')!={'path':str(PARTITION),'sha256':PARTITION_SHA}:
        raise ValueError('Exact same independent split partition required')
    expected=[x for x in read(PARTITION)['rows']if x['role']in ('fit','selection')]
    if len(v.get('parents',[]))!=12:raise ValueError('All fixed12 teacher attempts required, including failures')
    for old,parent in zip(expected,v['parents']):
        if any(parent.get(k)!=x for k,x in old.items()):raise ValueError('Parent source/seed/split changed')
        if parent['role']=='heldout'or parent.get('contact_teacher_capture_allowed')is not True:
            raise ValueError('No heldout teacher capture')
        if parent.get('episode_id')!=parent['parent_id']+'_teacherpass_v1':raise ValueError('Independent teacher-pass episode id required')
        if parent.get('budget_ticks')!=160:raise ValueError('Original8seconds/160ticks maximum required')
    if v.get('reference_config')!={'path':str(REFERENCE),'sha256':sha(REFERENCE)}:
        raise ValueError('Original P1 physics/planner configuration reference required')
    if v.get('retry_or_outcome_selection')is not False:raise ValueError('No success-only selection or retries')
    return v
def source_map(path):
    validate_plan(path)
    if sha(BASE)!=BASE_SHA:raise ValueError('Frozen observation/mechanics dependency changed')
    prior=read(BASE);sources=dict(prior['files_sha256']);sources[str(BASE.relative_to(ROOT))]=BASE_SHA
    paths=[Path(__file__).with_name(n)for n in ('__init__.py','contracts.py','teacher.py','observer.py','worker.py')]
    paths += [ROOT/'scripts'/n for n in ('collect_cube_contact_teacher_TRAIN_20261104_v1.py','launch_cube_contact_teacher_TRAIN_20261104_v1.py','run_cube_contact_teacher_TRAIN_20261104_v1.sh')]
    paths += [Path(path).resolve(),PARTITION]
    for p in paths:sources[str(p.relative_to(ROOT))]=sha(p)
    for name,value in sources.items():
        if sha(ROOT/name)!=value:raise ValueError('Teacher source changed: '+name)
    native=prior['native_files_sha256']
    for name,value in native.items():
        if sha(name)!=value:raise ValueError('Original Native sources changed')
    return sources,native
def check_freeze(plan,path,digest):
    if sha(path)!=digest:raise ValueError('External exact teacher source freeze SHA required')
    v=read(path);s,n=source_map(plan)
    if v.get('schema')!='Cube_privileged_contact_teacher_execution_freeze_v1'or v.get('files_sha256')!=s or v.get('native_files_sha256')!=n:
        raise ValueError('Complete prospectively frozen teacher code/model/Native/config closure required')
    return s,n
