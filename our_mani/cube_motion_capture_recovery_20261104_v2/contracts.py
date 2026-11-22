"""Frozen independent TRAIN16 whole-parent roster and original task sources."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='Cube_trueclock_keep_open_TRAIN16_capture_plan_20261104_v2'
PARTITION=ROOT/'artifacts/framework/local_mpc_20261102/cube_matched_motion_candidate_partition_20261104_v1/partition.json'
PARTITION_SHA='38ac76250d382d36b3c28b01cc8551a25cc95edb850a3c21cc9bf1b47d3ade6a'
BASE_FREEZE=ROOT/'artifacts/framework/local_mpc_20261102/cube_v72_paired_development_preparation_v1/execution_source_freeze.json'
BASE_SHA='6ae962cf6be0b3173333095545876c4816a1389eb2ccb7b6689786e762c7e088'
OWN=('__init__.py','backend.py','collector.py','contracts.py','worker.py')
ENTRIES=('collect_cube_motion_TRAIN_20261104_v2.py','launch_cube_motion_TRAIN_20261104_v2.py','run_cube_motion_TRAIN_20261104_v2.sh')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def validate_plan(path):
    plan=read(path)
    if plan.get('schema')!=SCHEMA or plan.get('formal_VAL_TEST')is not False:
        raise ValueError('Generated TRAIN motion-only capture plan required')
    if sha(PARTITION)!=PARTITION_SHA or plan.get('partition')!={'path':str(PARTITION),'sha256':PARTITION_SHA}:
        raise ValueError('Exact independent whole-parent TRAIN16 partition required')
    rows=read(PARTITION)['rows']
    if len(plan.get('parents',[]))!=16:raise ValueError('Keep all16 parents regardless of visibility/outcome')
    for original,parent in zip(rows,plan['parents']):
        if any(parent.get(k)!=v for k,v in original.items()):raise ValueError('Whole-parent split identity cannot change')
        if parent.get('episode_id')!=parent['parent_id']+'_motionpass_v2':raise ValueError('Independent single motion pass id required')
        if parent.get('budget_ticks')!=160 or parent.get('variation')!=0:raise ValueError('Original160tick/variation0 schedule required')
        p=Path(parent['calibration_episode']).resolve(strict=True)
        if 'train_package_compressed'not in p.parts or p.name!='data.h5':raise ValueError('TRAIN home/camera calibration only')
    if plan.get('keep_open')is not True or plan.get('initial_actual_q_hold')is not True:
        raise ValueError('This version has no contact teacher or closing policy')
    if plan.get('retry_or_visibility_selection')is not False:raise ValueError('No resampling/failed-parent replacement')
    return plan


def source_map(path):
    validate_plan(path)
    if sha(BASE_FREEZE)!=BASE_SHA:raise ValueError('Original frozen Cube mechanical/perception dependency changed')
    sources=dict(read(BASE_FREEZE)['files_sha256'])
    sources[str(BASE_FREEZE.relative_to(ROOT))]=BASE_SHA
    extra=[ROOT/'our_mani/generic_reference_mpc_20261104_v1'/n for n in
           ('__init__.py','train_labels.py','reference_mpc.py','contracts.py','public_observation.py')]
    extra += [ROOT/'our_mani/physmani_feedback_interface_20261104_v1'/n for n in
              ('__init__.py','observation.py','contracts.py','facade.py')]
    for p in [Path(__file__).with_name(n)for n in OWN]+[ROOT/'scripts'/n for n in ENTRIES]+[
            Path(path).resolve(),PARTITION]+extra:
        sources[str(p.relative_to(ROOT))]=sha(p)
    for name,value in sources.items():
        if sha(ROOT/name)!=value:raise ValueError('Capture source/model changed: '+name)
    native=read(BASE_FREEZE)['native_files_sha256']
    for name,value in native.items():
        if sha(name)!=value:raise ValueError('Native original task/common source changed')
    return sources,native


def check_freeze(plan_path,path,expected):
    if sha(path)!=expected:raise ValueError('External final capture freeze SHA required')
    value=read(path);sources,native=source_map(plan_path)
    if value.get('schema')!='Cube_trueclock_motion_capture_execution_freeze_v2' or value.get('files_sha256')!=sources or value.get('native_files_sha256')!=native:
        raise ValueError('Complete explicit source/model/native/partition freeze required')
    return sources,native
