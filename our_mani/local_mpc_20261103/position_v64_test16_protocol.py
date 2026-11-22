"""Metadata-only reserved TEST16 protocol; materialization needs later Root auth.

No simulator, model inference, TEST trajectory or label is opened here.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'artifacts/framework/local_mpc_20261102/V64_reserved_TEST16_evaluation_only_protocol_v1/protocol.json'
PROTOCOL_SHA256='17ed236d7da53c7b04a552e237e452d0415fd0a7adfd0d3f5ede6bc9a9962136'
PREPARATION_PLAN=ROOT/'config/framework/P4_frozen_V64_TEST16_evaluation_preparation_20261103_v1.json'
PREPARATION_PLAN_SHA256='36438e75a13aae8d153bbbc513f242d67a6021b1093e3c030fdd3589461269b6'
CONFIG_DIRECTORY=ROOT/'config/framework/P4_frozen_V64_newTEST16_eval_v1'
PENDING_RECEIPT=CONFIG_DIRECTORY/'materialization_receipt.pending_freeze.json'
FINAL_RECEIPT=CONFIG_DIRECTORY/'materialization_receipt.json'
EXECUTION_PLAN=CONFIG_DIRECTORY/'execution_plan.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def pointer(path):return dict(path=str(Path(path).resolve().relative_to(ROOT)),sha256=sha(path))
def checked(ref):
    if not isinstance(ref,dict)or set(ref)!={'path','sha256'}:raise ValueError('Exact metadata pointer required')
    p=(ROOT/ref['path']).resolve(strict=True);p.relative_to(ROOT)
    if sha(p)!=ref['sha256']:raise ValueError('Metadata/source SHA changed: '+str(p))
    return p
def write_new(path,value):
    with Path(path).open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def metadata():
    if sha(PROTOCOL)!=PROTOCOL_SHA256 or sha(PREPARATION_PLAN)!=PREPARATION_PLAN_SHA256:
        raise ValueError('Predeclared evaluation-only protocol/assignment plan changed')
    protocol=load(PROTOCOL);plan=load(PREPARATION_PLAN);partition=load(checked(protocol['partition']))
    jobs=plan['jobs']
    if (plan['partition']!=protocol['partition']or plan['protocol']!=pointer(PROTOCOL)
        or len(jobs)!=16 or [j['seed']for j in jobs]!=list(range(261103500,261103516))
        or [j['assignment']for j in jobs]!=partition['jobs']
        or [j['scene']for j in jobs].count('static')!=2
        or [j['scene']for j in jobs].count('uniform')!=8
        or [j['scene']for j in jobs].count('speed_change')!=6
        or plan.get('public_hold_canary_gate',{}).get('passed')is not False
        or plan.get('evaluation_only')is not True):
        raise ValueError('Fixed16 original assignments and retained failed public gate required')
    return protocol,plan


def check_preparation():
    protocol,plan=metadata();freeze=load(checked(protocol['validation_source_freeze']))
    for path,digest in freeze['files_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Frozen V64/base-model source changed: '+path)
    for key in ('base_config','position_checkpoint','candidate_checkpoint'):checked(protocol[key])
    return dict(schema='V64_reserved_TEST16_metadata_preparation_check_v1',complete=True,
        original_partition=protocol['partition'],seeds=[j['seed']for j in plan['jobs']],
        motion_counts=dict(static=2,uniform=8,speed_change=6),V64_source234_current=True,
        materialized=CONFIG_DIRECTORY.exists(),Native_started=False,model_forward_calls=0,
        TEST_data_or_labels_read=False,public_hold_canary_gate_passed=False,
        execution_ready=False,requires_later_Root_authorization_after_complete_devVAL8=True)


def expected_config(job):
    protocol,plan=metadata()
    if job not in plan['jobs']:raise ValueError('Only one fixed reserved TEST assignment allowed')
    base=load(checked(protocol['base_config']));v=deepcopy(base);a=job['assignment']
    v.pop('v64_development_reference',None)
    v.update(schema='P4_frozen_V64_reserved_TEST16_evaluation_only_v1',
        information_setting='frozen_V64_reserved_TEST16_evaluation_only',
        single_change='Frozen V64/a3 applied to preassigned unseen parents; public-hold qualification still FAIL',
        seed=a['seed'],initial_target_position=[*a['initial_xy'],base['initial_target_position'][2]],
        uniform_velocity=a['initial_velocity_m_s'],speed_change_at_tick=a['speed_change_at_tick'],
        changed_velocity=a['changed_velocity_m_s'],dataset_use='test',data_split='test',evaluation_split='test',
        paired_reference=None,v64_preparation_only=False,
        v64_test16_assignment=dict(id=a['id'],parent_id=a['parent_id'],seed=a['seed'],
            canonical_motion_sha256=a['canonical_motion_sha256'],partition=protocol['partition'],protocol=pointer(PROTOCOL)))
    info=deepcopy(v['P4_collection_metadata']);info.update(parent_id=a['parent_id'],episode_id=a['id'],split='test',
        variant_id='frozen_V64_position_a3_reserved_TEST16_evaluation_only',new_development_VAL_only=False,
        frozen_evaluation_only=True,official_VAL_or_TEST_source=False)
    v['P4_collection_metadata']=info
    return v


def validate_authorization(path,expected_sha256):
    path=checked(dict(path=str(Path(path).resolve().relative_to(ROOT)),sha256=expected_sha256))
    auth=load(path);protocol,_=metadata()
    if (auth.get('schema')!='V64_TEST16_evaluation_only_materialization_authorization_v1'
        or auth.get('authorize_materialization')is not True
        or auth.get('protocol')!=pointer(PROTOCOL)or auth.get('partition')!=protocol['partition']
        or auth.get('public_hold_gate_passed')is not False
        or auth.get('evaluation_only')is not True or auth.get('TEST_outputs_read')is not False
        or auth.get('validation_source_freeze')!=protocol['validation_source_freeze']):
        raise ValueError('Later explicit Root evaluation-only authorization required; no quality-pass substitution')
    checked(auth['validation_report'])
    validation=load(checked(protocol['validation_plan']));sources=load(checked(protocol['validation_source_freeze']))['files_sha256']
    # Require the entire cohort to have actual exits before opening any case metadata.
    for job in validation['jobs']:
        output=ROOT/job['path'];prefix=output.parent/(output.name+'.owner')
        if not Path(str(prefix)+'.actual_exit.json').is_file():raise ValueError('Complete devVAL8 actual exits required')
    for job in validation['jobs']:
        output=ROOT/job['path'];prefix=output.parent/(output.name+'.owner')
        owner=load(Path(str(prefix)+'.actual_exit.json'));worker=load(output/'worker_exit.json')
        result=load(output/'result.json');run=load(output/'run_inputs.json')
        if (owner.get('actual_process_returncode')!=0 or owner.get('actual_wait_completed')is not True
            or owner.get('source_unchanged')is not True or worker.get('native_closed')is not True
            or worker.get('exit_code')!=0 or result.get('complete')is not True
            or result.get('source_unchanged')is not True or run.get('source_sha256')!=sources
            or run.get('config')!=load(ROOT/job['config'])):
            raise ValueError('Full devVAL8 clean closure/source identity required, independent of task outcome')
    return path,auth


def materialize(authorization,authorization_sha256):
    check_preparation();path,auth=validate_authorization(authorization,authorization_sha256)
    protocol,plan=metadata()
    if CONFIG_DIRECTORY.exists():raise FileExistsError('Never overwrite/resume reserved TEST configs')
    CONFIG_DIRECTORY.mkdir(parents=True,exist_ok=False)
    rows=[]
    for job in plan['jobs']:
        dest=ROOT/job['config'];write_new(dest,expected_config(job))
        rows.append({k:job[k]for k in ('id','parent_id','seed','scene','canonical_motion_sha256')}|dict(config=pointer(dest)))
    receipt=dict(schema='V64_reserved_TEST16_pending_materialization_v1',complete=True,evaluation_only=True,
        public_hold_gate_passed=False,partition=protocol['partition'],protocol=pointer(PROTOCOL),
        authorization=pointer(path),configs=rows,case_configs_generated=True,TEST_data_generated=False,
        Native_started=False,all16_regardless_of_task_outcomes=True)
    write_new(PENDING_RECEIPT,receipt)
    return dict(complete=True,Native_started=False,TEST_data_generated=False,pending_materialization=pointer(PENDING_RECEIPT))


def assert_materialized_config(config):
    _,plan=metadata();item=config.get('v64_test16_assignment',{})
    job=next((j for j in plan['jobs']if j['id']==item.get('id')),None)
    if job is None or config!=expected_config(job):raise ValueError('Exact immutable TEST recipe/config required')
    pending=load(PENDING_RECEIPT)
    if (pending.get('schema')!='V64_reserved_TEST16_pending_materialization_v1'
        or pending.get('complete')is not True or pending.get('public_hold_gate_passed')is not False
        or pending.get('partition')!=plan['partition']or pending.get('protocol')!=pointer(PROTOCOL)
        or pending.get('case_configs_generated')is not True or pending.get('TEST_data_generated')is not False):
        raise ValueError('Genuine later-authorized materialization receipt required')
    validate_authorization(checked(pending['authorization']),pending['authorization']['sha256'])
    if len(pending['configs'])!=16:raise ValueError('All16 materialized configs required')
    for row,j in zip(pending['configs'],plan['jobs']):
        if (any(row[k]!=j[k]for k in ('id','parent_id','seed','scene','canonical_motion_sha256'))
            or checked(row['config'])!=(ROOT/j['config']).resolve(strict=True)):
            raise ValueError('Materialized cohort identity changed')
    return job,checked(plan['partition'])


def metadata_source_paths():
    protocol,_=metadata();pending=load(PENDING_RECEIPT)
    auth=load(checked(pending['authorization']))
    return [PROTOCOL,PREPARATION_PLAN,checked(protocol['partition']),checked(protocol['base_config']),
            checked(protocol['validation_plan']),checked(protocol['validation_source_freeze']),
            checked(auth['validation_report']),PENDING_RECEIPT,checked(pending['authorization'])]


def bind_source_freeze(source_freeze,expected_sha256):
    source=checked(dict(path=str(Path(source_freeze).resolve().relative_to(ROOT)),sha256=expected_sha256))
    freeze=load(source);protocol,plan=metadata();pending=load(PENDING_RECEIPT)
    if (freeze.get('schema')!='V64_TEST16_execution_source_freeze_v1'
        or freeze.get('position_measurement_qualified')is not True
        or freeze.get('checkpoint')!=protocol['position_checkpoint']):raise ValueError('Exact TEST16 sourcefreeze required')
    for path,digest in freeze['files_sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Prospective TEST16 source changed')
    for job in plan['jobs']:assert_materialized_config(load(ROOT/job['config']))
    from .position_v64_test16_worker import check_config
    first=plan['jobs'][0]
    admission=check_config(ROOT/first['config'],first['scene'],source_freeze=source,
                           expected_source_freeze_sha256=expected_sha256,require_ready=True)
    if admission.get('ready')is not True:raise ValueError('Full exact TEST16 execution source closure required')
    receipt=dict(pending,schema='V64_reserved_TEST16_evaluation_materialization_receipt_v1',
                 pending_materialization=pointer(PENDING_RECEIPT),source_freeze=pointer(source))
    write_new(FINAL_RECEIPT,receipt)
    jobs=[{k:v for k,v in j.items()if k!='assignment'}for j in plan['jobs']]
    execution=dict(schema='V64_reserved_TEST16_evaluation_only_execution_plan_v1',partition=protocol['partition'],
        source_freeze=pointer(source),protocol=pointer(PROTOCOL),materialization_receipt=pointer(FINAL_RECEIPT),
        jobs=jobs,evaluation_only=True,public_hold_canary_gate=protocol['public_hold_canary_gate'],
        raw_model_qualified=False,formal_pose_real_time_force_qualified=False,all16_regardless_of_task_outcomes=True,
        no_task_retries_or_case_replacement=True,Native_started=False,TEST_data_or_labels_read=False)
    write_new(EXECUTION_PLAN,execution)
    return dict(complete=True,execution_plan=pointer(EXECUTION_PLAN),Native_started=False,TEST_data_generated=False)
