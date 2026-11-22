"""Root-only CPU public replay, then separate private contact compilation.

Default does metadata checks only. --predict-public loads the exact newly
trained GRU on CPU, never private labels. --compile validates the sealed public
predictions before opening teacher labels. Heldout is not a teacher role.
"""
import argparse
from collections import Counter
import hashlib,json,os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
PARTITION=ROOT/'artifacts/framework/local_mpc_20261102/cube_matched_motion_candidate_partition_20261104_v1/partition.json'
PARTITION_SHA='38ac76250d382d36b3c28b01cc8551a25cc95edb850a3c21cc9bf1b47d3ade6a'
PROTOCOL=ROOT/'artifacts/framework/local_mpc_20261102/cube_local_grasp_teacher_protocol_20261104_v3/protocol.json'
PROTOCOL_SHA='1067eb0500813ef3f317091ae7c7a44b76f7618f2f368478251283982d2fcfa4'
CAPTURE_PLAN=ROOT/'config/framework/cube_privileged_contact_teacher_TRAIN12_20261104_v1/plan.json'
FRAME='causal_public_forecast_object_orientation_basis_at_actual_contact'
SCHEMA='Cube_public_reference_local_grasp_teacher_supervision_20261104_v3'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def pointer(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def lines(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s.strip()]
def checked(item):
 p=Path(item['path']).resolve(strict=True)
 if sha(p)!=item['sha256']:raise ValueError('Frozen artifact changed: '+str(p))
 return p
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def write_rows(p,rows):
 with Path(p).open('x')as f:
  for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')


def metadata(capture_plan):
 if sha(PARTITION)!=PARTITION_SHA or sha(PROTOCOL)!=PROTOCOL_SHA:raise ValueError('Prospective split/acceptance protocol changed')
 capture=read(capture_plan);expected=[r for r in read(PARTITION)['rows']if r['role']in('fit','selection')]
 if capture.get('schema')!='Cube_privileged_canonical_P1_contact_teacher_TRAIN12_20261104_v1'or capture.get('partition')!={'path':str(PARTITION),'sha256':PARTITION_SHA}or len(capture.get('parents',[]))!=12:
  raise ValueError('Predeclared teacher12 only; never heldout')
 for a,b in zip(expected,capture['parents']):
  if any(b.get(k)!=v for k,v in a.items())or b.get('episode_id')!=b['parent_id']+'_teacherpass_v1':raise ValueError('Fixed whole parent role/pass required')
 return capture


def source_map():
 names=['scripts/compile_cube_teacher_candidate_dataset_20261104_v3.py']
 names +=['our_mani/cube_candidate_teacher_20261104_v3/'+n for n in('__init__.py','labels.py','data.py','replay.py')]
 names +=['our_mani/cube_candidate_matched_20261104_v2/'+n for n in('__init__.py','data.py','losses.py','evaluation.py')]
 names +=['our_mani/cube_motion_residual_20261104_v1/'+n for n in('__init__.py','model.py','losses.py','data.py','runtime.py')]
 names +=['our_mani/multitask/context_se3.py','our_mani/context_dynamics.py','our_mani/contracts.py','our_mani/local_mpc_p4_20261103/data.py','our_mani/local_mpc_p4_20261103/model.py','our_mani/local_mpc_20261102/records.py']
 names +=[str(PARTITION.relative_to(ROOT)),str(PROTOCOL.relative_to(ROOT))]
 return {name:sha(ROOT/name)for name in names}
def verify_sources(sources):
 for name,h in sources.items():
  if sha(ROOT/name)!=h:raise ValueError('Prepared source changed: '+name)


def teacher_gate(owner_plan,capture_plan,capture):
    """All-role lifecycle/source metadata only; no public/private frame streams."""
    owner_path=Path(owner_plan).resolve(strict=True);owner=read(owner_path);out=Path(owner['output_root'])
    completion=read(out/'completion.json')
    if (completion.get('complete')is not True or completion.get('jobs_completed')!=12
        or completion.get('all_actual_child_waits_completed')is not True or completion.get('source_unchanged')is not True):
        raise ValueError('All12 actual child waits/source must complete before dataset decode')
    request=read(out/'request.json')
    if request.get('plan')!=pointer(owner_path):raise ValueError('Actual owner request must bind this plan')
    expected=[r['parent_id']for r in capture['parents']]
    if [j['id']for j in owner.get('jobs',[])]!=expected:raise ValueError('All12 fixed owner assignments in order required')
    entries={};common=None;freeze_pointer=None;repo_checked=set();native_checked=set()
    for parent,job in zip(capture['parents'],owner['jobs']):
        case=Path(job['output']).resolve(strict=True)
        if 'cube_contact_teacher_TRAIN_20261104_v1'not in case.parts:raise ValueError('Only the fixed privileged teacher v1 capture family is accepted')
        argv=job['argv']
        def arg(name):return argv[argv.index(name)+1]
        if Path(arg('--plan')).resolve()!=Path(capture_plan).resolve()or arg('--parent-id')!=parent['parent_id']:
            raise ValueError('Owner actually executed different capture assignment')
        freeze=Path(arg('--source-freeze'));digest=arg('--source-freeze-sha256')
        if sha(freeze)!=digest:raise ValueError('Captured source freeze changed')
        freeze_json=read(freeze)
        if not freeze_json.get('files_sha256')or not freeze_json.get('native_files_sha256')or freeze_json.get('schema')!='Cube_privileged_contact_teacher_execution_freeze_v1':raise ValueError('Exact teacher source closure required')
        if freeze_pointer is None:freeze_pointer=pointer(freeze);common=freeze_json
        elif common!=freeze_json:raise ValueError('One unchanged complete capture closure required')
        actual_path=out/'owners'/job['id']/'actual_exit.json';actual=read(actual_path)
        worker=read(case/'worker_exit.json')
        if (type(actual.get('actual_process_returncode'))is not int or actual['actual_process_returncode']!=0
            or actual.get('actual_wait_completed')is not True or actual.get('source_unchanged')is not True
            or actual.get('unreaped_descendants')!=[] or actual.get('stop_reason')is not None
            or type(worker.get('actual_worker_returncode'))is not int or worker['actual_worker_returncode']!=0
            or worker.get('native_closed')is not True or worker.get('result_recorded')is not True):
            raise ValueError('All12 typed actual0/wait/Nativeclosed/source required')
        inputs=read(case/'request.json')
        if inputs.get('parent')!=parent or inputs.get('source_sha256')!=common['files_sha256']or inputs.get('native_files_sha256')!=common['native_files_sha256']:
            raise ValueError('Actual captured parent/source identity differs')
        for name,h in common['files_sha256'].items():
            relative=Path(name)
            if relative.is_absolute()or '..'in relative.parts:raise ValueError('Repo-relative capture source required')
            if name not in repo_checked:
                if sha(ROOT/relative)!=h:raise ValueError('Captured live source changed: '+name)
                repo_checked.add(name)
            # Only captured source code/evidence bytes, never heldout frame/label payload.
            if sha(case/'source_snapshot'/relative)!=h:raise ValueError('Captured source archive changed: '+name)
        for name,h in common['native_files_sha256'].items():
            if name not in native_checked:
                if sha(name)!=h:raise ValueError('Original Native source changed')
                native_checked.add(name)
        entries[parent['parent_id']]=dict(case=case,parent=parent,actual_exit=pointer(actual_path),worker_exit=pointer(case/'worker_exit.json'))
    return entries,dict(owner_plan=pointer(owner_path),owner_completion=pointer(out/'completion.json'),capture_plan=pointer(capture_plan),source_freeze=freeze_pointer,
        all12_actual0_wait_nativeclosed_source=True,repo_source_files=len(repo_checked),native_source_files=len(native_checked),
        all_role_public_private_payloads_read=False,heldout_metadata_and_source_archives_only=True)

def predict_public(owner_plan,capture_plan,motion_checkpoint,motion_sha,output):
    import torch
    from our_mani.cube_motion_residual_20261104_v1.runtime import CubeResidualMotionDynamics
    from our_mani.cube_candidate_teacher_20261104_v3.replay import predict_public_episode
    capture=metadata(capture_plan);entries,gate=teacher_gate(owner_plan,capture_plan,capture)
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh public prediction freeze only')
    method=source_map();verify_sources(method)
    torch.set_num_threads(2)
    dynamics=CubeResidualMotionDynamics(motion_checkpoint,expected_sha256=motion_sha)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'request.json',dict(capture_gate=gate,motion_checkpoint=pointer(motion_checkpoint),source_sha256=method,private_label_files_read=0,CPU_threads=2))
    files={};parents=[];inputs=[];calls=0
    try:
        for parent in capture['parents']:
            case=entries[parent['parent_id']]['case'];paths=[case/'public_motion/public_stream.jsonl',case/'public_motion/public_outputs.jsonl']
            before=[pointer(p)for p in paths]
            packets=lines(paths[0]);archived=lines(paths[1])
            rows,report=predict_public_episode(packets,archived,dynamics,episode_id=parent['episode_id'],expected_forecast_sha256=motion_sha)
            name=parent['parent_id']+'.jsonl';write_rows(output/name,rows)
            files[name]=pointer(output/name);parents.append(dict(parent=parent,predictions=files[name],report=report));calls+=report['GRU_forward_batch_calls']
            for p in before:checked(p)
            inputs.extend(before)
        verify_sources(method);dynamics._assert_live()
        freeze=dict(schema='Cube_corrected_GRU_teacher_public_prediction_freeze_20261104_v3',complete=True,capture_gate=gate,
            motion_checkpoint=pointer(motion_checkpoint),source_sha256=method,files=files,parents=parents,public_inputs=inputs,
            GRU_forward_batch_calls=calls,private_label_files_read=0,old_ebec_forecasts_used_as_inputs=False,
            target_frame=FRAME,Native_started=False,GPU_used=False,training_started=False)
        write(output/'public_prediction_freeze.json',freeze)
        return dict(freeze=pointer(output/'public_prediction_freeze.json'),parents=12,actual_GRU_forward_batch_calls=calls,private_label_files_read=0)
    except BaseException as error:
        write(output/'failure.json',dict(complete=False,type=type(error).__name__,message=str(error),completed_parents=len(parents),GRU_forward_batch_calls=calls,private_label_files_read=0,Native_started=False))
        raise


def validate_public_freeze(path,digest):
    path=Path(path).resolve(strict=True)
    if sha(path)!=digest:raise ValueError('External public prediction freeze SHA required')
    freeze=read(path)
    if (freeze.get('schema')!='Cube_corrected_GRU_teacher_public_prediction_freeze_20261104_v3'or freeze.get('complete')is not True
        or freeze.get('private_label_files_read')!=0 or freeze.get('target_frame')!=FRAME or len(freeze.get('parents',[]))!=12):
        raise ValueError('All12 corrected-provider public forecasts must be frozen before labels')
    verify_sources(freeze['source_sha256']);checked(freeze['motion_checkpoint'])
    for p in freeze['files'].values():checked(p)
    for p in freeze['public_inputs']:checked(p)
    return freeze


def compile_contacts(public_freeze,public_freeze_sha,role,output):
    import numpy as np
    from our_mani.cube_candidate_teacher_20261104_v3.data import compile_contact_episode,combine_episode_results
    from our_mani.cube_candidate_teacher_20261104_v3.replay import normalize_private_labels
    if role not in('fit','selection'):raise ValueError('Heldout cannot supply teacher contact supervision')
    freeze=validate_public_freeze(public_freeze,public_freeze_sha)  # before any private labels
    capture_plan=checked(freeze['capture_gate']['capture_plan']);capture=metadata(capture_plan)
    owner_plan=checked(freeze['capture_gate']['owner_plan']);entries,gate=teacher_gate(owner_plan,capture_plan,capture)
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh candidate dataset output only')
    results=[];evidence=[]
    for item in freeze['parents']:
        parent=item['parent']
        if parent['role']!=role:continue
        case=entries[parent['parent_id']]['case'];label_path=case/'private_labels.jsonl';before=pointer(label_path)
        rows=lines(checked(item['predictions']));private=lines(label_path)
        initial,ticks=normalize_private_labels(private,parent_id=parent['parent_id'],episode_id=parent['episode_id'],role=role)
        arrays,report=compile_contact_episode(rows,initial,ticks,parent_id=parent['parent_id'],episode_id=parent['episode_id'],expected_forecast_sha256=freeze['motion_checkpoint']['sha256'])
        results.append((arrays,report));checked(before);evidence.append(before)
    arrays,report=combine_episode_results(results)
    declared_parents=[r['parent_id']for r in capture['parents']if r['role']==role]
    if arrays:
        # Aggregator numbers supervised parents; published report keeps all fixed
        # parents, so remap indices without erasing zero-positive failures.
        supervised=report['supervised_parents']
        arrays['parent_index']=np.asarray([declared_parents.index(supervised[int(i)])for i in arrays['parent_index']],np.int64)
    if not arrays:
        shapes={'future_tokens':(17,16),'robot':(25,),'geometry':(4,),'context_c':(16,),'task_e':(16,),'target_position':(3,),'target_rotation':(3,3),'target_dt':()}
        arrays={k:np.empty((0,*shape),np.float32)for k,shape in shapes.items()}
        arrays.update(target_time_valid=np.empty(0,bool),parent_index=np.empty(0,np.int64),sample_weight=np.empty(0,np.float32))
    output.mkdir(parents=True,exist_ok=False)
    with(output/'dataset.npz').open('xb')as f:np.savez_compressed(f,**arrays)
    classes=Counter()
    for r in report['episodes']:classes.update(r['acceptance_counts'])
    report.update(complete=True,dataset_sha256=sha(output/'dataset.npz'),parents=declared_parents,
        public_prediction_freeze=pointer(public_freeze),forecast_checkpoint_sha256=freeze['motion_checkpoint']['sha256'],
        acceptance_protocol=pointer(PROTOCOL),acceptance_counts=dict(classes),original_Native_goal_required_for_local_label=False,
        full_task_judge_changed=False,score_head_trained=False,unmatched_candidates_are_negative=False,
        label_streams=evidence,capture_gate=gate,source_sha256=source_map(),heldout_payloads_read=False,
        NN_forward_calls=0,Native_started=False,GPU_used=False)
    verify_sources(freeze['source_sha256'])
    for p in evidence:checked(p)
    write(output/'report.json',report)
    write(output/'completion.json',dict(complete=True,stage='candidate_teacher_dataset_compile',role=role,candidate_training_ready=report['candidate_training_ready'],report=pointer(output/'report.json'),dataset=pointer(output/'dataset.npz'),heldout_payloads_read=False,NN_forward_calls=0,Native_started=False))
    return dict(report=pointer(output/'report.json'),dataset=pointer(output/'dataset.npz'),parents=len(report['parents']),anchors=report['supervised_anchors'],acceptance_counts=dict(classes),candidate_training_ready=report['candidate_training_ready'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture-plan',type=Path,default=CAPTURE_PLAN);p.add_argument('--owner-plan',type=Path)
    modes=p.add_mutually_exclusive_group();modes.add_argument('--predict-public',action='store_true');modes.add_argument('--compile',action='store_true')
    p.add_argument('--motion-checkpoint',type=Path);p.add_argument('--expected-motion-sha256');p.add_argument('--public-prediction-freeze',type=Path);p.add_argument('--expected-public-freeze-sha256')
    p.add_argument('--role',choices=('fit','selection'),default='fit');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.predict_public:
        if not all((a.owner_plan,a.motion_checkpoint,a.expected_motion_sha256,a.output)):p.error('Predict needs closed teacher owner, trained new GRU/SHA, fresh output')
        result=predict_public(a.owner_plan,a.capture_plan,a.motion_checkpoint,a.expected_motion_sha256,a.output)
    elif a.compile:
        if not all((a.public_prediction_freeze,a.expected_public_freeze_sha256,a.output)):p.error('Compile needs frozen corrected public predictions/SHA and fresh output')
        result=compile_contacts(a.public_prediction_freeze,a.expected_public_freeze_sha256,a.role,a.output)
    else:
        capture=metadata(a.capture_plan);result=dict(schema=SCHEMA,preparation_only=True,teacher_parents=len(capture['parents']),private_payload_reads=0,model_loads=0,NN_forward_calls=0,Native_started=False,source_sha256=source_map())
    print(json.dumps(result,indent=2,allow_nan=False));return 0
if __name__=='__main__':raise SystemExit(main())
