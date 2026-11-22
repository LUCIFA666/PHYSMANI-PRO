"""Compile captured v2 Cube motion by fixed whole-parent role; no model/Native.

Default checks plan metadata only. Heldout payloads are never read for fit or
selection. Actual public timestamps remain unchanged; labels never fill input.
"""
import argparse
from collections import Counter
import hashlib,json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
PARTITION=ROOT/'artifacts/framework/local_mpc_20261102/cube_matched_motion_candidate_partition_20261104_v1/partition.json'
PARTITION_SHA='38ac76250d382d36b3c28b01cc8551a25cc95edb850a3c21cc9bf1b47d3ade6a'
DEFAULT_CAPTURE_PLAN=ROOT/'config/framework/cube_trueclock_motion_TRAIN16_20261104_v2/plan.json'
COMPILER_DATA=ROOT/'our_mani/cube_motion_residual_20261104_v1/data.py'
COMPILER_DATA_SHA='bb056b68017148c26b22f62e9f1fd0fb011590b143f5457208d873f60d15c792'
OLD_GRU_SHA='ebec3d1b3a83ceb0d4fb3bc889f8b62376cb4581071f53de29e0b6bd71c162ef'
SCHEMA='Cube_trueclock_motion_role_dataset_20261104_v1'
CLOCK_TOL=2e-5


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def pointer(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def lines(path):
    with Path(path).open()as stream:
        for line in stream:
            if line.strip():yield json.loads(line)
def checked(item):
    p=Path(item['path']).resolve(strict=True)
    if sha(p)!=item['sha256']:raise ValueError('Frozen pointer changed: '+str(p))
    return p

def metadata(capture_plan,role):
    if role not in('fit','selection','heldout'):raise ValueError('Explicit fixed role required')
    if sha(PARTITION)!=PARTITION_SHA or sha(COMPILER_DATA)!=COMPILER_DATA_SHA:raise ValueError('Frozen partition/compiler changed')
    partition=read(PARTITION);p=Path(capture_plan).resolve(strict=True);capture=read(p)
    if capture.get('schema')!='Cube_trueclock_keep_open_TRAIN16_capture_plan_20261104_v2':raise ValueError('Only fresh v2 capture, never failed v1 payload')
    if capture.get('partition')!={'path':str(PARTITION),'sha256':PARTITION_SHA}:raise ValueError('Exact pre-capture partition required')
    if len(capture.get('parents',[]))!=16:raise ValueError('All sixteen fixed parents required')
    for declared,actual in zip(partition['rows'],capture['parents']):
        if any(actual.get(k)!=v for k,v in declared.items()):raise ValueError('Whole-parent assignment differs')
        if actual.get('episode_id')!=actual['parent_id']+'_motionpass_v2':raise ValueError('Fresh v2 physical pass required')
    selected=[r for r in capture['parents']if r['role']==role]
    return capture,selected


def selected_gate(path,digest):
    if path is None or digest is None:raise ValueError('Heldout needs external frozen selected training completion and SHA')
    p=Path(path).resolve(strict=True)
    if sha(p)!=digest:raise ValueError('Heldout selected completion SHA differs')
    status=read(p)
    if (status.get('complete')is not True or status.get('training_complete')is not True
        or status.get('source_unchanged')is not True or status.get('heldout_payloads_read')is not False):
        raise ValueError('Actual selected training must complete without heldout use')
    cp=checked(status['selected_checkpoint'])
    return dict(training_completion=pointer(p),selected_checkpoint=pointer(cp))


def capture_gate(owner_plan,capture_plan,capture):
    """All-role lifecycle/source metadata only; no public/private frame streams."""
    owner_path=Path(owner_plan).resolve(strict=True);owner=read(owner_path);out=Path(owner['output_root'])
    completion=read(out/'completion.json')
    if (completion.get('complete')is not True or completion.get('jobs_completed')!=16
        or completion.get('all_actual_child_waits_completed')is not True or completion.get('source_unchanged')is not True):
        raise ValueError('All16 actual child waits/source must complete before dataset decode')
    request=read(out/'request.json')
    if request.get('plan')!=pointer(owner_path):raise ValueError('Actual owner request must bind this plan')
    expected=[r['parent_id']for r in capture['parents']]
    if [j['id']for j in owner.get('jobs',[])]!=expected:raise ValueError('All16 fixed owner assignments in order required')
    entries={};common=None;freeze_pointer=None;repo_checked=set();native_checked=set()
    for parent,job in zip(capture['parents'],owner['jobs']):
        case=Path(job['output']).resolve(strict=True)
        if 'cube_motion_TRAIN_20261104_v2'not in case.parts:raise ValueError('No v1 or other-family payload')
        argv=job['argv']
        def arg(name):return argv[argv.index(name)+1]
        if Path(arg('--plan')).resolve()!=Path(capture_plan).resolve()or arg('--parent-id')!=parent['parent_id']:
            raise ValueError('Owner actually executed different capture assignment')
        freeze=Path(arg('--source-freeze'));digest=arg('--source-freeze-sha256')
        if sha(freeze)!=digest:raise ValueError('Captured source freeze changed')
        freeze_json=read(freeze)
        if not freeze_json.get('files_sha256')or not freeze_json.get('native_files_sha256')or freeze_json.get('schema')!='Cube_trueclock_motion_capture_execution_freeze_v2':raise ValueError('Exact v2 source closure required')
        if freeze_pointer is None:freeze_pointer=pointer(freeze);common=freeze_json
        elif common!=freeze_json:raise ValueError('One unchanged complete capture closure required')
        actual_path=out/'owners'/job['id']/'actual_exit.json';actual=read(actual_path)
        worker=read(case/'worker_exit.json')
        if (type(actual.get('actual_process_returncode'))is not int or actual['actual_process_returncode']!=0
            or actual.get('actual_wait_completed')is not True or actual.get('source_unchanged')is not True
            or actual.get('unreaped_descendants')!=[] or actual.get('stop_reason')is not None
            or type(worker.get('actual_worker_returncode'))is not int or worker['actual_worker_returncode']!=0
            or worker.get('native_closed')is not True or worker.get('result_recorded')is not True):
            raise ValueError('All16 typed actual0/wait/Nativeclosed/source required')
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
        all16_actual0_wait_nativeclosed_source=True,repo_source_files=len(repo_checked),native_source_files=len(native_checked),
        all_role_public_private_payloads_read=False,heldout_metadata_and_source_archives_only=True)


def project_episode(public,private,parent):
    """Explicit field projection. Public never obtains a private position."""
    import math
    if len(public)!=len(private)or not public:raise ValueError('One public and committed private row per real tick required')
    pid=parent['parent_id'];episode=parent['episode_id'];p_rows=[];labels=[];segment=0;was_free=False
    private_by_tick={}
    for i,row in enumerate(private):
        if row.get('parent_id')!=pid or row.get('episode_id')!=episode or row.get('role')!=parent['role']or row.get('reset_boundary_id')!=episode:
            raise ValueError('No cross-role/reset/episode private joins')
        tick=row['physics_tick']
        if type(tick)is not int or tick!=i:raise ValueError('All actual private physics rows required')
        private_by_tick[tick]=row
    for i,row in enumerate(public):
        if row.get('schema')!='Cube_trueclock_motion_public_frame_v1'or row.get('parent_id')!=pid or row.get('episode_id')!=episode or row.get('physical_tick')!=i or row.get('index')!=i:
            raise ValueError('All original v2 public rows required, including UNKNOWN')
        if row.get('Native_labels_in_input')is not False:raise ValueError('Public input channel is not label-free')
        stamp=row['timestamp']
        if isinstance(stamp,bool)or not isinstance(stamp,(float,int))or not math.isfinite(stamp):raise ValueError('Actual public timestamp required')
        target=row['target'];available=target.get('available')
        if type(available)is not bool:raise ValueError('Explicit current measurement availability required')
        if not available and(target.get('position_world')is not None or target.get('confidence')is not None):
            raise ValueError('UNKNOWN must retain null current position/confidence')
        p_rows.append(dict(timestamp=stamp,available=available,position_world=target['position_world']if available else None,confidence=target['confidence']if available else None))
        label=private_by_tick[i];actual=label['actual_relative_physics_timestamp']
        if (abs(stamp-actual)>1e-9 or abs(stamp-(row['physics_timestamp']-row['physics_time_origin']))>1e-9
            or abs(actual-(label['physics_time_absolute']-label['physics_time_origin']))>1e-9):
            raise ValueError('Same actual physics timestamp join only, no rounding/interpolation')
        if abs((row['task_timestamp']-row['task_time_origin'])-(label['task_time_absolute']-label['task_time_origin']))>CLOCK_TOL:
            raise ValueError('Private/public task clocks differ')
        if type(label.get('held'))is not bool or isinstance(label.get('requested_opening'),bool):raise ValueError('Measured free/closed labels required')
        free=not label['held'] and float(label['requested_opening'])>=.5 and label.get('close_requested')is False
        if free and not was_free:segment+=1
        labels.append(dict(timestamp=actual,target_position_world=label['target_position_world'],free_motion=free,segment_id=episode+':free'+str(segment)))
        was_free=free
    return p_rows,labels


def old_forecasts(path,episode_public):
    """Existing saved predictions, no model or future labels used here."""
    import numpy as np
    if not path.is_file():return {},dict(available=False,reason='no_saved_public_outputs')
    bytime={};reasons=Counter();current_times={float(r['timestamp'])for r in episode_public}
    for row in lines(path):
        result=row['result'];at=float(result['timestamp'])
        if at not in current_times:raise ValueError('Saved forecast must match an exact captured public timestamp')
        forecast=result.get('forecast')
        if result.get('mode')!='FREE'or result.get('free_forecast_available')is not True or not forecast:
            reasons['no_available_FREE_forecast']+=1;continue
        p=np.asarray(forecast['positions'],np.float32);t=np.asarray(forecast['times'],np.float64)
        if (p.shape!=(17,1,3)or t.shape!=(17,)or not np.isfinite(p).all()or not np.isfinite(t).all()
            or not np.allclose(t,at+np.arange(17)*.05,atol=1e-7,rtol=0)):
            raise ValueError('Saved old forecast shape/world/query times invalid')
        if forecast['metadata'].get('checkpoint_sha256')!=OLD_GRU_SHA:
            reasons['not_original_frozen_GRU']+=1;continue
        if at in bytime:raise ValueError('Duplicate saved forecast anchor')
        bytime[at]=(p[:,0],t)
    return bytime,dict(available=bool(bytime),anchors=len(bytime),reasons=dict(reasons),checkpoint_sha256=OLD_GRU_SHA,input_to_new_model=False)


def compile_dataset(owner_plan,capture_plan,role,output,*,selected_training_completion=None,selected_training_completion_sha256=None):
    import numpy as np
    capture,parents=metadata(capture_plan,role)
    holdout=selected_gate(selected_training_completion,selected_training_completion_sha256)if role=='heldout'else None
    if role!='heldout'and(selected_training_completion is not None or selected_training_completion_sha256 is not None):raise ValueError('Fit/selection cannot masquerade as heldout scoring')
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh dataset output only')
    entries,gate=capture_gate(owner_plan,capture_plan,capture)
    from our_mani.cube_motion_residual_20261104_v1.data import compile_episode
    script_before=sha(__file__)
    windows=[];reports=[];old=[];oldtimes=[];oldvalid=[];parent_indices=[];evidence=[gate[k]for k in('owner_plan','owner_completion','capture_plan','source_freeze')]
    if holdout:evidence.extend(holdout.values())
    parent_ids=[p['parent_id']for p in parents]
    for index,parent in enumerate(parents):
        case=entries[parent['parent_id']]['case'];result=read(case/'result.json')
        if result.get('complete')is not True or result.get('source_unchanged')is not True or result.get('episode_id')!=parent['episode_id']:
            raise ValueError('Completed actual role episode required')
        paths=[case/'public_motion/public_stream.jsonl',case/'private_labels.jsonl']
        before={str(p):sha(p)for p in paths};public=list(lines(paths[0]));private=list(lines(paths[1]))
        projected,labels=project_episode(public,private,parent)
        if len(public)!=result['public_rows']or len(private)!=result['private_label_rows']or len(public)!=result['actual_physics_ticks']+1:
            raise ValueError('Every real public/private tick must remain in denominator')
        current,report=compile_episode(projected,labels,parent_id=parent['parent_id'])
        old_path=case/'public_motion/public_outputs.jsonl'
        if old_path.is_file():before[str(old_path)]=sha(old_path)
        saved,baseline=old_forecasts(old_path,public)
        for window in current:
            anchor=window['anchor_time'];match=saved.get(anchor)
            old.append(np.full((17,3),np.nan,np.float32)if match is None else match[0]);oldtimes.append(np.full(17,np.nan,np.float64)if match is None else match[1])
            oldvalid.append(np.full(17,match is not None,dtype=bool));parent_indices.append(index)
        windows.extend(current)
        report.update(task=parent['task'],speed_group=parent['speed_group'],role=role,episode_id=parent['episode_id'],baseline=baseline,Native_role_only=True,
            lifecycle={k:v for k,v in entries[parent['parent_id']].items()if k not in('parent','case')})
        reports.append(report)
        for name,h in before.items():
            if sha(name)!=h:raise ValueError('Role stream changed during compile')
            evidence.append(dict(path=name,sha256=h))
    shapes={'positions':((20,3),np.float32),'timestamps':((20,),np.float64),'confidence':((20,),np.float32),'lengths':((),np.int64),
            'target_positions':((17,3),np.float32),'target_valid':((17,),bool),'target_times':((17,),np.float64)}
    arrays={}
    for key,(shape,dtype)in shapes.items():
        which='inputs'if key in('positions','timestamps','confidence','lengths')else'targets'
        arrays[key]=np.stack([w[which][key]for w in windows])if windows else np.empty((0,*shape),dtype=dtype)
    ids=np.asarray(parent_indices,np.int64);counts=Counter(ids.tolist())
    arrays.update(parent_index=ids,sample_weight=np.asarray([1./counts[i]for i in ids],np.float32),
        recorded_old_positions=np.stack(old)if old else np.empty((0,17,3),np.float32),
        recorded_old_times=np.stack(oldtimes)if oldtimes else np.empty((0,17),np.float64),
        recorded_old_valid=np.stack(oldvalid)if oldvalid else np.empty((0,17),bool))
    for p in evidence:
        if sha(p['path'])!=p['sha256']:raise ValueError('Input changed before publishing dataset')
    if sha(__file__)!=script_before or sha(COMPILER_DATA)!=COMPILER_DATA_SHA:
        raise ValueError('Compiler sources changed during conversion')
    output.mkdir(parents=True,exist_ok=False)
    with(output/'dataset.npz').open('xb')as stream:np.savez_compressed(stream,**arrays)
    report=dict(schema=SCHEMA,complete=True,role=role,parents=parent_ids,parent_count=len(parent_ids),windows=len(windows),training_data_ready=bool(windows),
        zero_window_parents=[r['parent_id']for r in reports if r['windows']==0],per_parent=reports,
        dataset_sha256=sha(output/'dataset.npz'),shapes={k:list(v.shape)for k,v in arrays.items()},
        partition=pointer(PARTITION),capture_gate=gate,selected_training_gate=holdout,streams=evidence,
        compiler=pointer(__file__),compiler_data=pointer(COMPILER_DATA),time_basis='actual_physics_and_task_50ms',
        maximum_actual_label_vs_nominal_query_alignment_seconds=max((r['maximum_actual_label_vs_nominal_query_alignment_seconds']for r in reports),default=0.),
        targets_are_inputs=False,unknown_current_filled_from_GT=False,invalid_targets_may_be_NaN=True,
        heldout_payloads_read=role=='heldout',heldout_role_parent_count=4,official_TEST_read=False,Native_started=False,NN_forward_calls=0,GPU=False,
        metadata_scope='Lifecycle/request/source metadata audited for all16. Only selected role public/private frame payloads decoded. result JSON for selected role may contain terminal private metric keys; those fields are not consumed as inputs or window selection.')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'completion.json').write_text(json.dumps(dict(complete=True,method='current_network',stage='motion_dataset_compile',role=role,
        report=pointer(output/'report.json'),dataset=pointer(output/'dataset.npz'),source_unchanged=True,NN_forward_calls=0,Native_started=False,heldout_payloads_read=role=='heldout'),indent=2)+'\n')
    return dict(report=pointer(output/'report.json'),dataset=pointer(output/'dataset.npz'),parents=parent_ids,windows=len(windows),zero_window_parents=report['zero_window_parents'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--owner-plan',type=Path);p.add_argument('--capture-plan',type=Path,default=DEFAULT_CAPTURE_PLAN)
    p.add_argument('--role',choices=('fit','selection','heldout'),default='fit');p.add_argument('--output',type=Path);p.add_argument('--compile',action='store_true')
    p.add_argument('--selected-training-completion',type=Path);p.add_argument('--selected-training-completion-sha256');a=p.parse_args()
    if not a.compile:
        capture,parents=metadata(a.capture_plan,a.role)
        print(json.dumps(dict(schema=SCHEMA,role=a.role,parents=[r['parent_id']for r in parents],metadata_only=True,all_payload_reads=0,
            capture_preparation_only=capture['preparation_only'],Native_started=False,NN_forward_calls=0)));return 0
    if a.owner_plan is None or a.output is None:p.error('--compile requires owner plan and fresh output')
    print(json.dumps(compile_dataset(a.owner_plan,a.capture_plan,a.role,a.output,selected_training_completion=a.selected_training_completion,
        selected_training_completion_sha256=a.selected_training_completion_sha256),indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
