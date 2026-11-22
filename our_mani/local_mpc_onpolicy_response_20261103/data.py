"""Actual public robot/reference transitions, from preassigned TRAIN only.

Every 50ms reference is read from its dispatch receipt, never reconstructed as
a training label. Public robot samples supply measured joint/TCP labels. No
object, Native held/success, action advice, or task outcome field is accessed.
"""
from copy import deepcopy
import hashlib,json
from pathlib import Path

import numpy as np

from ..local_mpc_20261102.executor import JointLimits
from ..local_mpc_p2_20261103.reference import reference_step

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='public_onpolicy_joint_response_TRAIN_20261103_v1'
DT=.05
PROBE='ACTIVE_VISUAL_VERIFICATION_PROBE'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(v) for v in Path(path).read_text().splitlines() if v.strip()]
def vector(value,name,size=7):
    value=np.asarray(value,dtype=np.float64)
    if value.shape!=(size,) or not np.isfinite(value).all():raise ValueError('finite actual '+name+' required')
    return value.copy()
def tick_id(value):
    if isinstance(value,(bool,np.bool_)) or not np.isfinite(value) or value<0:raise ValueError('actual timestamp required')
    tick=int(round(value/DT))
    if abs(value-tick*DT)>1e-8:raise ValueError('actual50ms timestamp required; no interpolation')
    return tick
def equal(a,b,name):
    if not np.allclose(a,b,rtol=0,atol=1e-8):raise ValueError('actual '+name+' disagreement, no silent correction')


def canonical_h(value,q,dq,now):
    if not isinstance(value,dict) or type(value.get('reference_valid')) is not bool:
        raise ValueError('actual executor h with explicit reference_valid required')
    valid=value['reference_valid'];initialized=value.get('reference_positions') is not None
    clock=value.get('reference_clock')
    if (valid or initialized) and (clock is None or abs(float(clock)-now)>1e-8):
        raise ValueError('fresh actual executor reference_clock required')
    if value.get('reference_is_measured_state') is not False:
        raise ValueError('reference state must not be mislabeled as measured robot state')
    if initialized:
        vals={short:vector(value.get(long),long) for short,long in (
            ('q_ref','reference_positions'),('dq_ref','reference_velocities'),
            ('ddq_ref','reference_accelerations'),('goal_q','target_joints'))}
    else:
        if valid:raise ValueError('valid reference without initialized arrays')
        vals=dict(q_ref=q.copy(),dq_ref=dq.copy(),ddq_ref=np.zeros(7),goal_q=q.copy())
    grip=value.get('gripper_target')
    if isinstance(grip,(bool,np.bool_)) or not isinstance(grip,(int,float,np.number)) or not 0<=grip<=1:
        raise ValueError('known actual gripper target required in h')
    return dict(**vals,valid=valid,initialized=initialized,gripper_target=float(grip))


def limits_from_receipt(value):
    return JointLimits(value['native_joint_position_lower'],value['native_joint_position_upper'],
        value['native_joint_velocity_limits'],value['controller_reference_acceleration_limits'],
        value['controller_reference_jerk_limits'],value['controller_reference_velocity_cap'],
        provenance=value.get('provenance',{}))


def extract_transitions(controls,public_frames):
    """Pure aligned extraction. Unknown/failure task phases remain included."""
    if not controls or not public_frames:raise ValueError('complete nonempty actual records required')
    robots={}
    for frame in public_frames:
        robot=frame['public_robot'];index=tick_id(robot['timestamp'])
        if index in robots:raise ValueError('duplicate actual public robot frame')
        robots[index]=robot
    if sorted(robots)!=list(range(len(robots))):raise ValueError('public robot frame gap; do not fabricate')
    q=[];dq=[];tcp=[];times=[];refs=[];goals=[];phase=[];anchors=[];openings=[]
    last_tick=0;previous_ref=None;last_limits=None
    for macro in controls:
        receipt=macro['executor'];ticks=receipt['ticks'];now=float(macro['time'])
        if tick_id(now)!=last_tick:raise ValueError('macro time must continue actual tick ledger')
        if not 1<=len(ticks)<=2 or receipt['executed_ticks']!=len(ticks):raise ValueError('executed tick count mismatch')
        if receipt.get('unexpected_physics_ticks',0)!=0:raise ValueError('hidden physics cannot be a training transition')
        target=vector(receipt['target_joints'],'known joint goal')
        for offset,tick in enumerate(ticks):
            if tick.get('tick_completed') is not True:raise ValueError('only actually completed tick labels allowed')
            before=tick_id(tick['clock_before']);after=tick_id(tick['clock_after'])
            if before!=last_tick or after!=before+1:raise ValueError('actual tick gap/overlap')
            qb=vector(tick['measured_positions_before'],'q_before');db=vector(tick['measured_velocities_before'],'dq_before')
            qa=vector(tick['measured_positions_after'],'q_after');da=vector(tick['measured_velocities_after'],'dq_after')
            for index,measured,velocity in ((before,qb,db),(after,qa,da)):
                robot=robots[index]
                equal(vector(robot['q'],'public q'),measured,'public/dispatch joint q')
                equal(vector(robot['dq'],'public dq'),velocity,'public/dispatch joint dq')
            if offset==0:
                equal(vector(macro['public_robot']['q'],'macro public q'),qb,'macro q')
                equal(vector(macro['public_robot']['dq'],'macro public dq'),db,'macro dq')
                anchor=canonical_h(macro['actuator_state_before'],qb,db,now)
                if previous_ref is not None:
                    for key,value in previous_ref.items():
                        if key in ('q_ref','dq_ref','ddq_ref','goal_q'):equal(anchor[key],value,'macro h '+key)
            else:anchor=deepcopy(previous_ref)
            if q:
                equal(q[-1],qb,'adjacent tick q');equal(dq[-1],db,'adjacent tick dq')
            else:
                q.append(qb);dq.append(db);times.append(before*DT)
                tcp.append(vector(robots[before]['tcp_position'],'public TCP',3))
            reference=np.r_[vector(tick['reference_positions'],'dispatched qref'),
                vector(tick['reference_velocities'],'dispatched vref'),vector(tick['reference_accelerations'],'dispatched aref')]
            refs.append(reference);goals.append(target.copy());anchors.append(anchor)
            phase.append(macro['decision']['mode']);openings.append(float(tick['gripper_target']))
            q.append(qa);dq.append(da);times.append(after*DT)
            tcp.append(vector(robots[after]['tcp_position'],'public TCP',3))
            previous_ref=dict(q_ref=reference[:7].copy(),dq_ref=reference[7:14].copy(),ddq_ref=reference[14:].copy(),
                goal_q=target.copy(),valid=True,initialized=True,gripper_target=float(tick['gripper_target']))
            last_tick=after
        current_limits=limits_from_receipt(receipt['limits'])
        if last_limits is not None and current_limits.receipt()!=last_limits.receipt():raise ValueError('controller limits changed within episode')
        last_limits=current_limits
    if last_tick!=len(robots)-1:raise ValueError('every actual public robot tick must join dispatch exactly')
    h={key:np.asarray([value[key] for value in anchors]) for key in anchors[0]}
    result=dict(q=np.asarray(q),dq=np.asarray(dq),tcp=np.asarray(tcp),times=np.asarray(times),
        refs=np.asarray(refs),goals=np.asarray(goals),phase=np.asarray(phase),
        probe_mask=np.asarray([value==PROBE for value in phase],bool),h=h,
        command_opening=np.asarray(openings),limits=last_limits)
    result['audit']=dict(actual_ticks=len(refs),real_public_robot_frames=len(robots),
        synthetic_or_interpolated_frames=0,reconstructed_reference_labels=False,
        Native_object_held_success_accessed=False,unknown_or_failed_task_phases_filtered=False,
        actual_dq_vs_position_FD_max_rad_s=float(np.max(np.abs(result['dq'][1:]-np.diff(result['q'],axis=0)/DT))))
    return result


def oracle_reference_audit(episode):
    """Oracle current q/dq/h reconstruction audit; never a free rollout score."""
    result=reference_step(episode['q'][:-1],episode['dq'][:-1],episode['goals'],
        deepcopy(episode['h']),episode['limits'],dt=DT)
    if not result['feasible'].all() or not result['applied'].all():raise ValueError('actual oracle reference unexpectedly infeasible')
    refs=np.c_[result['q_ref'],result['dq_ref'],result['ddq_ref']]
    error=np.abs(refs-episode['refs'])
    report=dict(maximum_absolute=float(error.max()),qref_max_rad=float(error[:,:7].max()),
        vref_max_rad_s=float(error[:,7:14].max()),aref_max_rad_s2=float(error[:,14:].max()),
        actual_state_oracle_only_not_free_model_score=True)
    if report['maximum_absolute']>1e-9:raise ValueError('dispatched reference reconstruction mismatch: '+str(report))
    return report


def preflight_plan(plan_path,protocol_path,root=ROOT):
    root=Path(root);plan_path=Path(plan_path);protocol_path=Path(protocol_path)
    plan=read(plan_path);protocol=read(protocol_path)
    if (plan.get('split')!='train' or plan.get('all_parents_assigned_before_generation') is not True
            or protocol.get('plan_sha256')!=sha(plan_path) or protocol.get('assigned_before_collection') is not True
            or protocol.get('no_development_VAL_or_TEST_for_fitting') is not True
            or protocol.get('ridge')!=1e-3 or protocol.get('native_dt')!=DT):
        raise ValueError('preassigned TRAIN-only plan and immutable fixed fit protocol required')
    fit=set(protocol['fit_parent_ids']);internal=set(protocol['internal_val_parent_ids'])
    if fit&internal or len(fit)!=6 or len(internal)!=2:raise ValueError('fixed6fit/2internal disjoint parents required')
    jobs=plan['jobs'];parents=[j['parent_id'] for j in jobs]
    if len(jobs)!=8 or len(set(parents))!=8 or set(parents)!=fit|internal:raise ValueError('exact eight preassigned TRAIN parents required')
    if len({j['seed'] for j in jobs})!=8:raise ValueError('distinct preassigned parent seeds required')
    prepared=[]
    for job in jobs:
        if job['split']!='train':raise ValueError('official/dev VAL or TEST may not enter training source')
        config_path=root/job['config'];config=read(config_path);meta=config['P2_response_collection_metadata']
        if (sha(config_path)!=job['config_sha256'] or config.get('dataset_use')!='train'
                or config.get('seed')!=job['seed'] or meta.get('parent_id')!=job['parent_id']
                or meta.get('split')!='train' or meta.get('official_VAL_or_TEST_source') is not False
                or config.get('candidate_transformer_checkpoint_sha256')!=plan['candidate_checkpoint_sha256']):
            raise ValueError('preassigned config/parent/seed/TRAIN source identity mismatch')
        prepared.append(dict(job,role='fit' if job['parent_id'] in fit else 'internal_val',
            case_exists=(root/job['path']).is_dir()))
    return dict(schema=SCHEMA,plan_sha256=sha(plan_path),protocol_sha256=sha(protocol_path),
        fit_started=False,Native_started=False,actual_state_files_opened=False,jobs=prepared,protocol=protocol)


def extract_case(directory,parent_id,role):
    """Require source TRAIN role before opening actual robot/reference data."""
    directory=Path(directory).resolve();run=read(directory/'run_inputs.json');config=run['config']
    meta=config.get('P2_response_collection_metadata',{})
    if (run.get('TEST_read') is not False or config.get('dataset_use')!='train' or meta.get('split')!='train'
            or meta.get('parent_id')!=parent_id or meta.get('official_VAL_or_TEST_source') is not False
            or role not in ('fit','internal_val')):raise ValueError('only preassigned new TRAIN parent hardware data permitted')
    owner=read(directory.parent/(directory.name+'.owner.actual_exit.json'))
    worker=read(directory/'worker_exit.json');result=read(directory/'result.json')
    if (owner['actual_process_returncode']!=0 or worker.get('exit_code')!=0 or worker.get('native_closed') is not True
            or result.get('complete') is not True or result.get('source_unchanged') is not True):
        raise ValueError('closed actual childexit/source-unchanged case required; task failures remain included')
    for relative,digest in run['source_sha256'].items():
        if sha(directory/'source_snapshot'/relative)!=digest:raise ValueError('executed archived source identity mismatch: '+relative)
    paths=[directory/name for name in ('run_inputs.json','worker_exit.json','result.json','public_grasp_candidates_v5_decisions.jsonl','public_shadow/public_outputs.jsonl')]
    hashes={str(p):sha(p) for p in paths}
    episode=extract_transitions(lines(paths[3]),lines(paths[4]))
    if hashes!={str(p):sha(p) for p in paths}:raise ValueError('closed source changed during extraction')
    episode.update(episode_id=directory.name,parent_id=parent_id,role=role,
        source_files_sha256=hashes,source_config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())
    episode['audit']['oracle_reference']=oracle_reference_audit(episode)
    return episode


def prepare_dataset(plan_path,protocol_path,output,root=ROOT):
    """Explicitly compile all eight closed sources; do not select task successes."""
    contract=preflight_plan(plan_path,protocol_path,root)
    pending=[j['id'] for j in contract['jobs'] if not (Path(root)/j['path']).joinpath('worker_exit.json').is_file()]
    if pending:raise ValueError('wait for all eight actual closed TRAIN sources: '+','.join(pending))
    episodes=[]
    for job in contract['jobs']:
        directory=Path(root)/job['path']
        if read(directory/'run_inputs.json')['config']!=read(Path(root)/job['config']):
            raise ValueError('executed TRAIN config differs from preassigned pinned config')
        episodes.append(extract_case(directory,job['parent_id'],job['role']))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    records=[]
    for index,e in enumerate(episodes):
        name=f'episode_{index:02d}.npz';path=output/name
        arrays={key:e[key] for key in ('q','dq','tcp','times','refs','goals','phase','probe_mask','command_opening')}
        arrays.update({'h_'+key:value for key,value in e['h'].items()})
        with path.open('xb') as stream:np.savez_compressed(stream,**arrays)
        records.append(dict(file=name,sha256=sha(path),episode_id=e['episode_id'],parent_id=e['parent_id'],role=e['role'],
            limits=e['limits'].receipt(),audit=e['audit'],source_files_sha256=e['source_files_sha256']))
    sources=list(Path(__file__).parent.glob('*.py'))
    sources+=[ROOT/'our_mani/local_mpc_control_20261103/coupled_response.py',
        ROOT/'our_mani/local_mpc_p2_20261103/reference.py',ROOT/'our_mani/local_mpc_p2_20261103/kinematics.py']
    manifest=dict(schema=SCHEMA,complete=True,plan_path=str(Path(plan_path).resolve()),
        protocol_path=str(Path(protocol_path).resolve()),plan_sha256=contract['plan_sha256'],
        protocol_sha256=contract['protocol_sha256'],protocol=contract['protocol'],episodes=records,
        source_sha256={str(p.resolve()):sha(p) for p in sources},
        all_sources_dataset_use_train=True,official_or_development_VAL_or_TEST_opened=False,
        task_success_filter_used=False,fit_started=False,Native_started=False)
    with (output/'manifest.json').open('x') as stream:json.dump(manifest,stream,indent=2,allow_nan=False)
    return manifest


def load_dataset(output):
    output=Path(output);manifest=read(output/'manifest.json')
    if (manifest.get('schema')!=SCHEMA or manifest.get('complete') is not True
            or manifest.get('all_sources_dataset_use_train') is not True
            or manifest.get('official_or_development_VAL_or_TEST_opened') is not False
            or manifest.get('task_success_filter_used') is not False):raise ValueError('exact TRAIN-only unfiltered dataset required')
    if sha(manifest['plan_path'])!=manifest['plan_sha256'] or sha(manifest['protocol_path'])!=manifest['protocol_sha256']:
        raise ValueError('preassigned source/fit protocol changed after preparation')
    for path,digest in manifest['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('prepared data/math source identity changed: '+path)
    episodes=[]
    for row in manifest['episodes']:
        path=output/row['file']
        if sha(path)!=row['sha256']:raise ValueError('prepared sensor cache identity mismatch')
        with np.load(path,allow_pickle=False) as payload:
            arrays={key:np.asarray(payload[key]).copy() for key in payload.files}
        h={key[2:]:arrays.pop(key) for key in tuple(arrays) if key.startswith('h_')}
        if set(arrays)!={'q','dq','tcp','times','refs','goals','phase','probe_mask','command_opening'}:
            raise ValueError('only actual hardware/reference arrays may enter cache')
        e=dict(**arrays,h=h,limits=limits_from_receipt(row['limits']),
            **{key:row[key] for key in ('episode_id','parent_id','role','audit','source_files_sha256')})
        if row['role'] not in ('fit','internal_val'):raise ValueError('unknown preassigned training role')
        expected=manifest['protocol']['fit_parent_ids'] if row['role']=='fit' else manifest['protocol']['internal_val_parent_ids']
        if row['parent_id'] not in expected:raise ValueError('cached parent/role mismatch')
        episodes.append(e)
    return episodes,manifest
