#!/usr/bin/env python3
"""CPU-only P2 VAL audit under recorded references and known joint commands.

No Native, fitting, policy action selection, old advice, or formal TEST. The
regenerated-reference path uses only anchor actual state/controller memory and
known future joint commands. Its next reseed uses its own predicted q/dq.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from our_mani.local_mpc_p2_20261103.dataset import ServoWindowDataset, file_sha
from our_mani.local_mpc_p2_20261103.reference import FailureCode, ReferenceLimits, reference_step
from our_mani.local_mpc_p2_20261103.servo import baseline_rollout, rollout_recorded_references
from our_mani.local_mpc_p2_20261103.servo_adapter import TorchServoTransition


SCOPE = 'offline_known_recorded_joint_commands_not_policy_action_choice'
FIELDS = ('q', 'dq', 'opening', 'tcp_matrix')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def limits_from_episode(episode):
    initial = json.loads((Path(episode['path'])/'initial_snapshot.json').read_text())
    limits = initial['joint_limits']
    return ReferenceLimits(lower=limits['lower'], upper=limits['upper'], velocity=limits['velocity'],
                           acceleration=limits['controller_reference_acceleration'],
                           jerk=limits['controller_reference_jerk'], reference_velocity_cap=3.)


@torch.inference_mode()
def rollout_regenerated_references(model, initial_state, initial_h, command_goal_q,
                                    command_opening, valid, limits):
    """Free rollout. No future measured state or future h parameter exists.

    Once a row fails, its later predictions are explicitly unavailable. No
    clipping, fallback hold, reference substitution, or failed-row deletion.
    """
    if initial_state.ndim != 2 or initial_state.shape[-1] != 17:
        raise ValueError('anchor actual state [B,17] required')
    b, horizon = valid.shape
    if (initial_state.shape[0] != b or command_goal_q.shape != (b,horizon,7)
            or command_opening.shape != (b,horizon) or valid.dtype != torch.bool):
        raise ValueError('aligned known commands and real-label mask required')
    state = initial_state.detach().cpu().clone()
    h = {k:v.detach().cpu().numpy().copy() if isinstance(v,torch.Tensor) else np.asarray(v).copy()
         for k,v in initial_h.items()}
    goals = command_goal_q.detach().cpu().numpy().astype(np.float64)
    openings = command_opening.detach().cpu().numpy().astype(np.float64)
    mask = valid.detach().cpu().numpy()
    alive = np.ones(b,dtype=bool)
    available = np.zeros((b,horizon),dtype=bool)
    first_failure = np.full(b,-1,dtype=np.int64)
    failure_codes = np.zeros((b,horizon),dtype=np.int8)
    regenerated = np.full((b,horizon,22),np.nan,dtype=np.float64)
    shapes = {'q':(7,), 'dq':(7,), 'opening':(2,), 'tcp_matrix':(4,4)}
    outputs = {k:torch.full((b,horizon,*shape),float('nan')) for k,shape in shapes.items()}
    for tick in range(horizon):
        active = alive & mask[:,tick]
        if not active.any():
            continue
        step = reference_step(state[:,:7].numpy(),state[:,7:14].numpy(),goals[:,tick],h,limits,
                              active_mask=active,apply_mask=active)
        failed = active & (~step['feasible'] | ~step['applied'])
        codes = step['failure_code'].copy()
        # Applied=False with OK on an active row is a contract error, not an
        # invented physical transition.
        if np.any(failed & (codes == int(FailureCode.OK))):
            raise RuntimeError('active feasible reference was not dispatched')
        failure_codes[failed,tick] = codes[failed]
        first_failure[failed] = tick
        alive[failed] = False
        h = step['next_h']
        good = active & ~failed
        if not good.any():
            continue
        reference = np.c_[step['q_ref'][good],step['dq_ref'][good],step['ddq_ref'][good],openings[good,tick]]
        if not np.isin(reference[:,-1],(0.,1.)).all():
            raise ValueError('known binary gripper commands required')
        ids = torch.from_numpy(np.flatnonzero(good))
        prediction = model(state[ids],torch.as_tensor(reference,dtype=state.dtype))
        if not all(bool(torch.isfinite(prediction[k]).all()) for k in (*FIELDS,'state')):
            raise ValueError('nonfinite model output cannot be dropped')
        state[ids] = prediction['state']
        # h is controller memory. It never receives measured future states or
        # future memory snapshots from the episode.
        h['gripper_target'][good] = openings[good,tick]
        for k in FIELDS:
            outputs[k][ids,tick] = prediction[k]
        regenerated[good,tick] = reference
        available[good,tick] = True
    return dict(prediction=outputs,available=torch.from_numpy(available),
                references=regenerated,first_failure_tick=first_failure,failure_codes=failure_codes,
                final_predicted_state=state)


def _stats(values):
    if not len(values):
        return dict(mean=None,p95=None,max=None)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite predicted or measured eligible sample cannot be filtered')
    return dict(mean=float(values.mean()),p95=float(np.quantile(values,.95)),max=float(values.max()))


def metric_group(prediction,target_states,target_tcp,expected,available):
    expected = np.asarray(expected,dtype=bool)
    available = np.asarray(available,dtype=bool)
    if expected.shape != available.shape:
        raise ValueError('explicit expected and prediction availability masks required')
    observed = expected & available
    expected_count, observed_count = int(expected.sum()), int(observed.sum())
    missing = expected_count-observed_count
    p = {k:np.asarray(v)[observed] for k,v in prediction.items() if k in FIELDS}
    truth, tcp = np.asarray(target_states)[observed], np.asarray(target_tcp)[observed]
    if observed_count:
        if not all(np.isfinite(a).all() for a in (*p.values(),truth,tcp)):
            raise ValueError('nonfinite available cohort sample cannot be removed')
        relative = np.swapaxes(p['tcp_matrix'][...,:3,:3],-1,-2) @ tcp[...,:3,:3]
        arrays = dict(TCP_error_m=np.linalg.norm(p['tcp_matrix'][...,:3,3]-tcp[...,:3,3],axis=-1),
                      rotation_error_rad=np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1)),
                      aperture_MAE=np.abs(p['opening']-truth[...,14:16]).mean(-1),
                      q_rms_rad=np.sqrt(np.square(p['q']-truth[...,:7]).mean(-1)),
                      dq_rms_rad_s=np.sqrt(np.square(p['dq']-truth[...,7:14]).mean(-1)))
    else:
        arrays = {k:np.empty(0) for k in ('TCP_error_m','rotation_error_rad','aperture_MAE','q_rms_rad','dq_rms_rad_s')}
    result = dict(expected_target_samples=expected_count,available_predictions=observed_count,
                  infeasible_or_unavailable_predictions=missing,complete_cohort=missing==0,
                  statistics_scope='all_expected_samples' if missing==0 else 'available_prefix_only_incomplete_cohort_no_pass')
    result.update({k:_stats(v) for k,v in arrays.items()})
    position = result['TCP_error_m']
    result['precision_gate'] = dict(mean_target_m=.03,all_samples_upper_bound_m=.05,
        passed=bool(expected_count and not missing and position['mean']<=.03 and position['max']<=.05))
    return result


def metrics(prediction,target_states,target_tcp,valid,available):
    return dict(all_available_target_ticks=metric_group(prediction,target_states,target_tcp,valid,available),
                horizons={str(t+1):dict(time_seconds=(t+1)*.05,
                    **metric_group({k:v[:,t] for k,v in prediction.items()},target_states[:,t],target_tcp[:,t],
                                   valid[:,t],available[:,t])) for t in range(valid.shape[1])})


def evaluate(args):
    torch.set_num_threads(args.threads)
    if args.batch_size<=0 or args.threads<=0:
        raise ValueError('positive CPU batch size and thread count required')
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError('fresh evaluation output required; never overwrite existing artifacts')
    manifest_path = Path(args.manifest).resolve(strict=True)
    checkpoint_path = Path(args.checkpoint).resolve(strict=True)
    adapter = TorchServoTransition.from_checkpoint(checkpoint_path,expected_sha256=args.expected_checkpoint_sha256)
    if file_sha(manifest_path)!=adapter.evidence['identity']['manifest']['sha256']:
        raise ValueError('evaluation manifest differs from fitted split/episode identity')
    manifest = json.loads(manifest_path.read_text())
    dataset = ServoWindowDataset(manifest,'validation',horizon_ticks=16)
    if not len(dataset):
        raise ValueError('empty VAL cohort cannot pass')
    output.mkdir(parents=True)
    start = time.perf_counter()
    write_json(output/'status.json',dict(status='running',complete=False,scope=SCOPE,formal_TEST_read=False))
    try:
        model=adapter.model
        buffers={name:{k:[] for k in FIELDS} for name in ('NN_recorded','hold','constant_velocity','instant_reference','NN_regenerated')}
        targets,tcps,masks,regenerated_available=[],[],[],[]
        all_recorded_refs,all_generated_refs,failures,codes=[],[],[],[]
        episode_ids,start_ticks=[],[]
        ledger=[]
        for episode_index,row in enumerate(dataset.rows):
            indices=[i for i,(e,_) in enumerate(dataset.anchors) if e==episode_index]
            limits=limits_from_episode(row)
            ledger.append(dict(episode_id=row['episode_id'],parent_id=row['parent_id'],split='validation',
                               path=row['path'],expected_windows=len(indices),all_source_windows_retained=True,
                               worker_failed_prefix=row['failed'],source_files=row['files']))
            for batch in DataLoader(Subset(dataset,indices),batch_size=args.batch_size,shuffle=False):
                with torch.inference_mode():
                    recorded=rollout_recorded_references(model,batch['initial_state'],batch['references'])
                    predictions={'NN_recorded':recorded}
                    for kind in ('hold','constant_velocity','instant_reference'):
                        predictions[kind]=baseline_rollout(model,batch['initial_state'],batch['references'],kind)
                    regenerated=rollout_regenerated_references(model,batch['initial_state'],batch['initial_actuator_state'],
                        batch['command_goal_q'],batch['command_opening'],batch['valid'],limits)
                    predictions['NN_regenerated']=regenerated['prediction']
                for name,prediction in predictions.items():
                    for key in FIELDS:buffers[name][key].append(prediction[key].cpu().numpy())
                targets.append(batch['target_states'].numpy());tcps.append(batch['target_tcp'].numpy());masks.append(batch['valid'].numpy())
                regenerated_available.append(regenerated['available'].numpy())
                all_recorded_refs.append(batch['references'].numpy());all_generated_refs.append(regenerated['references'])
                failures.append(regenerated['first_failure_tick']);codes.append(regenerated['failure_codes'])
                episode_ids.extend(batch['episode_id']);start_ticks.extend(batch['start_tick'].tolist())
        predictions={name:{k:np.concatenate(parts) for k,parts in value.items()} for name,value in buffers.items()}
        target,tcp,valid=np.concatenate(targets),np.concatenate(tcps),np.concatenate(masks)
        available=np.concatenate(regenerated_available);first_failure=np.concatenate(failures);failure_codes=np.concatenate(codes)
        recorded_refs,generated_refs=np.concatenate(all_recorded_refs),np.concatenate(all_generated_refs)
        if len(episode_ids)!=len(dataset) or len(set(zip(episode_ids,start_ticks)))!=len(dataset):
            raise RuntimeError('missing/duplicated VAL windows')
        comparison={name:metrics(pred,target,tcp,valid,available if name=='NN_regenerated' else valid)
                    for name,pred in predictions.items()}
        per_episode={name:{episode:metrics({k:v[np.asarray(episode_ids)==episode] for k,v in pred.items()},
            target[np.asarray(episode_ids)==episode],tcp[np.asarray(episode_ids)==episode],valid[np.asarray(episode_ids)==episode],
            (available if name=='NN_regenerated' else valid)[np.asarray(episode_ids)==episode])
            for episode in sorted(set(episode_ids))} for name,pred in predictions.items()}
        ref_mask=valid & available
        delta=generated_refs[ref_mask]-recorded_refs[ref_mask]
        counts=Counter(int(v) for v in failure_codes.ravel() if v)
        reference_comparison=dict(samples_compared=int(ref_mask.sum()),expected_samples=int(valid.sum()),
            incomplete_due_to_infeasibility=int((valid & ~available).sum()),
            qref_abs_error_rad=_stats(np.max(np.abs(delta[:,:7]),axis=-1)),
            vref_abs_error_rad_s=_stats(np.max(np.abs(delta[:,7:14]),axis=-1)),
            aref_abs_error_rad_s2=_stats(np.max(np.abs(delta[:,14:21]),axis=-1)),
            gripper_mismatches=int(np.count_nonzero(delta[:,-1])) if len(delta) else 0,
            note='Reference differences are expected after predicted-state reseeding; recorded references are a comparison, never regenerated-path input.')
        arrays=dict(target_states=target,target_tcp=tcp,valid=valid,regenerated_available=available,
                    recorded_references=recorded_refs,regenerated_references=generated_refs,
                    first_failure_tick=first_failure,reference_failure_codes=failure_codes,
                    episode_id=np.asarray(episode_ids),start_tick=np.asarray(start_ticks))
        for name,prediction in predictions.items():
            for key,value in prediction.items():arrays[name+'__'+key]=value
        np.savez_compressed(output/'predictions.npz',**arrays)
        diagnostic=ROOT/'artifacts/framework/local_mpc_20261102/P2_initial_diagnostics_v1/dynamic_fk_and_dataset_admission.json'
        report=dict(schema='P2_offline_all_VAL_execution_comparison_v1',complete=True,device='cpu',torch_threads=args.threads,
            batch_size=args.batch_size,wall_seconds=time.perf_counter()-start,scope=SCOPE,
            formal_TEST_read=False,optimizer_steps=0,Native_started=False,policy_action_choice_evaluated=False,
            contact_transition_qualified=False,held_mode='fixed_at_each_actual_anchor',
            full_known_reference_chain_qualified=False,
            checkpoint=adapter.evidence,manifest=dict(path=str(manifest_path),sha256=file_sha(manifest_path)),
            frozen_package_source_sha256=adapter.evidence['identity']['source_sha256'],
            evaluation_source=dict(path=str(Path(__file__).resolve()),sha256=file_sha(__file__)),
            dataset_summary=dataset.summary,expected_windows=len(dataset),processed_windows=len(episode_ids),
            all_expected_windows_processed=True,source_success_filter_used=False,episode_ledger=ledger,
            comparison=comparison,by_episode=per_episode,reference_comparison=reference_comparison,
            regenerated_failure_windows=int((first_failure>=0).sum()),
            regenerated_reference_failure_codes={FailureCode(k).name:v for k,v in sorted(counts.items())},
            infeasible_windows=[dict(episode_id=episode_ids[i],start_tick=start_ticks[i],
                first_failure_relative_tick=int(t+1),relative_seconds=float((t+1)*.05),
                code=FailureCode(int(failure_codes[i,t])).name) for i,t in enumerate(first_failure) if t>=0],
            regenerated_inputs='anchor_actual_q_dq_fingers_held_actual_h_and_known_future_joint_goal_grip_commands',
            regenerated_future_actual_state_or_h_used=False,
            dynamic_FK_diagnostic=dict(path=str(diagnostic),sha256=file_sha(diagnostic),
                actual_TCP_labels_projected=False,approximate_kinematics=True) if diagnostic.is_file() else None,
            predictions=dict(path=str(output/'predictions.npz'),sha256=file_sha(output/'predictions.npz')))
        write_json(output/'report.json',report)
        with (output/Path(__file__).name).open('xb') as stream:stream.write(Path(__file__).read_bytes())
        # Recheck the strict checkpoint/source identity after all CPU computation.
        TorchServoTransition.from_checkpoint(checkpoint_path,expected_sha256=adapter.evidence['sha256'])
        write_json(output/'status.json',dict(status='complete',complete=True,scope=SCOPE,formal_TEST_read=False,
            checkpoint_sha256=adapter.evidence['sha256'],report_sha256=file_sha(output/'report.json'),
            processed_windows=len(dataset),wall_seconds=time.perf_counter()-start))
        print(json.dumps(dict(output=str(output),processed_windows=len(dataset),wall_seconds=report['wall_seconds'],
            summary={name:value['all_available_target_ticks'] for name,value in comparison.items()},
            regenerated_failure_windows=report['regenerated_failure_windows']),allow_nan=False))
        return report
    except BaseException as error:
        write_json(output/'status.json',dict(status='failed',complete=False,scope=SCOPE,formal_TEST_read=False,
            error_type=type(error).__name__,error=str(error),wall_seconds=time.perf_counter()-start))
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default=str(ROOT/'artifacts/framework/local_mpc_20261102/P2_data_v1/manifest.json'))
    p.add_argument('--checkpoint',default=str(ROOT/'artifacts/framework/local_mpc_20261102/P2_servo_training_v1/best.pt'))
    p.add_argument('--expected-checkpoint-sha256',default='559faa65b4e56f91dd009514454bcb51c970ffec0fd2e473adf8c8777e0bef89')
    p.add_argument('--output',default=str(ROOT/'artifacts/framework/local_mpc_20261102/offline_P2_evaluation_v1'))
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--threads',type=int,default=2)
    return p


if __name__=='__main__':evaluate(parser().parse_args())
