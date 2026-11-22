"""P2 CPU admission check or explicitly selected residual fitting.

No optimization occurs with --check. This driver does not launch Native, use old
expert advice or evaluate formal TEST. It trains a differentiable executor-
reference/actual-state chain conditioned on known future joint commands, not a
policy that chooses those commands or a complete contact model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import ServoWindowDataset,file_sha,train_statistics,common_reference_limits
from .losses import horizon_metrics,servo_loss,teacher_forced_prediction,position_precision_gate
from .servo import JointServoResidual,rollout_recorded_references,rollout_regenerated_references,baseline_rollout


def move(batch,device):
    if isinstance(batch,dict):return {k:move(v,device) for k,v in batch.items()}
    return batch.to(device) if isinstance(batch,torch.Tensor) else batch


def load_validated_fk(path,receipt):
    path,receipt=Path(path),Path(receipt)
    snapshot,result=json.loads(path.read_text()),json.loads(receipt.read_text())
    if (path.parent.resolve()!=receipt.parent.resolve() or snapshot.get('Native_FK_validation_passed') is not True
            or result.get('complete') is not True or result.get('passed') is not True
            or result.get('Native_controlled_physics_ticks')!=0 or result.get('restoration_checked') is not True
            or result.get('maximum_TCP_position_error_m',float('inf'))>1e-5
            or result.get('maximum_rotation_error_rad',float('inf'))>1e-4):
        raise ValueError('source-bound restoring zero-step Native FK canary must pass before residual fitting')
    return snapshot,dict(snapshot_path=str(path.resolve()),snapshot_sha256=file_sha(path),
                         validation_path=str(receipt.resolve()),validation_sha256=file_sha(receipt))


SCOPE = 'free_actual_state_and_regenerated_executor_reference_rollout_under_known_joint_commands_held_fixed_at_anchor'
LOSS_WEIGHTS = dict(teacher_forced_recorded=1.,free_recorded=.5,free_regenerated=1.)
SELECTION_RULE = dict(order=['worst_episode_over_5cm_fraction','worst_episode_max_TCP_m',
                            'worst_episode_mean_TCP_m','regenerated_known_command_loss'],
                      comparison='lexicographic_minimum_strict_improvement_first_epoch_wins_ties',
                      admission='all_16_horizons_all_real_samples_finite_no_infeasible',
                      validation_used_for_scaling=False)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def _save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def _batch_size(batch):
    # The admitted v2 model uses a [B,17] tensor. Explicit q support keeps the
    # accounting correct if an input-only named state is admitted later.
    state = batch['initial_state']
    q = state['q'] if isinstance(state, dict) else state
    if not isinstance(q, torch.Tensor) or q.ndim < 2 or q.shape[0] <= 0:
        raise ValueError('a nonempty batched actual state is required')
    size = int(q.shape[0])
    if len(batch['episode_id']) != size or batch['valid'].shape[0] != size:
        raise ValueError('batch identities and labels must match actual-state batch size')
    return size


def _weight_mass(batch, size):
    weight = batch.get('weight')
    if weight is None:
        return float(size)
    if (weight.shape != (size,) or not bool(torch.isfinite(weight).all())
            or not bool((weight > 0).all())):
        raise ValueError('one finite positive balancing weight per window required')
    return float(weight.detach().double().sum().cpu())


def _verify_sources(identity):
    for path, digest in identity['source_sha256'].items():
        if file_sha(path) != digest:
            raise RuntimeError('fitting source changed: ' + path)
    if file_sha(identity['manifest']['path']) != identity['manifest']['sha256']:
        raise RuntimeError('split/episode manifest changed during fitting')
    for name in ('snapshot', 'validation'):
        if file_sha(identity['fk'][name + '_path']) != identity['fk'][name + '_sha256']:
            raise RuntimeError('validated FK identity changed during fitting')


def _archive_inputs(output, identity):
    archive = output / 'source_snapshot'
    archive.mkdir()
    copies = []
    sources = [(Path(p), 'package/' + Path(p).name, sha)
               for p, sha in sorted(identity['source_sha256'].items())]
    sources += [(Path(identity['manifest']['path']), 'episode_manifest.json', identity['manifest']['sha256']),
                (Path(identity['fk']['snapshot_path']), 'fk_snapshot.json', identity['fk']['snapshot_sha256']),
                (Path(identity['fk']['validation_path']), 'fk_validation.json', identity['fk']['validation_sha256'])]
    for source, relative, expected in sources:
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if file_sha(source) != expected:
            raise RuntimeError('input changed before archival: ' + str(source))
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
        if file_sha(target) != expected or file_sha(source) != expected:
            raise RuntimeError('archived input fingerprint mismatch')
        copies.append(dict(source=str(source), archive_path=str(target.resolve()), sha256=expected))
    receipt = dict(schema='P2_v2_servo_training_input_archive_v1', files=copies,
                   byte_exact=True, formal_TEST_used=False)
    _write_json(output / 'source_archive.json', receipt)
    return dict(path=str((output / 'source_archive.json').resolve()),
                sha256=file_sha(output / 'source_archive.json'), files=len(copies))


def _finalize_training(output, history, identity, *, epochs, optimization_steps, started_at, started_clock):
    if (len(history) != epochs or [row['epoch'] for row in history] != list(range(1, epochs + 1))
            or optimization_steps <= 0):
        raise RuntimeError('all requested epochs and real optimizer steps are required for completion')
    _verify_sources(identity)
    selected = min(history, key=lambda row: tuple(row['validation']['selection_key']))
    status = dict(schema='P2_v2_servo_training_status_v1', status='complete', training_complete=True,
                  started_at=started_at, completed_at=_utc_now(), wall_seconds=time.perf_counter()-started_clock,
                  epochs_requested=epochs, epochs_completed=len(history),
                  optimization_steps_completed=optimization_steps,
                  selected_epoch=selected['epoch'], selection_metric='validation.fixed_lexicographic_position_tail_priority',
                  selection_value=selected['validation']['selection_key'],
                  selection_rule=SELECTION_RULE,
                  scope=SCOPE, full_known_reference_chain_qualified=False, formal_TEST_used=False,
                  normalization_source_split='train',
                  validation_or_holdout_used_for_scaling=False,
                  history_sha256=file_sha(output / 'history.json'),
                  identity_sha256=file_sha(output / 'identity.json'))
    selection_receipts = {}
    for name, expected_epoch in (('best.pt', selected['epoch']), ('last.pt', epochs)):
        path = output / name
        before_sha = file_sha(path)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload['epoch'] != expected_epoch or payload.get('optimization_steps', 0) <= 0:
            raise RuntimeError('checkpoint epoch/optimization evidence disagrees with completed history')
        if (name == 'best.pt' and
                (payload['validation']['loss'] != selected['validation']['loss']
                 or payload['validation']['selection_key'] != selected['validation']['selection_key'])):
            raise RuntimeError('selected checkpoint no longer matches validation selection history')
        # Preserve the selected weights, epoch, and steps at selection. This new
        # metadata describes the completed run, not additional updates to them.
        payload['training_complete'] = True
        payload['training_status'] = status.copy()
        payload['model_contract'] = dict(payload['model_contract'], training_complete=True,
                                         checkpoint_weights_epoch=expected_epoch)
        _save_checkpoint(path, payload)
        selection_receipts[name] = dict(epoch=expected_epoch, selection_checkpoint_sha256=before_sha,
                                        completed_checkpoint_sha256=file_sha(path),
                                        optimizer_steps_at_checkpoint=payload['optimization_steps'],
                                        selected_weights_unchanged=True)
    status['checkpoints'] = selection_receipts
    status['best_checkpoint_sha256'] = file_sha(output / 'best.pt')
    status['last_checkpoint_sha256'] = file_sha(output / 'last.pt')
    _write_json(output / 'training_status.json', status)
    return status


@torch.no_grad()
def evaluate(model,dataset,device,batch_size=64):
    limits,_=common_reference_limits(dataset)
    predictions={mode:{} for mode in ('regenerated','recorded')};targets={};episode_groups={}
    loss_sums={mode:0. for mode in predictions};weight_mass,count=0.,0
    for raw in DataLoader(dataset,batch_size=batch_size,shuffle=False):
        batch=move(raw,device);model.eval()
        values=dict(regenerated=rollout_regenerated_references(model,batch,limits),
                    recorded=rollout_recorded_references(model,batch['initial_state'],batch['references']))
        size=_batch_size(raw);mass=_weight_mass(raw,size);offset=count
        for mode,prediction in values.items():
            loss_sums[mode]+=float(servo_loss(prediction,batch))*mass
            for key in ('q','dq','opening','tcp_matrix'):
                predictions[mode].setdefault(key,[]).append(prediction[key].cpu())
        weight_mass+=mass;count+=size
        for key in ('target_states','target_tcp','valid'):
            targets.setdefault(key,[]).append(raw[key])
        for index,episode in enumerate(raw['episode_id']):
            episode_groups.setdefault(episode,[]).append(offset+index)
    if not count:raise ValueError('empty evaluation cohort is not a passed result')
    b={k:torch.cat(v) for k,v in targets.items()}
    reports={}
    for mode,parts in predictions.items():
        p={k:torch.cat(v) for k,v in parts.items()}
        reports[mode]=dict(loss=loss_sums[mode]/weight_mass,horizons=horizon_metrics(p,b),
            precision_gate=position_precision_gate(p,b),
            by_episode_precision={e:position_precision_gate({k:v[idx] for k,v in p.items()},{k:v[idx] for k,v in b.items()})
                                  for e,idx in episode_groups.items()},
            by_episode={e:horizon_metrics({k:v[idx] for k,v in p.items()},{k:v[idx] for k,v in b.items()})
                        for e,idx in episode_groups.items()})
    episode_precision=reports['regenerated']['by_episode_precision']
    if not all(row['all_horizons_have_labels'] for row in episode_precision.values()):
        raise ValueError('all 16 horizons need actual validation targets before checkpoint selection')
    key=[max(row['above_5cm_fraction'] for row in episode_precision.values()),
         max(row['max_m'] for row in episode_precision.values()),
         max(row['mean_m'] for row in episode_precision.values()),reports['regenerated']['loss']]
    if not np.isfinite(key).all():raise ValueError('nonfinite checkpoint selection key')
    return dict(loss=reports['regenerated']['loss'],selection_metric='fixed_lexicographic_position_tail_priority',
                selection_key=key,selection_rule=SELECTION_RULE,
                windows=count,weight_mass=weight_mass,
                loss_aggregation='exact_positive_parent_balancing_weight_mass',
                regenerated=reports['regenerated'],recorded_reference_auxiliary=reports['recorded'],
                precision_gate=reports['regenerated']['precision_gate'],scope=SCOPE,
                future_actual_state_or_h_used=False,known_future_joint_commands_used=True,
                full_policy_action_execution_chain_qualified=False,
                full_known_reference_chain_qualified=False,formal_TEST_used=False,Native_task_success_measured=False)


def run(args):
    if args.epochs<=0 or args.batch_size<=0 or not np.isfinite(args.learning_rate) or args.learning_rate<=0:
        raise ValueError('positive epochs, batch size, and finite learning rate required')
    torch.set_num_threads(2)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    manifest=json.loads(Path(args.manifest).read_text())
    datasets={split:ServoWindowDataset(manifest,split,horizon_ticks=16) for split in ('train','validation')}
    if not all(len(d) for d in datasets.values()):
        raise ValueError('nonempty distinct-parent TRAIN and validation cohorts required; no automatic split')
    snapshot,fk_identity=load_validated_fk(args.fk_snapshot,args.fk_validation)
    reference_limits,reference_limits_metadata=common_reference_limits(datasets['train'])
    _,validation_reference_limits=common_reference_limits(datasets['validation'])
    if validation_reference_limits!=reference_limits_metadata:
        raise ValueError('TRAIN and validation reference contracts differ')
    stats=train_statistics(datasets['train'])
    model=JointServoResidual(snapshot,**{k:stats[k] for k in
        ('residual_q_bound','residual_dq_bound','feature_mean','feature_std')})
    identity=dict(manifest=dict(path=str(Path(args.manifest).resolve()),sha256=file_sha(args.manifest)),
                  model_version='P2_reference_tracking_servo_v2',fk=fk_identity,normalization=stats,
                  reference_limits=reference_limits_metadata,loss_weights=LOSS_WEIGHTS,selection_rule=SELECTION_RULE,
                  future_actual_state_or_h_used_in_free_rollout=False,
                  source_sha256={str(p.resolve()):file_sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                  formal_TEST_used=False,old_expert_advice_used=False,P0_used_for_fitting=False)
    if args.check:
        model.eval()
        with torch.no_grad():
            batch=next(iter(DataLoader(datasets['train'],batch_size=2)))
            one=teacher_forced_prediction(model,batch)
            multi=rollout_recorded_references(model,batch['initial_state'],batch['references'])
            regenerated=rollout_regenerated_references(model,batch,reference_limits)
            report=dict(mode='CPU_contract_check_only',optimizer_steps=0,
                        summaries={s:d.summary for s,d in datasets.items()},identity=identity,
                        one_step_loss=float(servo_loss(one,batch)),recorded_multistep_loss=float(servo_loss(multi,batch)),
                        regenerated_multistep_loss=float(servo_loss(regenerated,batch)),
                        regenerated_precision_gate=position_precision_gate(regenerated,batch),
                        baselines={name:horizon_metrics(baseline_rollout(model,batch['initial_state'],batch['references'],name),batch)
                                   for name in ('hold','constant_velocity','instant_reference')})
        print(json.dumps(report,ensure_ascii=False,allow_nan=False));return report
    # This branch is available for later explicit use; implementing/testing this
    # file does not authorize its invocation or any optimization by the agent.
    if args.output is None:raise ValueError('explicit fresh output namespace required for --fit')
    output=Path(args.output)
    if output.exists():raise ValueError('output already exists; never overwrite/resume an old run')
    device=torch.device(args.device)
    model.to(device);output.mkdir(parents=True)
    _verify_sources(identity)
    identity['source_archive'] = _archive_inputs(output, identity)
    _write_json(output/'identity.json', identity)
    started_at, started_clock = _utc_now(), time.perf_counter()
    history=[];optimization_steps=0
    initial_status=dict(schema='P2_v2_servo_training_status_v1',status='running',training_complete=False,
                        started_at=started_at,epochs_requested=args.epochs,epochs_completed=0,
                        optimization_steps_completed=0,scope=SCOPE,formal_TEST_used=False,
                        full_known_reference_chain_qualified=False,normalization_source_split='train',
                        validation_or_holdout_used_for_scaling=False)
    _write_json(output/'training_status.json',initial_status)
    try:
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
        loader=DataLoader(datasets['train'],batch_size=args.batch_size,shuffle=True)
        best_key=None
        for epoch in range(1,args.epochs+1):
            _verify_sources(identity)
            epoch_clock=time.perf_counter()
            model.train();total,count,weight_mass=0.,0,0.
            for raw in loader:
                batch=move(raw,device);optimizer.zero_grad(set_to_none=True)
                one=teacher_forced_prediction(model,batch)
                multi=rollout_recorded_references(model,batch['initial_state'],batch['references'])
                regenerated=rollout_regenerated_references(model,batch,reference_limits)
                loss=(LOSS_WEIGHTS['teacher_forced_recorded']*servo_loss(one,batch)
                      +LOSS_WEIGHTS['free_recorded']*servo_loss(multi,batch)
                      +LOSS_WEIGHTS['free_regenerated']*servo_loss(regenerated,batch))
                if not bool(torch.isfinite(loss)):raise ValueError('nonfinite residual training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                optimization_steps+=1;model.metadata['trained']=True
                size=_batch_size(raw);mass=_weight_mass(raw,size)
                total+=float(loss.detach())*mass;weight_mass+=mass;count+=size
            validation=evaluate(model,datasets['validation'],device,args.batch_size)
            row=dict(epoch=epoch,train_loss=total/weight_mass,train_windows=count,train_weight_mass=weight_mass,
                     loss_aggregation='exact_positive_parent_balancing_weight_mass',
                     validation=validation,wall_seconds=time.perf_counter()-epoch_clock)
            history.append(row)
            _write_json(output/'history.json',history)
            payload=dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},identity=identity,
                         model_contract=dict(model.metadata,training_complete=False),epoch=epoch,normalization=stats,
                         validation=validation,optimization_steps=optimization_steps,training_complete=False,
                         checkpoint_selection=dict(metric='validation.fixed_lexicographic_position_tail_priority',
                             value=validation['selection_key'],epoch=epoch,rule=SELECTION_RULE),
                         full_known_reference_chain_qualified=False,formal_TEST_used=False)
            _save_checkpoint(output/'last.pt',payload)
            candidate_key=tuple(validation['selection_key'])
            if best_key is None or candidate_key<best_key:
                best_key=candidate_key;_save_checkpoint(output/'best.pt',payload)
            _write_json(output/'training_status.json',dict(initial_status,epochs_completed=epoch,
                        optimization_steps_completed=optimization_steps,last_completed_epoch=epoch,
                        wall_seconds=time.perf_counter()-started_clock))
            print(json.dumps(dict(epoch=epoch,train_loss=total/weight_mass,validation_loss=validation['loss'],
                                  selection_key=validation['selection_key'],
                                  position_precision_passed=validation['precision_gate']['passed'],wall_seconds=row['wall_seconds'])),flush=True)
        return _finalize_training(output,history,identity,epochs=args.epochs,optimization_steps=optimization_steps,
                                  started_at=started_at,started_clock=started_clock)
    except BaseException as error:
        _write_json(output/'training_status.json',dict(initial_status,status='failed',epochs_completed=len(history),
                    optimization_steps_completed=optimization_steps,failed_at=_utc_now(),
                    wall_seconds=time.perf_counter()-started_clock,error_type=type(error).__name__,error=str(error)))
        raise



def parser():
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check',action='store_true');modes.add_argument('--fit',action='store_true')
    p.add_argument('--manifest',required=True);p.add_argument('--fk-snapshot',required=True);p.add_argument('--fk-validation',required=True)
    p.add_argument('--output');p.add_argument('--device',default='cpu');p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=20261103)
    return p


if __name__=='__main__':run(parser().parse_args())
