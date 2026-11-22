"""Joint/known-hardware diagnostics, with no future-state teacher forcing."""
from copy import deepcopy
import numpy as np
from ..local_mpc_p2_20261103.reference import reference_step
from .ridge import DT


def stats(values):
    values=np.asarray(values,float)
    return (dict(count=0,mean=None,p50=None,p95=None,maximum=None) if not len(values)
            else dict(count=len(values),mean=float(values.mean()),p50=float(np.percentile(values,50)),
                      p95=float(np.percentile(values,95)),maximum=float(values.max())))


def _errors(q,dq,true_q,true_dq,tcp,fk):
    predicted=fk(q)[...,:3,3];actual_fk=fk(true_q)[...,:3,3]
    return dict(q_max_abs_rad=np.max(np.abs(q-true_q),axis=-1),
        dq_max_abs_rad_s=np.max(np.abs(dq-true_dq),axis=-1),
        joint_model_FK_position_l2_m=np.linalg.norm(predicted-actual_fk,axis=-1),
        predicted_FK_to_public_TCP_l2_m=np.linalg.norm(predicted-tcp,axis=-1),
        measured_q_FK_to_public_TCP_l2_m=np.linalg.norm(actual_fk-tcp,axis=-1))


def evaluate(model,episodes,fk):
    """H1 conditional plus H1..H16 free recorded/regenerated-reference loops.

    Recorded future commands/references are an exogenous input-sequence
    diagnostic. They are not asserted to be available to a deployed planner.
    Probe cohorts are explicitly anchor-based; a separate pure-probe-window
    cohort avoids calling later reopen transitions a probe-only experiment.
    """
    result=dict(scope='new TRAIN6 fit / TRAIN2 internal reporting only',roles={},
        actual_TCP_is_public_robot_sensor_not_object_GT=True,Native_held_or_success_used=False,
        future_actual_q_or_dq_teacher_forcing=False,forecast_future_commands_are_recorded_exogenous_inputs=True)
    for role in ('fit','internal_val'):
        subset=[e for e in episodes if e['role']==role]
        conditional=[];rollouts={mode:[] for mode in ('recorded_reference','regenerated_reference')}
        for e in subset:
            n=len(e['refs']);ref=e['refs']
            pred=model.predict_joint(e['q'][:-1],e['dq'][:-1],ref[:,:7],ref[:,7:14],ref[:,14:])
            errors=_errors(pred['q'],pred['dq'],e['q'][1:],e['dq'][1:],e['tcp'][1:],fk)
            conditional.append((e,errors))
            for mode in rollouts:
                q=e['q'][:-1].copy();dq=e['dq'][:-1].copy();h=deepcopy(e['h'])
                alive=np.ones(n,bool);data=[];failures=[]
                for k in range(16):
                    index=np.arange(n)+k;labelled=index<n;take=np.minimum(index,n-1)
                    active=alive&labelled
                    if mode=='recorded_reference':r=e['refs'][take]
                    else:
                        regenerated=reference_step(q,dq,e['goals'][take],h,e['limits'],dt=DT,
                            active_mask=active,apply_mask=active)
                        failed=active&(~regenerated['feasible']|~regenerated['applied'])
                        failures.extend(dict(anchor=int(a),horizon_tick=k+1,
                            code=int(regenerated['failure_code'][a])) for a in np.flatnonzero(failed))
                        alive&=~failed;active&=~failed;h=regenerated['next_h']
                        r=np.c_[regenerated['q_ref'],regenerated['dq_ref'],regenerated['ddq_ref']]
                    pred=model.predict_joint(q,dq,r[:,:7],r[:,7:14],r[:,14:])
                    nq,ndq=pred['q'],pred['dq']
                    failed=active&(~np.isfinite(nq).all(-1)|~np.isfinite(ndq).all(-1)
                        |np.any((nq<e['limits'].lower)|(nq>e['limits'].upper),axis=-1))
                    failures.extend(dict(anchor=int(a),horizon_tick=k+1,code='nonfinite_or_joint_bounds') for a in np.flatnonzero(failed))
                    alive&=~failed;active&=~failed
                    errors=_errors(nq,ndq,e['q'][take+1],e['dq'][take+1],e['tcp'][take+1],fk)
                    pure=np.asarray([bool(labelled[a] and e['probe_mask'][a:index[a]+1].all()) for a in range(n)])
                    data.append(dict(labelled=labelled.copy(),available=active.copy(),pure_probe_window=pure,errors=errors))
                    q=np.where(active[:,None],nq,q);dq=np.where(active[:,None],ndq,dq)
                rollouts[mode].append((e,data,failures))
        role_result=dict(parents=sorted({e['parent_id'] for e in subset}),episodes=len(subset),
            transitions=sum(len(e['refs']) for e in subset),probe_ticks=sum(int(e['probe_mask'].sum()) for e in subset),
            conditional_H1={},rollouts={})
        for group in ('all','probe'):
            metrics={}
            for e,errors in conditional:
                mask=np.ones(len(e['refs']),bool) if group=='all' else e['probe_mask']
                for name,value in errors.items():metrics.setdefault(name,[]).extend(value[mask].tolist())
            role_result['conditional_H1'][group]={name:stats(values) for name,values in metrics.items()}
        for mode,values in rollouts.items():
            by_horizon={}
            for k in range(16):
                groups={}
                for group in ('all','probe_anchor','pure_probe_window'):
                    labelled=available=0;metrics={}
                    for e,data,failures in values:
                        row=data[k]
                        mask=(np.ones(len(e['refs']),bool) if group=='all' else
                              e['probe_mask'] if group=='probe_anchor' else row['pure_probe_window'])
                        labelled+=int((mask&row['labelled']).sum());ok=mask&row['available'];available+=int(ok.sum())
                        for name,error in row['errors'].items():metrics.setdefault(name,[]).extend(error[ok].tolist())
                    groups[group]=dict(labelled=labelled,predictable=available,
                        errors={name:stats(v) for name,v in metrics.items()})
                by_horizon[str(k+1)]=dict(seconds=(k+1)*DT,cohorts=groups)
            role_result['rollouts'][mode]=dict(by_horizon=by_horizon,
                infeasible=[dict(episode_id=e['episode_id'],parent_id=e['parent_id'],**failure)
                            for e,data,failures in values for failure in failures],
                missing_labels_or_failed_predictions_imputed=False)
        result['roles'][role]=role_result
    return result
