"""CPU fitting of executed R10 native-continuation decisions.

TRAIN parent sources define classifier folds, normalisation, positive Bayesian
bootstrap members and support. VAL only calibrates probabilities and can turn
ranking off. TEST, unexecuted candidates and early skill labels cannot be fit
labels. No simulator or GPU APIs are imported.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.special import expit, logit

from .candidate_success import (CONTINUATION_POLICY, FEATURE_NAMES,
    FEATURE_SCHEMA_SHA256, LABEL_SCOPE, SCHEMA)

MANIFEST_SCHEMA = 'r10_executed_continuation_manifest_v1'
CALIBRATION_LAMBDAS = (.01, .1, 1.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parent_group(row):
    if row.get('split') not in ('train','validation') or not row.get('task') or not row.get('raw_episode_id'):
        raise ValueError('native continuation source requires TRAIN/VAL task and raw episode identity')
    group = ':'.join((row['split'],row['task'],row.get('parent_raw_episode_id') or row['raw_episode_id']))
    if row.get('parent_source_group',group) != group:
        raise ValueError('augmentation parent identity differs from canonical raw source group')
    return group


def load_executed_manifest(path, *, split, task_name, predictor,
                           continuation_sha256, library_sha256, project_root=None):
    """Read frozen actual execution labels and preserve zero-candidate cases.

    entries need split/task/predictor/raw_episode_id/path/sha256; parent raw ID
    is explicit for native-generated replicas. case JSON owns the complete
    ContinuationAttempts rows under continuation_attempts, the independent
    official_task_success, complete and downstream_selection_policy fields.
    Paths are relative to project_root when supplied, else manifest directory.
    """
    path = Path(path)
    manifest = json.loads(path.read_text())
    entries = manifest.get('entries',[])
    if (manifest.get('schema') != MANIFEST_SCHEMA or not entries
            or any(row.get('split') != split for row in entries)
            or split not in ('train','validation')):
        raise ValueError('critic manifest must contain the requested TRAIN/VAL split only')
    rows, selected, receipts, excluded = [], [], [], []
    root = Path(project_root) if project_root else path.parent
    identities = set()
    for entry in entries:
        parent_group(entry)
        if entry.get('task') != task_name or entry.get('predictor') != predictor:
            continue
        identity = (entry['task'],entry['predictor'],entry['raw_episode_id'])
        if identity in identities:
            raise ValueError('duplicate critic rollout identity')
        identities.add(identity)
        source_path = Path(entry['path'])
        source_path = source_path if source_path.is_absolute() else root/source_path
        if sha(source_path) != entry['sha256']:
            raise ValueError('executed continuation case changed')
        case = json.loads(source_path.read_text())
        for key in ('task','predictor','split','raw_episode_id'):
            if case.get(key) != entry[key]:
                raise ValueError('case and manifest identity disagree: '+key)
        if ((case.get('parent_raw_episode_id') or case['raw_episode_id'])
                != (entry.get('parent_raw_episode_id') or entry['raw_episode_id'])):
            raise ValueError('case and manifest augmentation parent disagree')
        if (case.get('continuation_sha256') != continuation_sha256
                or case.get('interaction_library_sha256') != library_sha256
                or case.get('continuation_selection_policy') != CONTINUATION_POLICY
                or case.get('downstream_selection_policy') != 'geometry'
                or case.get('unexecuted_candidates_labelled') is not False
                or case.get('first_selection_policy') not in ('geometry','TRAIN_explore_once')):
            raise ValueError('critic labels do not come from the frozen geometry downstream')
        if split != 'train' and case['first_selection_policy'] == 'TRAIN_explore_once':
            raise ValueError('TRAIN candidate exploration cannot label VAL')
        selected.append(dict(entry,path=str(source_path.resolve())))
        attempts = case.get('continuation_attempts',[])
        if not isinstance(attempts,list):
            raise ValueError('native continuation attempts must be a ledger list')
        if (case.get('complete') is not True or case.get('infrastructure_error')
                or type(case.get('official_task_success')) is not bool):
            excluded.append(dict(raw_episode_id=entry['raw_episode_id'],reason='incomplete_or_infrastructure_censored',
                                 attempts=len(attempts), counted_as_policy_negative=False))
            continue
        group = parent_group(entry)
        for index,attempt in enumerate(attempts):
            if (attempt.get('task') != task_name or attempt.get('predictor') != predictor
                    or attempt.get('feature_schema') != SCHEMA
                    or attempt.get('feature_schema_sha256') != FEATURE_SCHEMA_SHA256
                    or attempt.get('label_scope') != LABEL_SCOPE
                    or attempt.get('continuation_sha256') != continuation_sha256
                    or attempt.get('interaction_library_sha256') != library_sha256
                    or attempt.get('continuation_selection_policy') != CONTINUATION_POLICY
                    or attempt.get('episode_first_actionable_decision') is not (index == 0)
                    or attempt.get('decision_id') != index):
                raise ValueError('attempt is not an actual matching R10 continuation decision')
            if attempt.get('complete') is not True or attempt.get('censored') is not False or type(attempt.get('official_task_success')) is not bool:
                raise ValueError('completed native cases cannot contain fabricated/unresolved continuation labels')
            abandoned = attempt.get('skill_reason') == 'candidate_abandoned_before_skill_completion'
            if attempt['official_task_success'] != case['official_task_success'] and not (abandoned and attempt['official_task_success'] is False):
                raise ValueError('early skill completion cannot replace the independent native task outcome')
            features = np.asarray(attempt['features'],float)
            if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
                raise ValueError('critic schema is not the observed R10 feature vector')
            rows.append(dict(x=features,y=int(attempt['official_task_success']),group=group,
                phase=attempt['phase'],raw_episode_id=entry['raw_episode_id'],decision_id=index,
                first_actionable=index == 0))
        receipts.append(dict(raw_episode_id=entry['raw_episode_id'],parent_source_group=group,
            attempts=len(attempts),native_success=case['official_task_success']))
    if not selected:
        raise ValueError('requested task/predictor has no frozen execution cases')
    return rows,dict(schema=MANIFEST_SCHEMA,entries=selected),dict(
        source_manifest_path=str(path.resolve()),source_manifest_sha256=sha(path),
        selected_rollouts=len(selected),completed_native_rollouts=len(receipts),censored_rollouts=excluded,
        zero_candidate_rollouts=[row for row in receipts if row['attempts'] == 0],
        completed_parent_sources=sorted({row['parent_source_group'] for row in receipts}),
        actual_labelled_decisions=len(rows),unexecuted_candidates_labelled=False)


def parent_weights(groups):
    groups = np.asarray(groups,str)
    names,counts = np.unique(groups,return_counts=True)
    lookup = dict(zip(names,counts))
    return np.array([1./lookup[group] for group in groups])


def grouped_folds(y, groups, count=2):
    y,groups = np.asarray(y,int),np.asarray(groups,str)
    names = np.unique(groups)
    if len(names) < 2:
        return None
    count = min(int(count),len(names))
    if count < 2:
        return None
    summary = [(name,int(y[groups == name].sum()),int((groups == name).sum())) for name in names]
    summary.sort(key=lambda row:(-int(row[1] > 0),-row[1],-row[2],row[0]))
    positive,negative,size = np.zeros(count),np.zeros(count),np.zeros(count)
    assignment = {}
    for index,(name,pos,n) in enumerate(summary):
        if index < count:
            fold = index
        else:
            fold = min(range(count),key=lambda k:(positive[k] if pos else negative[k],size[k],k))
        assignment[name] = fold
        positive[fold] += pos;negative[fold] += n-pos;size[fold] += n
    return np.array([assignment[name] for name in groups]),assignment


def metrics(y, p):
    y,p = np.asarray(y,int),np.asarray(p,float)
    if (y.ndim != 1 or y.shape != p.shape or not len(y) or not np.isin(y,(0,1)).all()
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
        raise ValueError('invalid native binary probability evaluation')
    clipped = np.clip(p,1e-7,1-1e-7)
    ece = 0.
    for index in range(10):
        selected = (p >= index/10) & ((p < (index+1)/10) if index < 9 else (p <= 1))
        if selected.any():
            ece += selected.mean()*abs(y[selected].mean()-p[selected].mean())
    positives,negatives = p[y == 1],p[y == 0]
    auc = None
    if len(positives) and len(negatives):
        ordered = np.sort(negatives)
        auc = float(np.mean((np.searchsorted(ordered,positives,side='left')+
                            np.searchsorted(ordered,positives,side='right'))/(2*len(negatives))))
    return dict(samples=len(y),positives=int(y.sum()),positive_rate=float(y.mean()),
        NLL=float(np.mean(-y*np.log(clipped)-(1-y)*np.log(1-clipped))),
        Brier=float(np.square(y-p).mean()),ECE=float(ece),AUROC=auc)


def _normalise(x, groups):
    weight = parent_weights(groups)
    mean = np.average(x,axis=0,weights=weight)
    scale = np.sqrt(np.average(np.square(x-mean),axis=0,weights=weight))
    scale = np.where(scale < 1e-6,1.,scale)
    return mean,scale


def _fit_linear(z, y, groups, regularization, multipliers=None):
    weight = parent_weights(groups)
    if multipliers is not None:
        weight *= np.array([multipliers[name] for name in groups])
    if np.any(weight <= 0) or not np.isfinite(weight).all():
        raise ValueError('every TRAIN parent source must have strictly positive bootstrap weight')
    weight *= len(np.unique(groups))/weight.sum()
    prior = float((np.dot(weight,y)+.5)/(weight.sum()+1.))
    if len(np.unique(y)) < 2:
        return np.zeros(z.shape[1]),float(logit(prior)),dict(constant_only=True,TRAIN_prior=prior)
    initial = np.r_[np.zeros(z.shape[1]),logit(prior)]
    def objective(theta):
        score = z@theta[:-1]+theta[-1]
        loss = np.dot(weight,np.logaddexp(0.,score)-y*score)/weight.sum()
        loss += regularization*np.dot(theta[:-1],theta[:-1])/2
        residual = weight*(expit(score)-y)/weight.sum()
        gradient = np.r_[z.T@residual+regularization*theta[:-1],residual.sum()]
        return float(loss),gradient
    fit = minimize(objective,initial,jac=True,method='L-BFGS-B',options=dict(maxiter=500,gtol=1e-7,ftol=1e-10))
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError('TRAIN linear critic optimization failed: '+str(fit.message))
    return fit.x[:-1],float(fit.x[-1]),dict(constant_only=False,TRAIN_prior=prior,iterations=int(fit.nit))


def _identity():
    return dict(slope=1.,intercept=0.,fit_split='identity',method='identity',regularization_lambda=None)


def _calibrate(p, correction):
    return expit(correction['slope']*logit(np.clip(p,1e-7,1-1e-7))+correction['intercept'])


def _affine(p, y, regularization):
    logits = logit(np.clip(p,1e-7,1-1e-7))
    def objective(theta):
        s = theta[0]*logits+theta[1]
        value = np.mean(np.logaddexp(0.,s)-y*s)+regularization*((theta[0]-1.)**2+theta[1]**2)
        r = expit(s)-y
        gradient = np.array([np.mean(r*logits)+2*regularization*(theta[0]-1.),
                             np.mean(r)+2*regularization*theta[1]])
        return float(value),gradient
    fit = minimize(objective,[1.,0.],jac=True,method='L-BFGS-B',bounds=[(.25,4.),(-6.,6.)],
                   options=dict(maxiter=300,gtol=1e-8,ftol=1e-12))
    if not fit.success:
        raise ValueError('positive affine calibration optimization failed')
    return dict(slope=float(fit.x[0]),intercept=float(fit.x[1]),fit_split='validation',
                method='strongly_shrunk_positive_affine',regularization_lambda=float(regularization))


def _can_fit_affine(y, groups):
    return all(len(set(np.asarray(groups)[np.asarray(y) == label])) >= 2 for label in (0,1))


def select_calibration(p, y, groups):
    """Inner source CV chooses identity or shrinkage, without an outer-held row."""
    folds = grouped_folds(y,groups,2)
    if folds is None or not _can_fit_affine(y,groups):
        return _identity(),dict(selected='identity',reason='insufficient positive/negative independent VAL parent sources',
            fit_sources=sorted(set(groups)),candidates=[])
    assignment,source_to_fold = folds
    trials = [dict(method='identity',regularization_lambda=None,Brier=metrics(y,p)['Brier'])]
    for regularization in CALIBRATION_LAMBDAS:
        held_predictions = np.empty(len(y))
        valid = True
        for fold in (0,1):
            fit,held = assignment != fold,assignment == fold
            if len(np.unique(y[fit])) < 2:
                valid = False
                break
            held_predictions[held] = _calibrate(p[held],_affine(p[fit],y[fit],regularization))
        trials.append(dict(method='strongly_shrunk_positive_affine',regularization_lambda=regularization,
                           Brier=metrics(y,held_predictions)['Brier'] if valid else None,valid=valid))
    best = min((row for row in trials if row.get('valid',True)),key=lambda row:
               (round(row['Brier'],12),row['method'] != 'identity',-(row['regularization_lambda'] or 0)))
    correction = (_identity() if best['method'] == 'identity' else _affine(p,y,best['regularization_lambda']))
    return correction,dict(selected=best['method'],selected_lambda=best['regularization_lambda'],
        source_to_fold=source_to_fold,fit_sources=sorted(set(groups)),candidates=trials)


def nested_calibration(p, y, groups):
    p,y,groups = np.asarray(p,float),np.asarray(y,int),np.asarray(groups,str)
    if p.shape != y.shape or y.shape != groups.shape or y.ndim != 1:
        raise ValueError('nested calibration requires aligned probability/outcome/parent arrays')
    if not len(y):
        return _identity(),dict(valid=False,reason='no labelled VAL native continuations',metrics=None,
                                raw_metrics=None,folds=[],final_selection=None)
    metrics(y,p)
    full,selection = select_calibration(p,y,groups)
    folds = grouped_folds(y,groups,2)
    if folds is None:
        # Identity uses no VAL labels; it remains honest without an invented fit
        # split, but there is insufficient evidence to enable learned ranking.
        return _identity(),dict(valid=False,reason='fewer than two independent VAL parent sources',
            metrics=metrics(y,p),raw_metrics=metrics(y,p),folds=[],final_selection=selection)
    assignment,mapping = folds
    cross = np.empty(len(y));traces=[]
    for fold in (0,1):
        fit,held = assignment != fold,assignment == fold
        correction,trace = select_calibration(p[fit],y[fit],groups[fit])
        cross[held] = _calibrate(p[held],correction)
        traces.append(dict(fold=fold,fit_sources=sorted(set(groups[fit])),held_sources=sorted(set(groups[held])),
            correction=correction,inner_selection=trace,held_metrics=metrics(y[held],cross[held])))
    return full,dict(valid=True,metrics=metrics(y,cross),raw_metrics=metrics(y,p),
        source_to_outer_fold=mapping,folds=traces,final_selection=selection,
        full_fit_metrics=metrics(y,_calibrate(p,full)),fit_target='attempt-uniform within task/predictor/phase',
        calibration_selector_used_VAL=True,outer_held_labels_used_to_select_own_correction=False)


@dataclass(frozen=True)
class CriticFitConfig:
    regularization_grid: tuple[float,...] = (.01,.1,1.)
    bootstrap_members: int = 5
    TRAIN_folds: int = 5
    seed: int = 10
    support_quantile: float = .99
    maximum_crossfit_ECE: float = .15
    maximum_Brier_regression: float = .01

    def __post_init__(self):
        if (self.bootstrap_members != 5 or self.TRAIN_folds < 2 or not self.regularization_grid
                or min(self.regularization_grid) <= 0
                or not np.isfinite([*self.regularization_grid,self.support_quantile,
                    self.maximum_crossfit_ECE,self.maximum_Brier_regression]).all()
                or not 0 < self.support_quantile <= 1 or self.maximum_crossfit_ECE < 0
                or self.maximum_Brier_regression < 0):
            raise ValueError('invalid frozen critic fit/calibration gate configuration')


def fit_phase_head(train, validation, config, *, seed_offset=0):
    """Fit one task/predictor/phase head; all supplied TRAIN parents participate."""
    x = np.stack([row['x'] for row in train]);y=np.array([row['y'] for row in train],int)
    groups=np.array([row['group'] for row in train],str)
    if (x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all()
            or not np.isin(y,(0,1)).all() or len(set(groups)) < 2
            or any(not group.startswith('train:') for group in groups)
            or len({row['phase'] for row in train}) != 1
            or any(row['phase'] != train[0]['phase'] or not row['group'].startswith('validation:') for row in validation)):
        raise ValueError('each critic phase needs executed data from at least two TRAIN parent sources')
    fold_result = grouped_folds(y,groups,config.TRAIN_folds)
    folds,source_to_fold=fold_result
    trials=[];oof_by_lambda={};prior=np.empty(len(y))
    for regularization in config.regularization_grid:
        oof=np.empty(len(y))
        for fold in sorted(set(folds)):
            fit,held=folds != fold,folds == fold
            mean,scale=_normalise(x[fit],groups[fit])
            weights,bias,info=_fit_linear((x[fit]-mean)/scale,y[fit],groups[fit],regularization)
            oof[held]=expit(((x[held]-mean)/scale)@weights+bias)
            prior[held]=info['TRAIN_prior']
        trials.append(dict(regularization=float(regularization),**metrics(y,oof)))
        oof_by_lambda[regularization]=oof
    selected=min(trials,key=lambda row:(row['NLL'],-row['regularization']))
    oof=oof_by_lambda[selected['regularization']];oof_metrics=metrics(y,oof);prior_metrics=metrics(y,prior)
    train_ranking=bool(_can_fit_affine(y,groups) and oof_metrics['AUROC'] is not None
        and oof_metrics['AUROC'] > .5 and oof_metrics['NLL'] < prior_metrics['NLL']-1e-6
        and oof_metrics['Brier'] < prior_metrics['Brier']-1e-6)
    mean,scale=_normalise(x,groups);z=(x-mean)/scale
    rng=np.random.default_rng(config.seed+seed_offset)
    names=sorted(set(groups));members=[];biases=[];bootstrap=[]
    for member in range(config.bootstrap_members):
        positive=np.maximum(rng.exponential(1.,len(names)),np.finfo(float).tiny)
        multipliers=dict(zip(names,positive))
        weights,bias,info=_fit_linear(z,y,groups,selected['regularization'],multipliers)
        members.append(weights);biases.append(bias)
        bootstrap.append(dict(member=member,parent_source_weights={name:float(multipliers[name]) for name in names},
                              all_TRAIN_parent_sources_have_positive_weight=True,fit=info))
    members,biases=np.stack(members),np.array(biases)
    distances=np.empty(len(y))
    for name in names:
        selected_rows=groups == name
        distances[selected_rows]=cKDTree(z[~selected_rows]).query(z[selected_rows])[0]
    order=np.argsort(distances);w=parent_weights(groups)[order];cumulative=np.cumsum(w)/w.sum()
    support_limit=max(1e-6,float(np.interp(config.support_quantile,cumulative,distances[order])))
    val_y=np.array([row['y'] for row in validation],int)
    val_groups=np.array([row['group'] for row in validation],str)
    if len(validation):
        val_x=np.stack([row['x'] for row in validation]);val_z=(val_x-mean)/scale
        raw=expit(val_z@members.T+biases).mean(1)
    else:
        raw=np.empty(0)
    calibration,crossfit=nested_calibration(raw,val_y,val_groups)
    cal_metrics=crossfit['metrics'];raw_metrics=crossfit['raw_metrics']
    gate=bool(crossfit['valid'] and cal_metrics and
        all(np.isfinite(cal_metrics[name]) for name in ('ECE','Brier','NLL')) and
        cal_metrics['ECE'] <= config.maximum_crossfit_ECE and
        cal_metrics['Brier'] <= raw_metrics['Brier']+config.maximum_Brier_regression)
    head=dict(TRAIN_mean=mean.tolist(),TRAIN_scale=scale.tolist(),ensemble_weights=members.tolist(),
        ensemble_bias=biases.tolist(),standardized_TRAIN_support=z.tolist(),support_max_distance=support_limit,
        support_threshold_fit_split='train',support_threshold_excludes_same_parent_source=True,
        support_parent_sources=groups.tolist(),source_coverage_complete=True,
        calibration=calibration,calibration_crossfit_gate_passed=gate,
        ranking_enabled=train_ranking and gate,TRAIN_ranking_guard_passed=train_ranking,
        ranking_fallback_reason=None if train_ranking and gate else
            ('TRAIN_source_CV_has_no_ranking_gain' if not train_ranking else 'VAL_crossfit_calibration_gate_failed'))
    report=dict(TRAIN_samples=len(train),TRAIN_positive_labels=int(y.sum()),TRAIN_parent_sources=len(names),
        VAL_samples=len(validation),VAL_parent_sources=len(set(val_groups)),
        classifier_selection_split='train',normalization_fit_split='train',
        TRAIN_grouping='split,task,parent_raw_episode_id_or_raw_episode_id',
        TRAIN_fit_weighting='equal parent source; uniform executed attempts within parent',
        TRAIN_selection=dict(candidates=trials,selected_regularization=selected['regularization'],
            source_to_fold=source_to_fold,OOF_metrics=oof_metrics,OOF_fit_constant_prior_metrics=prior_metrics),
        bootstrap=bootstrap,all_eligible_TRAIN_parent_sources_used=True,
        TRAIN_ranking_guard_passed=train_ranking,raw_metrics=raw_metrics,calibration=calibration,
        calibration_crossfit=crossfit,calibration_gate_passed=gate,ranking_enabled=head['ranking_enabled'],
        ranking_fallback_reason=head['ranking_fallback_reason'],
        interpretation='probability validation and ranking guard; no task success-rate gain is inferred')
    return head,report
