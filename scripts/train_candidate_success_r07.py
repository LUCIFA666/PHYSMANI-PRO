"""Fit versioned task critics using grouped TRAIN CV and author-GL VAL calibration.

Execute through scripts/run_local_sim.sh for the existing native sklearn runtime.
All sensor replicas of one raw source stay in the same model-selection fold.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import sklearn
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from our_mani.risk_v2.selector import attempt_controller_hash
from our_mani.risk_v3.model import (TASKS, FORMAT, FEATURE_SCHEMA, FEATURE_SCHEMA_SHA256,
    transform_features, fit_affine, calibrate, build_task_support)

CANDIDATES = (dict(kind='logreg', C=.1), dict(kind='logreg', C=1.),
    dict(kind='hist_gradient_boosting', max_leaf_nodes=7),
    dict(kind='hist_gradient_boosting', max_leaf_nodes=15))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def metrics(probability, labels):
    p, y = np.asarray(probability, float), np.asarray(labels, float)
    if not len(y):
        return dict(n=0, positive=0, brier=None, log_loss=None, auroc=None,
            ece_5_bins=None, calibration_bins=[], mean_probability=None, prevalence=None)
    if not np.isfinite(p).all():
        raise ValueError('nonfinite diagnostic probability')
    positive, negative = int(y.sum()), int((1-y).sum())
    clipped = np.clip(p, 1e-6, 1-1e-6)
    bins = []; ece = 0.
    for i in range(5):
        chosen = (p >= i/5) & (p < (i+1)/5 if i < 4 else p <= 1.)
        if chosen.any():
            predicted, actual = float(p[chosen].mean()), float(y[chosen].mean())
            ece += float(chosen.mean()) * abs(predicted-actual)
            bins.append(dict(low=i/5, high=(i+1)/5, n=int(chosen.sum()), probability=predicted, actual=actual))
    return dict(n=len(y), positive=positive, brier=float(np.mean((p-y)**2)),
        log_loss=float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped))),
        auroc=float((rankdata(p)[y==1].sum()-positive*(positive+1)/2)/(positive*negative)) if positive and negative else None,
        ece_5_bins=ece, calibration_bins=bins, mean_probability=float(p.mean()), prevalence=float(y.mean()))


def balanced_group_folds(rows, n_folds, seed=810):
    """Deterministic source folds balancing positives, attempts, then source count."""
    sources = sorted({r['source_episode'] for r in rows})
    if len(sources) < n_folds:
        raise ValueError('too few raw sources for grouped folds')
    groups = []
    for source in sources:
        local = [r for r in rows if r['source_episode'] == source]
        groups.append(dict(source=source, positive=sum(r['official_task_success'] for r in local), attempts=len(local)))
    random = np.random.default_rng(seed)
    groups = [groups[i] for i in random.permutation(len(groups))]
    groups.sort(key=lambda g: (-g['positive'], -g['attempts']))
    capacity = [len(groups)//n_folds + (i < len(groups)%n_folds) for i in range(n_folds)]
    totals = [dict(sources=0, attempts=0, positive=0) for _ in range(n_folds)]
    assignment = {}
    for group in groups:
        allowed = [i for i in range(n_folds) if totals[i]['sources'] < capacity[i]]
        if group['positive']:
            key = lambda i: (totals[i]['positive'], totals[i]['attempts'], totals[i]['sources'], i)
        else:
            key = lambda i: (totals[i]['attempts'], totals[i]['sources'], totals[i]['positive'], i)
        selected = min(allowed, key=key)
        assignment[group['source']] = selected
        for key in ('attempts', 'positive'): totals[selected][key] += group[key]
        totals[selected]['sources'] += 1
    folds = np.array([assignment[r['source_episode']] for r in rows], dtype=int)
    report = dict(seed=seed, source_to_fold=assignment, folds=totals,
        positive_source_count=sum(g['positive'] > 0 for g in groups),
        negative_source_count=sum(g['attempts'] > g['positive'] for g in groups),
        grouping='task + raw source, across every renderer and replica',
        assignment_rule='greedy positives/attempts/source-count balance with deterministic shuffled ties')
    return folds, report


def bayesian_source_weights(rows, all_sources, seed):
    random = np.random.default_rng(seed)
    sources = sorted(all_sources)
    draws = np.maximum(random.exponential(size=len(sources)), np.finfo(float).tiny)
    source_weights = dict(zip(sources, map(float, draws)))
    weights = np.array([source_weights[r['source_episode']] for r in rows])
    weights *= len(weights)/weights.sum()
    return weights, dict(seed=seed, source_weights=source_weights,
        all_source_weights_strictly_positive=bool(np.all(draws > 0)),
        target='Bayesian source multiplier on a task-uniform attempt loss; all observed attempts retained')


def make_model(candidate, seed):
    if candidate['kind'] == 'logreg':
        return make_pipeline(StandardScaler(), LogisticRegression(C=candidate['C'],
            max_iter=1500, solver='lbfgs', random_state=seed))
    return HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=candidate['max_leaf_nodes'],
        min_samples_leaf=40, learning_rate=.05, l2_regularization=10.,
        early_stopping=False, random_state=seed)


def fit_model(model, candidate, x, y, weights=None):
    if len(np.unique(y)) != 2:
        raise ValueError('candidate critic fitting requires positive and negative TRAIN attempts')
    if candidate['kind'] == 'logreg' and weights is not None:
        model.fit(x, y, logisticregression__sample_weight=weights)
    else:
        model.fit(x, y, **({'sample_weight': weights} if weights is not None else {}))
    return model


def select_model(x, y, rows, seed):
    folds, fold_report = balanced_group_folds(rows, 5, seed)
    prior = np.zeros(len(y))
    for fold in range(5):
        held = folds == fold
        prior[held] = y[~held].mean()
    prior_metrics = metrics(prior, y)
    candidates = []
    for number, candidate in enumerate(CANDIDATES):
        probabilities = np.zeros(len(y)); per_fold = []
        for fold in range(5):
            held = folds == fold
            model = fit_model(make_model(candidate, seed+fold), candidate, x[~held], y[~held])
            probabilities[held] = model.predict_proba(x[held])[:, 1]
            per_fold.append(metrics(probabilities[held], y[held]))
        candidates.append(dict(candidate=candidate, out_of_source_TRAIN=metrics(probabilities, y), folds=per_fold))
    selected = min(range(len(candidates)), key=lambda i: (candidates[i]['out_of_source_TRAIN']['log_loss'],
        candidates[i]['out_of_source_TRAIN']['brier'], i))
    chosen_metrics = candidates[selected]['out_of_source_TRAIN']
    rank_enabled = (chosen_metrics['log_loss'] < prior_metrics['log_loss'] - 1e-6
        and chosen_metrics['brier'] < prior_metrics['brier'] - 1e-6
        and chosen_metrics['auroc'] is not None and chosen_metrics['auroc'] >= .55
        and fold_report['positive_source_count'] >= 5 and fold_report['negative_source_count'] >= 5)
    return CANDIDATES[selected], dict(selected=CANDIDATES[selected], candidates=candidates, source_folds=fold_report,
        grouped_TRAIN_task_constant_prior=prior_metrics, ranking_enabled=bool(rank_enabled),
        ranking_guard='OOF NLL and Brier each beat fold-TRAIN task prior by >1e-6; OOF AUC>=.55; positive and negative raw sources each>=5',
        selection='lowest pooled attempt-uniform grouped TRAIN out-of-fold NLL; Brier then fixed order break ties',
        VAL_or_CAL_used_for_algorithm_or_hyperparameters=False)


def crossfit_calibration(raw, y, rows, seed):
    folds, grouping = balanced_group_folds(rows, 2, seed)
    probabilities = np.full(len(y), np.nan); priors = np.full(len(y), np.nan); details = []; reasons = []
    if grouping['positive_source_count'] < 2:
        reasons.append('fewer than two positive raw sources; two independent positive calibration folds unavailable')
    if grouping['negative_source_count'] < 2:
        reasons.append('fewer than two negative raw sources; two independent negative calibration folds unavailable')
    for fold in range(2):
        held = folds == fold; fit = ~held
        prior = float(y[fit].mean()); priors[held] = prior
        detail = dict(held_fold=fold, fit_sources=sorted({r['source_episode'] for r, h in zip(rows, held) if not h}),
            held_sources=sorted({r['source_episode'] for r, h in zip(rows, held) if h}),
            fit_positive=int(y[fit].sum()), held_positive=int(y[held].sum()),
            fit_attempts=int(fit.sum()), held_attempts=int(held.sum()), constant_prior=prior)
        if len(np.unique(y[fit])) < 2:
            reasons.append(f'fold {fold} affine fit has a single outcome class; no valid crossfit correction')
            detail['valid'] = False
        else:
            correction = fit_affine(raw[fit], y[fit]); probabilities[held] = calibrate(raw[held], correction)
            detail.update(valid=True, correction=correction)
        details.append(detail)
    valid = not reasons and np.isfinite(probabilities).all()
    return dict(valid=bool(valid), reasons=reasons, grouping=grouping, folds=details,
        metrics=metrics(probabilities, y) if valid else None,
        raw_metrics=metrics(raw, y), task_constant_prior=metrics(priors, y),
        interpretation='source-stratified two-fold calibration crossfit; full final affine uses all author-GL VAL sources')


def load_domain(base, predictor, split, replicas, domain, expected, controller):
    rolls = []; samples = []; files = {}; seen = set()
    directories = sorted(base.glob(f'collection_{predictor}_{split}_*'))
    if not directories: raise ValueError('no collection directories: ' + str(base))
    for directory in directories:
        report = json.loads((directory/'report.json').read_text())
        if not report['complete']: raise ValueError('candidate collection is incomplete: ' + str(directory))
        for path in sorted(directory.glob('pick*__episode*__*.json')):
            if path.name.endswith('infrastructure_error.json'):
                continue
            row = json.loads(path.read_text()); group = (row['task'], row['source_episode'])
            key = (*group, row['replica'])
            if key in seen: raise ValueError('duplicate raw-source rollout within sensor domain')
            if group not in expected or row['split'] != split or row['formal_test_used']:
                raise ValueError('unrecognized source, wrong split, or TEST used by critic loader')
            if row.get('attempt_controller_sha256') != controller:
                raise ValueError('continuation controller differs across candidate data')
            observed_domain = row.get('sensor_domain', 'OPENGL3' if domain == 'OPENGL3' else None)
            if observed_domain != domain: raise ValueError('candidate sensor domain differs from declared collection')
            seen.add(key); rolls.append(row); files[str(path.relative_to(ROOT))] = sha(path)
            for attempt in row['attempts']:
                if not attempt.get('complete'): raise ValueError('censored candidate outcome cannot be fitted')
                samples.append(dict(attempt, task=row['task'], source_episode=row['source_episode'], sensor_domain=domain))
    if seen != {(*g, replica) for g in expected for replica in range(replicas)}:
        raise ValueError(f'{domain} {split}: every expected raw source/replica must execute')
    return samples, rolls, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictor', choices=('cv', 'context'), required=True)
    parser.add_argument('--config', default='config/framework/aligned_protocol_r07.json')
    parser.add_argument('--out')
    args = parser.parse_args()
    config = json.loads((ROOT/args.config).read_text()); controller = attempt_controller_hash(ROOT, config, args.predictor)
    expected = {}
    for split in ('train', 'validation'):
        manifest = json.loads((ROOT/f'artifacts/framework/protocol/{split}_manifest.json').read_text())
        expected[split] = {(r['task'], r['raw_episode_id']) for r in manifest['episodes'] if r['task'] in TASKS}
    if expected['train'] & expected['validation']: raise ValueError('TRAIN/VAL raw sources overlap')
    out = ROOT/(args.out or f'artifacts/framework/aligned_r07/{args.predictor}_success')
    if any((out/name).exists() for name in ('value.pt', 'value_support.json', 'report.json')):
        raise FileExistsError('critic artifacts are immutable; choose a new version')
    legacy, legacy_rolls, legacy_files = load_domain(ROOT/'artifacts/framework/aligned_r06', args.predictor,
        'train', 3, 'OPENGL3', expected['train'], controller)
    author, author_rolls, author_files = load_domain(ROOT/'artifacts/framework/aligned_r07', args.predictor,
        'train', 1, 'author_OPENGL', expected['train'], controller)
    val, val_rolls, val_files = load_domain(ROOT/'artifacts/framework/aligned_r07', args.predictor,
        'validation', 3, 'author_OPENGL', expected['validation'], controller)
    train = legacy + author; raw_dim = 87 if args.predictor == 'context' else 71
    encoder_hashes = {r.get('context_encoder_sha256') for r in train + val}
    if args.predictor == 'context' and (len(encoder_hashes) != 1 or None in encoder_hashes):
        raise ValueError('Context samples must use one frozen causal history encoder')
    source_files = [Path(__file__), ROOT/'our_mani/risk_v3/model.py', ROOT/'our_mani/risk_v3/selector.py']
    if not all(path.exists() for path in source_files): raise ValueError('versioned critic selector is required before training')
    selection_files = {str(path.relative_to(ROOT)): sha(path) for path in source_files}
    selection_hash = hashlib.sha256(json.dumps(dict(files=selection_files, candidates=CANDIDATES,
        folds=5, ensemble='5 positive Bayesian source multipliers', loss='within-task attempt uniform'), sort_keys=True).encode()).hexdigest()
    heads = {}; supports = {}; selections = {}; coverage = {}; audits = {}; crossfits = {}; metrics_raw = {}; metrics_cal = {}; priors = {}
    for task_index, task in enumerate(TASKS):
        rows = [r for r in train if r['task'] == task]; calibration_rows = [r for r in val if r['task'] == task]
        all_sources = {source for t, source in expected['train'] if t == task}
        feature_sources = {r['source_episode'] for r in rows}
        if not rows or not calibration_rows: raise ValueError('no candidate features for one task/split')
        raw_x = np.asarray([r['features'] for r in rows], float)
        raw_val = np.asarray([r['features'] for r in calibration_rows], float)
        x, xv = transform_features(raw_x, raw_dim), transform_features(raw_val, raw_dim)
        y = np.array([r['official_task_success'] for r in rows], int)
        yv = np.array([r['official_task_success'] for r in calibration_rows], int)
        seed = 810 + task_index*100 + (1000 if args.predictor == 'context' else 0)
        candidate, selection = select_model(x, y, rows, seed)
        selections[task] = selection; models = []; weights_audit = []; members = []
        for member in range(5):
            weights, audit = bayesian_source_weights(rows, all_sources, seed + 10000 + member)
            model = fit_model(make_model(candidate, seed + member), candidate, x, y, weights)
            models.append(model); weights_audit.append(audit); members.append(model.predict_proba(xv)[:, 1])
        raw = np.stack(members).mean(0)
        correction = fit_affine(raw, yv)
        calibrated = calibrate(raw, correction)
        heads[task] = dict(models=models, calibration=correction, selected_algorithm=candidate)
        supports[task] = build_task_support(raw_x, [r['source_episode'] for r in rows], raw_dim)
        from scipy.spatial import cKDTree
        support = supports[task]
        distance = cKDTree(support['normalized_fit_features']).query(
            (xv - np.asarray(support['mean'])) / np.asarray(support['scale']))[0]
        supported = distance <= support['maximum_nearest_fit_distance']
        supported &= (xv >= np.asarray(support['lower'])).all(1)
        supported &= (xv <= np.asarray(support['upper'])).all(1)
        crossfits[task] = crossfit_calibration(raw, yv, calibration_rows, seed)
        metrics_raw[task] = metrics(raw, yv); metrics_cal[task] = metrics(calibrated, yv)
        priors[task] = metrics(np.full(len(yv), y.mean()), yv)
        audits[task] = weights_audit
        coverage[task] = dict(TRAIN_executed_sources=len(all_sources), TRAIN_feature_sources=len(feature_sources),
            zero_candidate_TRAIN_sources=sorted(all_sources-feature_sources),
            all_observed_TRAIN_sources_have_positive_ensemble_weights=all(a['all_source_weights_strictly_positive'] for a in weights_audit),
            TRAIN_attempts=len(rows), TRAIN_positive=int(y.sum()),
            TRAIN_attempts_by_domain=dict(Counter(r['sensor_domain'] for r in rows)),
            author_GL_VAL_executed_sources=sum(t == task for t, _ in expected['validation']),
            author_GL_VAL_feature_sources=len({r['source_episode'] for r in calibration_rows}),
            VAL_attempts=len(calibration_rows), VAL_positive=int(yv.sum()),
            author_GL_VAL_supported_attempts=int(supported.sum()),
            author_GL_VAL_support_fraction=float(supported.mean()),
            limitation='a fully executed source with zero feasible candidates contributes no fabricated critic sample; geometry rejects it before critic use')
        print(json.dumps(dict(task=task, selected=candidate, coverage=coverage[task],
            raw=metrics_raw[task], calibrated=metrics_cal[task], calibration_crossfit=crossfits[task]['metrics']), ensure_ascii=False), flush=True)
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), kind=FORMAT,
        serialization='trusted local Python pickle protocol 5; sklearn native runtime', sklearn_version=sklearn.__version__,
        predictor=args.predictor, input_dim=raw_dim, transformed_input_dim=raw_dim+len(FEATURE_SCHEMA['derived_names']),
        context_dim=16 if args.predictor == 'context' else 0,
        feature_schema=FEATURE_SCHEMA, feature_schema_sha256=FEATURE_SCHEMA_SHA256,
        controller_sha256=controller, selection_sha256=selection_hash, selection_files=selection_files,
        per_task={task: dict(ranking_enabled=selections[task]['ranking_enabled'],
            ranking_guard=selections[task]['ranking_guard'], selected_algorithm=selections[task]['selected']) for task in TASKS},
        formal_test_used=False, all_200_TRAIN_source_episodes_executed=True,
        probability_scope='native task success on this selected candidate attempt before replacement; actual approach and remaining budget included',
        fitting_domains=['legacy_OPENGL3_TRAIN_3_replicas_per_raw_source', 'author_OPENGL_TRAIN_1_replica_per_raw_source'],
        calibration_domain='author_OPENGL_VAL_3_replicas_per_raw_source_all_40_sources',
        final_evaluation_domain='author_OPENGL',
        model_selection='task-specific algorithm/hyperparameters from grouped 5-fold TRAIN CV only; no VAL selection',
        fixed_model_parameters=dict(logreg=dict(solver='lbfgs', max_iter=1500, standard_scaling=True),
            hist_gradient_boosting=dict(max_iter=150, min_samples_leaf=40, learning_rate=.05,
                l2_regularization=10., early_stopping=False)),
        calibration='task-specific positive affine, attempt-uniform; source-stratified two-fold crossfit diagnostics',
        train_data_sha256=hashlib.sha256(json.dumps({**legacy_files, **author_files}, sort_keys=True).encode()).hexdigest(),
        validation_data_sha256=hashlib.sha256(json.dumps(val_files, sort_keys=True).encode()).hexdigest(),
        context_encoder_sha256=next(iter({r['context_encoder_sha256'] for r in train if r.get('context_encoder_sha256')}), None),
        domain_limitations=['mixed renderer TRAIN attempts are pooled uniformly within task; old-domain samples are not relabeled as new-domain data',
            'all sensor replicas of a raw TRAIN source share a CV fold',
            'HDF5 development resets and mixture of geometry/softmax/uniform collection policies; learned selection distribution can differ',
            'positive-slope calibration cannot improve within-task candidate ranking by itself',
            'CAL metrics fitted on all VAL are not independent generalization scores; only crossfit excludes held calibration sources',
            'public TEST was used in previous versions; no fresh-unseen-TEST claim'])
    out.mkdir(parents=True, exist_ok=True)
    with (out/'value.pt.tmp').open('wb') as stream:
        pickle.dump(dict(format=FORMAT, metadata=metadata, tasks=heads), stream, protocol=5)
    (out/'value.pt.tmp').replace(out/'value.pt')
    support = dict(value_checkpoint_sha256=sha(out/'value.pt'), feature_schema_sha256=FEATURE_SCHEMA_SHA256,
        selection_sha256=selection_hash, formal_test_used=False, validation_used=False, tasks=supports)
    write(out/'value_support.json', support)
    report = dict(metadata=metadata, source_coverage=coverage,
        train_episodes=len(legacy_rolls)+len(author_rolls), train_attempts=len(train),
        validation_episodes=len(val_rolls), validation_attempts=len(val),
        model_selection=selections, ensemble_source_weights=audits,
        validation_raw_by_task=metrics_raw, per_task=metrics_cal, task_prevalence_baseline=priors,
        calibration_source_crossfit={task: crossfits[task]['metrics'] for task in TASKS},
        calibration_source_raw=metrics_raw,
        calibration_source_crossfit_prior={task: crossfits[task]['task_constant_prior'] for task in TASKS},
        calibration_crossfit=crossfits, calibration_crossfit_valid=all(c['valid'] for c in crossfits.values()),
        data_files=dict(TRAIN={**legacy_files, **author_files}, author_GL_VAL=val_files))
    write(out/'report.json', report)


if __name__ == '__main__':
    main()
