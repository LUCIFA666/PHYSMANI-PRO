"""Recalibrate frozen r07 critics; no classifier fitting and no simulator.

Run through scripts/run_local_sim.sh for the existing sklearn pickle runtime.
The outer r07 source assignment is fixed. Each outer-fit subset independently
selects its correction using only inner source folds, before scoring outer-held.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import numpy as np
from scipy.optimize import minimize
from scipy.special import logit
from our_mani.risk_v2.selector import attempt_controller_hash
from our_mani.risk_v3.model import (TASKS, CandidateProbability as R07CandidateProbability,
    transform_features, calibrate)
from our_mani.risk_v4.model import (CALIBRATION_VERSION, LAMBDAS, TIE_TOLERANCE,
    calibration_program_hash)
from train_candidate_success_r07 import (balanced_group_folds, load_domain, metrics)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identity():
    return dict(slope=1., intercept=0., method='identity', regularization_lambda=None,
        target='unchanged raw frozen r07 ensemble probability')


def validate(raw, y, rows):
    raw, y = np.asarray(raw, float), np.asarray(y, float)
    if raw.ndim != 1 or raw.shape != y.shape or len(rows) != len(y) or not len(y):
        raise ValueError('invalid nested calibration observations')
    if not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)) or not np.isin(y, (0, 1)).all():
        raise ValueError('invalid probability or binary calibration label')
    if not all('source_episode' in row for row in rows):
        raise ValueError('nested calibration requires raw-source identities')
    return raw, y


def fit_shrunk_affine(raw, y, regularization):
    if regularization not in LAMBDAS:
        raise ValueError('unregistered calibration regularization')
    if len(np.unique(y)) != 2:
        raise ValueError('affine fit requires both outcome classes')
    z = logit(np.clip(np.asarray(raw, float), 1e-6, 1-1e-6))
    y = np.asarray(y, float)
    def objective(theta):
        logits = theta[0]*z + theta[1]
        return float(np.mean(np.logaddexp(0., logits)-y*logits)
            + regularization*((theta[0]-1.)**2+theta[1]**2))
    fitted = minimize(objective, [1., 0.], method='L-BFGS-B',
        bounds=[(.05, 5.), (-8., 8.)], options=dict(ftol=1e-12, gtol=1e-8, maxiter=300))
    if not fitted.success:
        raise ValueError('strong affine calibration optimization failed: '+str(fitted.message))
    return dict(slope=float(fitted.x[0]), intercept=float(fitted.x[1]),
        method='strongly_shrunk_positive_affine', regularization_lambda=regularization,
        target='uniform attempts within one task',
        fit_objective='NLL + lambda*((slope-1)^2+intercept^2)',
        slope_bounds=[.05, 5.], intercept_bounds=[-8., 8.])


def source_coverage(rows):
    sources = sorted({r['source_episode'] for r in rows})
    positive = sum(any(r['official_task_success'] for r in rows if r['source_episode']==s) for s in sources)
    negative = sum(any(not r['official_task_success'] for r in rows if r['source_episode']==s) for s in sources)
    return dict(sources=len(sources), positive_sources=int(positive), negative_sources=int(negative))


def select_correction(raw, y, rows, seed):
    """Receives only its fit subset; no access to any outer-held observations."""
    raw, y = validate(raw, y, rows)
    coverage = source_coverage(rows)
    fit_data = dict(raw=raw.tolist(), labels=y.tolist(), sources=[r['source_episode'] for r in rows])
    trace = dict(fit_sources=sorted({r['source_episode'] for r in rows}), seed=seed,
        fit_data_sha256=hashlib.sha256(json.dumps(fit_data, sort_keys=True).encode()).hexdigest(),
        source_coverage=coverage, criterion='pooled held-attempt Brier over two inner source folds',
        tie_priority='identity, lambda .01, .1, 1; improvement must exceed 1e-10',
        candidates=[], fallback_reason=None)
    identity_p = calibrate(raw, identity())
    trace['candidates'].append(dict(kind='identity', valid=True, pooled_brier=metrics(identity_p, y)['brier']))
    if coverage['positive_sources'] < 2 or coverage['negative_sources'] < 2:
        trace.update(selected='identity', fallback_reason='fewer than two positive or negative raw sources; do not fit an affine',
            selected_correction=identity(), inner_source_folds=None,
            selected_inner_pooled_brier=trace['candidates'][0]['pooled_brier'])
        return identity(), trace
    folds, grouping = balanced_group_folds(rows, 2, seed)
    trace['inner_source_folds'] = grouping
    if any(len(np.unique(y[folds!=fold])) != 2 for fold in range(2)):
        trace.update(selected='identity', fallback_reason='an inner fit fold lacks an outcome class',
            selected_correction=identity(), selected_inner_pooled_brier=trace['candidates'][0]['pooled_brier'])
        return identity(), trace
    best_index, best_brier = 0, trace['candidates'][0]['pooled_brier']
    for regularization in LAMBDAS:
        probabilities = np.zeros(len(y)); details = []; errors = []
        for fold in range(2):
            held = folds==fold; fit = ~held
            fit_sources = sorted({r['source_episode'] for r, h in zip(rows, held) if not h})
            held_sources = sorted({r['source_episode'] for r, h in zip(rows, held) if h})
            if set(fit_sources)&set(held_sources): raise AssertionError('inner raw-source leakage')
            try:
                correction = fit_shrunk_affine(raw[fit], y[fit], regularization)
                probabilities[held] = calibrate(raw[held], correction)
                details.append(dict(held_fold=fold, fit_sources=fit_sources, held_sources=held_sources,
                    correction=correction, held_metrics=metrics(probabilities[held], y[held])))
            except ValueError as error:
                errors.append(str(error))
        candidate = dict(kind='affine', regularization_lambda=regularization, valid=not errors,
            pooled_brier=metrics(probabilities, y)['brier'] if not errors else None,
            folds=details, errors=errors)
        trace['candidates'].append(candidate)
        if not errors and candidate['pooled_brier'] < best_brier-TIE_TOLERANCE:
            best_index, best_brier = len(trace['candidates'])-1, candidate['pooled_brier']
    if best_index == 0:
        correction = identity(); chosen = 'identity'
    else:
        regularization = trace['candidates'][best_index]['regularization_lambda']
        correction = fit_shrunk_affine(raw, y, regularization); chosen = 'affine_lambda_'+str(regularization)
    trace.update(selected=chosen, selected_correction=correction, selected_inner_pooled_brier=best_brier)
    return correction, trace


def nested_crossfit(raw, y, rows, original_grouping):
    raw, y = validate(raw, y, rows)
    mapping = original_grouping['source_to_fold']
    if set(mapping) != {r['source_episode'] for r in rows} or set(mapping.values()) != {0, 1}:
        raise ValueError('outer folds must match every exact frozen r07 feature source')
    folds = np.array([mapping[r['source_episode']] for r in rows])
    predicted = np.zeros(len(y)); prior = np.zeros(len(y)); details = []
    seed = int(original_grouping['seed'])
    for fold in range(2):
        held = folds==fold; fit = ~held
        fitrows = [r for r, h in zip(rows, held) if not h]
        held_sources = sorted({r['source_episode'] for r, h in zip(rows, held) if h})
        fit_sources = sorted({r['source_episode'] for r in fitrows})
        if set(fit_sources)&set(held_sources): raise AssertionError('outer raw-source leakage')
        correction, selection = select_correction(raw[fit], y[fit], fitrows, seed+2000+fold)
        predicted[held] = calibrate(raw[held], correction)
        prior[held] = float(y[fit].mean())
        details.append(dict(held_fold=fold, fit_sources=fit_sources, held_sources=held_sources,
            fit_attempts=int(fit.sum()), held_attempts=int(held.sum()),
            fit_positive=int(y[fit].sum()), held_positive=int(y[held].sum()),
            correction=correction, inner_selection=selection,
            held_metrics=metrics(predicted[held], y[held]), constant_prior=float(y[fit].mean()), valid=True))
    return dict(valid=True, reasons=[], grouping=copy.deepcopy(original_grouping), folds=details,
        metrics=metrics(predicted, y), raw_metrics=metrics(raw, y), task_constant_prior=metrics(prior, y),
        class_coverage=source_coverage(rows),
        interpretation='fixed r07 outer sources; selector/fitting only on each outer-fit; identity valid without affine-class identification; source coverage disclosed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictor', choices=('cv', 'context'), required=True)
    parser.add_argument('--config', default='config/framework/aligned_protocol_r08.json')
    args = parser.parse_args()
    config = json.loads((ROOT/args.config).read_text())
    controller = attempt_controller_hash(ROOT, config, args.predictor)
    old = ROOT/f'artifacts/framework/aligned_r07/{args.predictor}_success'
    out = ROOT/f'artifacts/framework/aligned_r08/{args.predictor}_success'
    if any((out/name).exists() for name in ('value.pt', 'value_support.json', 'report.json')):
        raise FileExistsError('r08 critic artifacts are immutable')
    loaded = R07CandidateProbability(old/'value.pt', controller_sha256=controller)
    with (old/'value.pt').open('rb') as stream: saved = pickle.load(stream)
    old_report = json.loads((old/'report.json').read_text())
    support = json.loads((old/'value_support.json').read_text())
    manifest = json.loads((ROOT/'artifacts/framework/protocol/validation_manifest.json').read_text())
    expected = {(r['task'], r['raw_episode_id']) for r in manifest['episodes'] if r['task'] in TASKS}
    rows, _, files = load_domain(ROOT/'artifacts/framework/aligned_r07', args.predictor,
        'validation', 3, 'author_OPENGL', expected, controller)
    if files != old_report['data_files']['author_GL_VAL']:
        raise ValueError('r08 calibration must use the identical frozen r07 author-GL VAL records')
    source_files = [ROOT/'our_mani/risk_v4/model.py', Path(__file__)]
    calibration_files = {str(path.relative_to(ROOT)): sha(path) for path in source_files}
    metadata = copy.deepcopy(saved['metadata'])
    metadata.update(created_utc=datetime.now(timezone.utc).isoformat(),
        kind='candidate_success_r08_nested_calibration', calibration_version=CALIBRATION_VERSION,
        calibration_program_sha256=calibration_program_hash(calibration_files), calibration_files=calibration_files,
        raw_head_checkpoint_sha256=sha(old/'value.pt'), raw_head_report_sha256=sha(old/'report.json'),
        raw_head_support_sha256=sha(old/'value_support.json'), raw_selection_sha256=loaded.metadata['selection_sha256'],
        raw_head_path=str((old/'value.pt').relative_to(ROOT)), classifier_retrained=False,
        classifier_selection_used_CAL=False, calibration_selector_used_CAL=True,
        classifier_selection='unchanged r07 task-specific grouped TRAIN five-fold selection',
        calibration='nested source-CV choice of identity/strong affine; final fullCAL inner selection then fit',
        calibration_evaluation='unchanged r07 outer source split; each outer-fit independently selects lambda without outer-held access',
        final_evaluation_domain='author_OPENGL')
    heads = {}; crossfits = {}; raw_metrics = {}; calibrated_metrics = {}; final_selection = {}
    for task in TASKS:
        local = [r for r in rows if r['task']==task]
        x = transform_features(np.asarray([r['features'] for r in local]), loaded.metadata['input_dim'])
        y = np.array([r['official_task_success'] for r in local], int)
        if args.predictor == 'context' and {r.get('context_encoder_sha256') for r in local} != {loaded.metadata['context_encoder_sha256']}:
            raise ValueError('r08 CAL context encoder differs from the frozen r07 raw head')
        head = saved['tasks'][task]
        raw = np.stack([model.predict_proba(x)[:, 1] for model in head['models']]).mean(0)
        raw_metrics[task] = metrics(raw, y)
        for key in ('brier', 'log_loss', 'auroc'):
            actual, prior = raw_metrics[task][key], old_report['validation_raw_by_task'][task][key]
            identical = (actual is None and prior is None) or (actual is not None and prior is not None
                and np.isclose(actual, prior, rtol=0, atol=1e-12))
            if not identical:
                raise ValueError('frozen r07 raw probabilities differ on '+task+' '+key)
        grouping = old_report['calibration_crossfit'][task]['grouping']
        crossfits[task] = nested_crossfit(raw, y, local, grouping)
        correction, trace = select_correction(raw, y, local, int(grouping['seed']))
        final_selection[task] = trace
        heads[task] = dict(head, calibration=correction)
        calibrated_metrics[task] = metrics(calibrate(raw, correction), y)
    metadata['calibration_per_task'] = {task: dict(selected=final_selection[task]['selected'],
        correction=final_selection[task]['selected_correction']) for task in TASKS}
    out.mkdir(parents=True, exist_ok=True)
    with (out/'value.pt.tmp').open('wb') as stream:
        pickle.dump(dict(saved, metadata=metadata, tasks=heads), stream, protocol=5)
    (out/'value.pt.tmp').replace(out/'value.pt')
    support.update(value_checkpoint_sha256=sha(out/'value.pt'),
        raw_support_sha256=sha(old/'value_support.json'), calibration_program_sha256=metadata['calibration_program_sha256'])
    write(out/'value_support.json', support)
    report = copy.deepcopy(old_report)
    report.update(metadata=metadata, per_task=calibrated_metrics, validation_raw_by_task=raw_metrics,
        calibration_crossfit=crossfits, calibration_crossfit_valid=all(d['valid'] for d in crossfits.values()),
        calibration_source_crossfit={task: d['metrics'] for task, d in crossfits.items()},
        calibration_source_raw=raw_metrics,
        calibration_source_crossfit_prior={task: d['task_constant_prior'] for task, d in crossfits.items()},
        nested_calibration_selection=final_selection,
        classifier_selection_used_CAL=False, calibration_selector_used_CAL=True,
        reused_r07_source_coverage=True, reused_r07_TRAIN_support=True,
        calibration_scope_limitations=['calibration selection uses CAL labels; classifier selection remains TRAIN-only',
            'final full-CAL fitted scores are development diagnostics; outer nested scores exclude held sources',
            'positive-slope correction retains raw within-task candidate ordering; no SR improvement claim',
            'rare class sources can force identity; this does not manufacture class-identification evidence'])
    write(out/'report.json', report)
    # Loading validates the copied support/raw feature/ranking contracts and the new calibration lineage.
    from our_mani.risk_v4.model import CandidateProbability
    CandidateProbability(out/'value.pt', controller_sha256=controller)
    print(json.dumps(dict(predictor=args.predictor, final_selection=metadata['calibration_per_task'],
        crossfit={task: d['metrics'] for task, d in crossfits.items()}), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
