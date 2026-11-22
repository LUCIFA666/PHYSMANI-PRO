"""Paired episode-level forecast uncertainty and an exportable scientific plot."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_domain_dynamics import aggregate


def main():
    directory = ROOT/'artifacts/framework/domain_adaptation'
    report = json.loads((directory/'report.json').read_text())
    rows = report['rows']
    keys = sorted({(r['task'], r['raw_episode_id']) for r in rows})
    methods = sorted({r['method'] for r in rows})
    metrics = {method: {} for method in methods}
    for method in methods:
        for horizon in ('0.1', '0.3', '0.6'):
            metrics[method][horizon] = np.array([np.mean([r['metrics'][horizon]['mean_position_error_m'] for r in rows
                    if r['method'] == method and (r['task'], r['raw_episode_id']) == key]) for key in keys])
    rng = np.random.default_rng(881)
    # Stratify task mix and cluster all prediction windows from a raw episode.
    samples = []
    for task in sorted({key[0] for key in keys}):
        indices = np.array([i for i,key in enumerate(keys) if key[0] == task])
        samples.append(indices[rng.integers(len(indices), size=(10000, len(indices)))])
    paired = {}
    for horizon in ('0.1', '0.3', '0.6'):
        difference = metrics['ngff_driver_adapter'][horizon]-metrics['driver_model_without_ngff'][horizon]
        draws = np.mean([difference[index].mean(1) for index in samples], axis=0)
        point = np.mean([np.mean(difference[[key[0] == task for key in keys]]) for task in sorted({k[0] for k in keys})])
        paired[horizon] = {'ngff_minus_no_ngff_error_m': float(point),
                           'task_stratified_episode_bootstrap_95_m': np.quantile(draws, [.025, .975]).tolist(),
                           'independent_episode_groups': len(keys)}
    output = {'created_utc': datetime.now(timezone.utc).isoformat(), 'kind': report['kind'], 'formal_test_used': False,
              'paired_ngff_contribution': paired,
              'per_task': {task: aggregate([r for r in rows if r['task'] == task]) for task in sorted({k[0] for k in keys})},
              'coverage': {'source_validation_episodes': 40, 'usable_episode_groups': len(keys),
                           'usable_windows': report['validation_windows'],
                           'exclusion': 'all methods require the same valid causal perception history and forecast window'},
              'interpretation': 'negative paired error difference favors NGFF; intervals crossing zero do not establish an independent benefit'}
    (directory/'paired_analysis.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1.5, 1.]})
    labels = {'ngff_frozen': 'Frozen NGFF', 'constant_velocity': 'Constant velocity',
              'causal_acceleration_with_declared_boundaries': 'CA + known boundaries',
              'driver_model_without_ngff': 'Driver MLP without NGFF', 'ngff_driver_adapter': 'NGFF + driver adapter'}
    colors = {'ngff_frozen': '#d75b54', 'constant_velocity': '#898d96',
              'causal_acceleration_with_declared_boundaries': '#b99131',
              'driver_model_without_ngff': '#45a286', 'ngff_driver_adapter': '#3e73c9'}
    horizons = np.array([.1, .3, .6])
    for method in methods:
        values = [report['summary'][method]['metrics'][str(h)]['mean_position_error_m']*1000 for h in horizons]
        axes[0].plot(horizons, values, '-o', label=labels[method], color=colors[method], lw=1.8)
    axes[0].set(xlabel='Forecast horizon (s)', ylabel='Mean position error (mm)', title='Held-out causal RGB-D forecasts')
    axes[0].grid(alpha=.2)
    axes[0].legend(fontsize=8)
    means = np.array([paired[str(h)]['ngff_minus_no_ngff_error_m'] for h in horizons])*1000
    bounds = np.array([paired[str(h)]['task_stratified_episode_bootstrap_95_m'] for h in horizons])*1000
    axes[1].errorbar(horizons, means, yerr=np.c_[means-bounds[:,0], bounds[:,1]-means].T, fmt='o', color='#3e73c9', capsize=5)
    axes[1].axhline(0, color='#555555', lw=1, ls='--')
    axes[1].set(xlabel='Forecast horizon (s)', ylabel='NGFF minus no-NGFF error (mm)', title='Paired contribution; 95% interval')
    axes[1].grid(alpha=.2)
    fig.suptitle('Two-task development validation; 38 episode groups, 126 shared windows', fontsize=11)
    fig.tight_layout()
    fig.savefig(directory/'domain_forecast_comparison.png', dpi=180)
    plt.close(fig)
    print(json.dumps(paired, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
