"""Full selected-task TRAIN, VAL selection/calibration, freeze, then four-way TEST.

Old r03 remains an incomplete exploratory run. Never reads its TEST outcomes
to select weights, budgets, candidates, or controller parameters.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.benchmark_protocol import file_sha256, verify_lock

BASE = ROOT/'artifacts/framework/pipeline_v2'
RUN = 'comparison_r04'
DYNAMICS = 'artifacts/framework/pipeline_v2/full_train_r04/dynamics/dynamics.pt'
CONFIG = 'config/framework/pipeline_v2_full_train_r04.json'
TASKS = ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')
VARIANTS = [(p, r) for p in ('cv', 'context') for r in ('geometry', 'value')]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def status(stage, **kwargs):
    write(BASE/'full_train_status.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'stage': stage, 'run': RUN, 'FASTER_used': False, 'NGFF_used': False,
        'train_source_episodes': 200, 'validation_source_episodes': 40, **kwargs})


def command(arguments, logname, sim=False):
    prefix = ['bash', 'scripts/run_local_sim.sh'] if sim else [sys.executable]
    with (ROOT/'logs/framework'/logname).open('a') as log:
        subprocess.run(prefix+arguments, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def collection(predictor, split, shard, shards):
    name = f'full_train_r04_{predictor}_{split}_shard{shard}'
    args = ['scripts/collect_full_task_rollouts.py', '--predictor', predictor, '--split', split,
        '--out', f'artifacts/framework/local_policy/{name}', '--shard-index', str(shard), '--shards', str(shards)]
    if predictor == 'context':
        args += ['--latent-checkpoint', DYNAMICS]
    command(args, name+'.log', sim=True)
    return name


def dynamics():
    report = ROOT/DYNAMICS
    if not report.exists():
        command(['scripts/train_context_dynamics_full.py', '--out', str(report.parent.relative_to(ROOT)),
            '--normalization', 'simnorm', '--updates', '10000', '--device', 'cuda'], 'full_train_r04_dynamics.log')
    data = json.loads(report.with_name('report.json').read_text())
    if data['metadata']['fit_episodes'] != 200 or data['metadata']['all_selected_task_TRAIN_used'] is not True:
        raise ValueError('dynamics did not fit all selected-task TRAIN episodes')


def merge(predictor, split, shards):
    out = ROOT/f'artifacts/framework/local_policy/full_train_r04_{predictor}_{split}'
    out.mkdir(parents=True, exist_ok=True)
    records = []
    keys = set()
    for shard in range(shards):
        source = out.with_name(out.name+f'_shard{shard}')
        report = json.loads((source/'report.json').read_text())
        if report['completed'] != report['expected']:
            raise ValueError(f'incomplete collection: {source}')
        for path in sorted(source.glob('pick*__*__*.json')):
            row = json.loads(path.read_text())
            if row['key'] in keys:
                raise ValueError('duplicate shard attempt')
            if row['split'] != split or row['predictor'] != predictor or row['formal_test_used']:
                raise ValueError('data contamination')
            keys.add(row['key'])
            records.append(row)
            for p in (path, path.with_suffix('.npz')):
                target = out/p.name
                if target.exists() and file_sha256(target) != file_sha256(p):
                    raise ValueError('merged record differs on resume')
                if not target.exists():
                    shutil.copy2(p, target)
    manifest = json.loads((ROOT/f'artifacts/framework/protocol/{split}_manifest.json').read_text())['episodes']
    expected = {(r['task'], r['raw_episode_id']) for r in manifest if r['task'] in TASKS}
    observed = {(r['task'], r['source_episode']) for r in records}
    valid = [r for r in records if r.get('initial_features') is not None and r.get('failure') != 'deadline_stop']
    valid_groups = {(r['task'], r['source_episode']) for r in valid}
    if len(records) != len(expected)*4 or observed != expected or valid_groups != expected:
        write(out/'coverage_failure.json', {'expected': len(expected), 'attempts': len(records),
            'missing_source_episodes': sorted(expected-observed), 'sources_without_valid_outcome': sorted(expected-valid_groups)})
        raise ValueError('every source episode must supply at least one valid gradient/calibration example')
    if len({r['controller_sha256'] for r in records}) != 1:
        raise ValueError('mixed controller versions')
    write(out/'report.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'completed': len(records), 'expected': len(expected)*4, 'valid_initializations': len(valid),
        'full_task_executed': True, 'formal_test_used': False, 'split': split,
        'source_episodes': len(expected), 'all_source_episodes_have_valid_outcomes': True,
        'controller_sha256': records[0]['controller_sha256'],
        'per_task': {t: {'source_episodes': sum(task == t for task, _ in valid_groups),
            'attempts': sum(r['task'] == t for r in records), 'valid': sum(r['task'] == t for r in valid),
            'successes': sum(r['task'] == t and r['official_task_success_immediate'] for r in valid)} for t in TASKS}})
    return str(out.relative_to(ROOT))


def value(predictor):
    train = merge(predictor, 'train', 3)
    val = merge(predictor, 'validation', 1)
    out = f'artifacts/framework/pipeline_v2/full_train_r04/{predictor}_success'
    if not (ROOT/out/'value.pt').exists():
        command(['scripts/train_success_predictor_full.py', '--data', train, '--validation', val,
            '--out', out, '--outcome', 'official_task_success_immediate', '--context-dim', '0', '--updates', '2000'],
            f'full_train_r04_{predictor}_success.log')
    report = json.loads((ROOT/out/'report.json').read_text())
    if report['partitions']['fit']['episodes'] != 200 or report['metadata']['all_selected_task_TRAIN_used'] is not True:
        raise ValueError('success predictor did not fit all selected-task TRAIN episodes')


def freeze():
    path = BASE/f'{RUN}_frozen_inputs.json'
    if path.exists():
        lock = json.loads(path.read_text())
        verify_lock(ROOT, lock)
        return lock
    config = json.loads((ROOT/'config/framework/pipeline_v2_intercept.json').read_text())
    config.update(version='pipeline_v2_full_task_TRAIN_r04_without_FASTER', dynamics_checkpoint=DYNAMICS,
        context_local_value='artifacts/framework/pipeline_v2/full_train_r04/context_success/value.pt',
        cv_local_value='artifacts/framework/pipeline_v2/full_train_r04/cv_success/value.pt')
    write(ROOT/CONFIG, config)
    files = [ROOT/CONFIG, ROOT/'scripts/evaluate_interception_v2.py', ROOT/'scripts/collect_local_rollouts.py',
        ROOT/'scripts/collect_full_task_rollouts.py', ROOT/'scripts/train_context_dynamics_full.py',
        ROOT/'scripts/train_success_predictor_full.py', Path(__file__).resolve(), *sorted((ROOT/'our_mani').glob('*.py'))]
    for key in ('dynamics_checkpoint', 'context_local_value', 'cv_local_value', 'goal_calibration'):
        files.append(ROOT/config[key])
    for key in ('context_local_value', 'cv_local_value'):
        files.append((ROOT/config[key]).with_name('value_support.json'))
        files.append((ROOT/config[key]).with_name('report.json'))
    for split in ('train', 'validation'):
        files.append(ROOT/f'artifacts/framework/protocol/{split}_manifest.json')
    files.append(ROOT/'artifacts/framework/pipeline_v1/official_test_manifest.json')
    lock = {'created_utc': datetime.now(timezone.utc).isoformat(), 'FASTER_used': False,
        'config': config, 'files': {str(p.relative_to(ROOT)): file_sha256(p) for p in files},
        'training_selection': 'all 200 task TRAIN episodes used for dynamics and both local Values; official VAL selects dynamics; disjoint VAL groups select/calibrate Values',
        'test_status': 'same public benchmark previously evaluated by v1 and partially by r03; no untouched TEST claim; no TEST-based tuning in r04',
        'local_outcomes': '4 matched perturbations per TRAIN source episode per local controller; invalid reset attempts disclosed'}
    write(path, lock)
    return lock


def evaluate(split, predictor, ranking, lock):
    verify_lock(ROOT, lock)
    name = f'{RUN}_{split}_{predictor}_{ranking}'
    args = ['scripts/evaluate_interception_v2.py', '--predictor', predictor, '--ranking', ranking,
        '--split', split, '--episodes-per-task', '0', '--config', CONFIG,
        '--out', f'artifacts/framework/pipeline_v2/{name}']
    if split == 'test' and predictor == 'context' and ranking == 'value':
        args += ['--videos', '2']
    command(args, name+'.log', sim=True)
    report = json.loads((BASE/name/'report.json').read_text())
    if report['complete'] is not True or report['completed'] != (80 if split == 'test' else 40):
        raise ValueError('incomplete evaluation')
    verify_lock(ROOT, lock)
    return name


def main():
    try:
        status('full_TRAIN_dynamics_and_CV_outcome_collection')
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(collection, 'cv', 'train', i, 3) for i in range(3)]
            futures.append(pool.submit(collection, 'cv', 'validation', 0, 1))
            dynamics()
            status('full_TRAIN_matching_controller_outcome_collection')
            futures.extend(pool.submit(collection, 'context', 'train', i, 3) for i in range(3))
            futures.append(pool.submit(collection, 'context', 'validation', 0, 1))
            for f in as_completed(futures):
                print(f.result(), flush=True)
        status('fitting_full_TRAIN_success_predictors_with_VAL_selection_calibration')
        for predictor in ('cv', 'context'):
            value(predictor)
        lock = freeze()
        for split in ('validation', 'test'):
            status(f'frozen_{split}_four_way_running', formal_test_used=split == 'test')
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(evaluate, split, p, r, lock) for p, r in VARIANTS]
                for f in as_completed(futures):
                    print(f.result(), flush=True)
        command(['scripts/report_interception_v2_full.py', '--run', RUN], 'full_train_r04_final_report.log')
        status('complete', formal_test_used=True, report='artifacts/framework/pipeline_v2/full_test_report.json')
        command(['scripts/overnight_report.py'], 'full_train_r04_final_report.log')
    except Exception as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
