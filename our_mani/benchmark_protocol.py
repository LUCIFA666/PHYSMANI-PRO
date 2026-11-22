"""Freeze/resume accounting for a single checkpoint benchmark."""
import hashlib
import json
import math
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(rows):
    keys = [(r['task'], r['variation'], r['official_episode_index']) for r in rows]
    raw = [(r['task'], r['raw_episode_id']) for r in rows]
    tasks = ('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed')
    expected = {(task,variation,index) for task in tasks for variation in (0,1) for index in range(20)}
    if len(keys) != 80 or len(set(raw)) != 80 or set(keys) != expected:
        raise ValueError('TEST requires every official task/variation/index exactly once')


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes/total
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [center-radius,center+radius]


def verify_lock(root, lock):
    for name, expected in lock['files'].items():
        if file_sha256(Path(root)/name) != expected:
            raise ValueError(f'frozen benchmark input changed: {name}')


def summarize(rows, expected, paper):
    results = {}
    for task in sorted({r['task'] for r in expected}):
        records = [r for r in rows if r['task']==task]
        n, k = len(records),sum(bool(r['official_task_success']) for r in records)
        results[task] = {'completed':n,'expected':sum(r['task']==task for r in expected),
            'successes':k,'success_rate':k/n if n else None,'wilson_95':wilson(k,n),
            'handovers':sum(r['handovers']>0 for r in records),
            'stable_workflow_successes_verified':sum(r['stable_workflow_success'] is True for r in records),
            'stable_workflow_success_not_assessed':sum(r['stable_workflow_success'] is None for r in records),
            'paper_reported_PhysMani':paper['task_success_reference']['methods']['PhysMani'][task]}
    return {'complete':len(rows)==len(expected),'completed':len(rows),'expected':len(expected),
        'formal_test_used':True,'per_task':results,'baseline_source':'paper_reported_not_local_reproduction',
        'paper_url':paper['paper'],'paper_aggregation':paper['task_success_reference']['aggregation'],
        'ours_aggregation':'one frozen checkpoint, all official episodes; Wilson interval is binomial, not paper checkpoint std',
        'paired_significance_test':False}
