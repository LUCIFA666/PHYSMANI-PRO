"""Full16 release checks for the three user-authorized comparison methods.

Native failures remain scored outcomes. Infrastructure failures, absent
measurements, changed policies and incomplete source grids cannot release TEST.
The original baseline's measured motion is reported without changing it.
"""
from collections import Counter
from .evaluation import digest
from .evaluation_roster16_v283 import METHODS,TASKS,validate_roster


def policy_fingerprint(routes):
    rows=routes.get('routes',[])
    keys=[(r['task'],r['method']) for r in rows]
    if len(keys)!=48 or set(keys)!={(t,m) for t in TASKS for m in METHODS}:
        raise ValueError('exactly one immutable route per full16 task/method')
    for r in rows:
        if r.get('GPU_required') is not (r['method']=='physmani'):
            raise ValueError('GPU allocation differs from the authorized method')
        for key in ('configuration','frozen_inputs','worker','consumer'):
            p=r.get(key,{})
            if not isinstance(p.get('path'),str) or len(p.get('sha256',''))!=64:
                raise ValueError('every executable route dependency requires a frozen pointer')
    return digest(dict(methods=list(METHODS),routes=sorted(rows,key=lambda r:(r['task'],r['method'])),
        shared_context_checkpoint_sha256=routes['shared_context_checkpoint_sha256']))


def measured_motion(case):
    a=case.get('actor_metrics') or {}
    value=(a.get('task_dispatch_v43') or {}).get('actual_observed_window_limits_passed')
    if type(value) is bool:return value
    m=a.get('continuous_reference_feedforward') or {}
    failed=m.get('observed_motion_qualification_failed')
    if type(failed) is bool and m.get('observed_events'):return not failed
    for report in (case.get('all_stage_counted_joint_motion'),
        (case.get('alignment_receipt') or {}).get('all_stage_counted_joint_motion_v191')):
        if isinstance(report,dict) and type(report.get('all_stage_actual_window_checks_passed')) is bool:
            return report['all_stage_actual_window_checks_passed']
    return None


def validate_grid(cases,manifest,split,policy_sha,*,completed_VAL_audit=False):
    sources=validate_roster(manifest,split,completed_VAL_audit=completed_VAL_audit)
    expected={(r['task'],r['raw_episode_id'],m) for r in sources for m in METHODS}
    source_sha={(r['task'],r['raw_episode_id']):digest(r) for r in sources}
    keys=[];rows=[]
    for case in cases:
        raw=case.get('source_raw_episode_id',case.get('parent_raw_episode_id',case.get('raw_episode_id')))
        key=(case.get('task'),raw,case.get('method'));keys.append(key)
        if key not in expected or case.get('split')!=split:raise ValueError('case is outside the exact source/method grid')
        if (case.get('complete') is not True or case.get('cleanup_completed') is not True
            or case.get('infrastructure_error') is not None):raise ValueError('all cases need clean completed native execution')
        witness=case.get('full16_frozen_execution') or {}
        if witness.get('policy_sha256')!=policy_sha or witness.get('source_row_sha256')!=source_sha[key[:2]]:
            raise ValueError('case policy or original source row differs from the frozen full16 suite')
        if case.get('demo_only') or case.get('development_demo_only') or case.get('TEST_used_for_fitting') is not False:
            raise ValueError('formal grid cannot contain development/demo cases or TEST fitting')
        native=case.get('official_task_success');motion=measured_motion(case)
        if type(native) is not bool or type(motion) is not bool:raise ValueError('native outcome and measured motion must be explicit')
        rows.append(dict(task=key[0],source=raw,method=key[2],native_success=native,motion_pass=motion))
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('incomplete or duplicated full16 three-method cohort')
    ours=[r for r in rows if r['method']!='physmani']
    return dict(source_cases=len(sources),method_cases=len(rows),completed_method_cases=len(rows),
        methods=list(METHODS),tasks=list(TASKS),policy_sha256=policy_sha,
        native_successes_by_method={m:sum(r['native_success'] for r in rows if r['method']==m) for m in METHODS},
        measured_motion_violations_by_method={m:sum(not r['motion_pass'] for r in rows if r['method']==m) for m in METHODS},
        ours_measured_motion_passed=all(r['motion_pass'] for r in ours),
        native_success_is_not_a_release_threshold=True,baseline_algorithm_unchanged=True,
        complete=True,split=split,TEST_used_for_fitting=False)


def validation_release(cases,manifest,routes):
    proof=validate_grid(cases,manifest,'validation',policy_fingerprint(routes))
    if proof['source_cases']!=315 or proof['method_cases']!=945:raise ValueError('complete315-source/945-case VAL required')
    return dict(proof,schema='r10_full16_three_method_validation_release_v1',
        formal_test_permitted=proof['ours_measured_motion_passed'],TEST_used=False)


def require_test_release(release,routes):
    if (release.get('schema')!='r10_full16_three_method_validation_release_v1'
        or release.get('complete') is not True or release.get('split')!='validation'
        or release.get('TEST_used') is not False or release.get('TEST_used_for_fitting') is not False
        or release.get('formal_test_permitted') is not True or release.get('ours_measured_motion_passed') is not True
        or release.get('source_cases')!=315 or release.get('method_cases')!=945
        or release.get('completed_method_cases')!=945 or release.get('methods')!=list(METHODS)
        or release.get('tasks')!=list(TASKS) or release.get('policy_sha256')!=policy_fingerprint(routes)):
        raise ValueError('complete unchanged full16 three-method VAL release required before TEST selection')
    return True
