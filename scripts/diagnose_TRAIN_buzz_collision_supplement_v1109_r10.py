"""Four preselected TRAIN Buzz representatives with both passive observers.

Never extends, changes or retries the running48 plan. Root may explicitly run
these isolated supplements only after the complete48 owner has naturally
exited. Frozen1103 owns all TRAIN/source/820/resource/Native checks.
"""
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'scripts/diagnose_TRAIN_buzz_collision_supplement_v1109_r10.py'
SCHEMA = 'r10_fixed_four_TRAIN_buzz_combined_collision_native_supplement_v1109'
BASE_PATH = 'scripts/diagnose_TRAIN_ring_postcomplete_v1103_r10.py'
BASE_SHA = 'd1b2aec0f3058433b3f2edc6d7e31c3022f17b84d1a7d333b45744eddf8c1349'
PROFILE_PATH = 'our_mani/multitask/train_buzz_failure_tap_v1105.py'
MAIN_RUNTIME = 'runtime/framework/TRAIN_zero_success_diagnosis_v1107'
MAIN_PLAN = 'artifacts/framework/multitask_r10/TRAIN_zero_success_diagnosis_v1107_setup/plan.json'
MAIN_PLAN_SHA = '7af96771e16f33506d8770fd75c010cbdc900e1e86d09b27c9118fb7b8948515'
SUFFIX = 'supplement_buzz_collision_v1109/'
OUT_PREFIX = 'artifacts/framework/multitask_r10/TRAIN_zero_success_diagnosis_v1107/development_demo_only/' + SUFFIX
RUNTIME_PREFIX = MAIN_RUNTIME + '/cases/' + SUFFIX
REPRESENTATIVES = {
    ('beat_the_rotating_buzz', 'episode10'),
    ('beat_the_rotating_buzz_high_speed', 'episode102'),
}
METHODS = ('cv_geometry', 'context_geometry')


def require(value, message):
    if not value:
        raise ValueError('Buzz supplement1109: ' + message)


def _raw(root, name, expected=None):
    path = Path(root).resolve() / name
    require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts
            and path.resolve(strict=True) == path, 'canonical existing project input required')
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    before = stamp(path.stat())
    value = path.read_bytes()
    require(before == stamp(path.stat()) and (expected is None or hashlib.sha256(value).hexdigest() == expected),
            'input changed while read: ' + name)
    return value


def _pointer(root, name):
    data = _raw(root, name)
    return dict(path=name, sha256=hashlib.sha256(data).hexdigest()), json.loads(data)


def _base():
    _raw(ROOT, BASE_PATH, BASE_SHA)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts import diagnose_TRAIN_ring_postcomplete_v1103_r10
    return diagnose_TRAIN_ring_postcomplete_v1103_r10


def validate_selection(selection):
    require(type(selection) is dict and (selection.get('task'), selection.get('raw_episode_id')) in REPRESENTATIVES
            and selection.get('method') in METHODS,
            'only preselected ordinary episode10/highspeed episode102 x CV/Context representatives')


def require_main48_complete(root):
    """Read only completion fields and owner process identity; no new job."""
    _raw(root, MAIN_PLAN, MAIN_PLAN_SHA)
    tp, terminal = _pointer(root, MAIN_RUNTIME + '/terminal.json')
    sp, status = _pointer(root, MAIN_RUNTIME + '/status.json')
    lp, launch = _pointer(root, MAIN_RUNTIME + '/launch.json')
    require(terminal.get('complete') is True and terminal.get('completed') == 48
            and terminal.get('infrastructure_failed') == 0 and terminal.get('automatic_retry') is False,
            'actual complete48 clean terminal required')
    completed = status.get('completed')
    require(status.get('stage') == 'complete' and status.get('expected') == 48
            and type(completed) is list and len(completed) == 48 and status.get('failed') == []
            and status.get('active') is None and len({r['job_id'] for r in completed}) == 48,
            'actual complete48 status without queued/active/failed cases required')
    require(launch.get('plan') == dict(path=MAIN_PLAN, sha256=MAIN_PLAN_SHA)
            and type(launch.get('pid')) is int and type(launch.get('start_ticks')) is int,
            'original1107 owner/plan identity required')
    try:
        stat = (Path('/proc') / str(launch['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        stat = None
    require(stat is None or stat[0] in ('Z', 'X') or int(stat[19]) != launch['start_ticks'],
            'the actual48 owner must have naturally exited before supplement')
    return dict(complete=True, plan=dict(path=MAIN_PLAN, sha256=MAIN_PLAN_SHA),
                terminal=tp, status=sp, launch=lp, completed=48,
                original_owner_identity_not_live=True, earlier_results_modified=False)


@contextmanager
def observer_context_factory(root, runtime, selection, events, *, permit,
                             collision_factory=None, native_factory=None):
    validate_selection(selection)
    base = _base()
    if native_factory is None:
        from our_mani.multitask.train_buzz_failure_tap_v1105 import observer_context_factory as native_factory
    collision_factory = collision_factory or base.ring_observer_scope
    collision_events, native_events = {}, {}
    original_error = None
    try:
        with ExitStack() as stack:
            stack.enter_context(collision_factory(root, runtime, selection, collision_events, permit=permit))
            stack.enter_context(native_factory(root, runtime, selection, native_events, permit=permit))
            yield
    except BaseException as error:
        original_error = error
        raise
    finally:
        try:
            # Scopes have both exited before collection. Each retains a
            # separate sink and its original unknown/drop/error semantics.
            events['collision'] = collision_events
            events['native'] = native_events
            events['restored'] = collision_events.get('restored') is True and native_events.get('restored') is True
            events['observer_receipts'] = [dict(observer='public_collision_admission', data=collision_events.get('observer_receipts', [])),
                                           dict(observer='native_task_outcome_profile', data=native_events.get('observer_receipts', []))]
            events['summary'] = dict(collision=collision_events.get('summary', {'status': 'unknown'}),
                                   native=native_events.get('summary', {'status': 'unknown'}),
                                   represents_all16_Buzz_cases=False, extra_physical_queries=0,
                                   return_values_and_original_predicate_calls_unchanged=True)
            if 'expected_identity' in collision_events:
                events['expected_identity'] = collision_events['expected_identity']
            if original_error is not None:
                events['original_scope_exception_type'] = type(original_error).__name__
        except BaseException as recording_error:
            events['combo_recording_error'] = type(recording_error).__name__
            if original_error is None:
                raise


def main(argv=None):
    base = _base()
    parser = base.parser()
    parser.description = __doc__
    parser.set_defaults(diagnostic='buzz')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute:
        return dict(enabled=False, native_started=False, GPU_started=False, materials_created=False)
    selection = dict(task=args.task, raw_episode_id=args.raw_episode_id, method=args.method)
    validate_selection(selection)
    require(args.diagnostic == 'buzz', 'Buzz diagnostic only')
    suffix = args.task + '/' + args.raw_episode_id + '/' + args.method
    require(args.out == OUT_PREFIX + suffix and args.runtime == RUNTIME_PREFIX + suffix,
            'fixed fresh separate1109 supplement output/runtime required')
    prerequisite = require_main48_complete(ROOT)
    def combo(root, runtime, selected, events, *, permit):
        events['supplement'] = dict(schema=SCHEMA, source=dict(path=SOURCE, sha256=hashlib.sha256(_raw(root, SOURCE)).hexdigest()),
            prerequisite=prerequisite, representative_selection=selection,
            rationale='outcome-independent first original TRAIN source in each speed x both frozen geometry methods',
            original48_plan_extended=False, current48_automatic_retry=False)
        return observer_context_factory(root, runtime, selected, events, permit=permit)
    result = base.run_case(args, observer_context_factory=combo)
    print(json.dumps(dict(schema=SCHEMA, complete=True, mode=result['execution']['mode'],
                         selection=selection, formal_score_contribution=False), ensure_ascii=False))
    return result


if __name__ == '__main__':
    main()
