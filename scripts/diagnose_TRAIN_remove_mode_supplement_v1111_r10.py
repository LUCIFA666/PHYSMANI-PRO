"""Two fixed TRAIN remove representatives, original AST gates plus mode locals.

Explicitly invoked only after the original48 owner naturally exits. Uses the
frozen1103 single-case entry and1104 AST scope unchanged. No new source choice,
model fitting, policy decision, native getter, physical probe or retry.
"""
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'scripts/diagnose_TRAIN_remove_mode_supplement_v1111_r10.py'
SCHEMA = 'r10_fixed_two_TRAIN_remove_mode_supplement_v1111'
BASE = 'scripts/diagnose_TRAIN_ring_postcomplete_v1103_r10.py'
BASE_SHA = 'd1b2aec0f3058433b3f2edc6d7e31c3022f17b84d1a7d333b45744eddf8c1349'
GATE = 'scripts/diagnose_TRAIN_buzz_collision_supplement_v1109_r10.py'
GATE_SHA = '41720a55b74daceb855815b7eb5dd0ad20fb158375d9d7ff2f6e0aefffc64994'
CAPTURE = 'scripts/diagnose_TRAIN_remove_postcomplete_v1104_r10.py'
MODE = 'our_mani/multitask/train_remove_mode_tap_v1111.py'
REPRESENTATIVES = {('remove_cups_from_rotating_frame', 'episode104'),
                   ('remove_cups_from_rotating_frame_high_speed', 'episode1')}
OUT_PREFIX = 'artifacts/framework/multitask_r10/TRAIN_zero_success_diagnosis_v1107/development_demo_only/supplement_remove_mode_v1111/'
RUNTIME_PREFIX = 'runtime/framework/TRAIN_zero_success_diagnosis_v1107/cases/supplement_remove_mode_v1111/'


def require(value, message):
    if not value:
        raise ValueError('remove mode supplement1111: ' + message)


def modules():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    for name, checksum in ((BASE, BASE_SHA), (GATE, GATE_SHA)):
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == checksum, 'frozen supplement dependency changed')
    from scripts import diagnose_TRAIN_ring_postcomplete_v1103_r10 as base
    from scripts import diagnose_TRAIN_remove_postcomplete_v1104_r10 as capture
    from scripts import diagnose_TRAIN_buzz_collision_supplement_v1109_r10 as gate
    from our_mani.multitask import train_remove_mode_tap_v1111 as mode
    return base, capture, gate, mode


def validate_selection(selection):
    require(type(selection) is dict and (selection.get('task'), selection.get('raw_episode_id')) in REPRESENTATIVES
            and selection.get('method') == 'cv_geometry', 'only fixed normal104/high1 CV representatives')


@contextmanager
def observer_context_factory(root, runtime, selection, events, *, permit,
                             capture_factory=None, mode_tap_class=None):
    validate_selection(selection)
    _, capture, _, mode = modules()
    capture_factory = capture_factory or capture.observer_context_factory
    mode_tap_class = mode_tap_class or mode.PassiveRemoveModeTap
    before = runtime._geometry
    capture_events, taps = {}, []
    original_error = None
    try:
        with ExitStack() as stack:
            stack.enter_context(capture_factory(root, runtime, selection, capture_events, permit=permit))
            geometry = runtime._geometry
            def observed_geometry(prepared, env, descriptions):
                require(not taps, 'one actual geometry construction per representative')
                result = geometry(prepared, env, descriptions)
                require(type(result) is tuple and len(result) == 5, 'original geometry tuple required')
                actor = result[0]
                require(actor is not None, 'original nonblock public actor required')
                # The original constructor has completely returned before the
                # profile is enabled. Never call observe to manufacture data.
                tap = mode_tap_class(actor.mode_observer)
                taps.append(tap)
                stack.enter_context(tap)
                return result
            stack.enter_context(patch.object(runtime, '_geometry', observed_geometry))
            yield
    except BaseException as error:
        original_error = error
        raise
    finally:
        try:
            receipts = [tap.receipt() for tap in taps]
            events['capture'] = capture_events
            events['mode'] = dict(observer_receipts=receipts,
                                 summary=[mode.summarize(r) for r in receipts])
            events['geometry_hook_restored'] = runtime._geometry is before
            events['restored'] = (capture_events.get('restored') is True and runtime._geometry is before
                                  and len(receipts) == 1 and receipts[0].get('restored') is True)
            events['observer_receipts'] = [dict(observer='original_remove_AST_gates', data=capture_events.get('observer_receipts', [])),
                                           dict(observer='original_mode_observer_locals', data=receipts)]
            events['summary'] = dict(capture=capture_events.get('summary', {'status': 'unknown'}),
                                    mode=events['mode']['summary'],
                                    represents_all16_remove_cases=False, extra_physical_queries=0,
                                    uncomputed_candidate_list_is_unknown_not_empty=True)
            if 'expected_identity' in capture_events:
                events['expected_identity'] = capture_events['expected_identity']
            if original_error is not None:
                events['original_scope_exception_type'] = type(original_error).__name__
        except BaseException as error:
            events['combo_recording_error'] = type(error).__name__
            if original_error is None:
                raise


def main(argv=None):
    base, _, gate, _ = modules()
    parser = base.parser()
    parser.description = __doc__
    parser.set_defaults(diagnostic='remove')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute:
        return dict(enabled=False, native_started=False, GPU_started=False, materials_created=False)
    selection = dict(task=args.task, raw_episode_id=args.raw_episode_id, method=args.method)
    validate_selection(selection)
    require(args.diagnostic == 'remove', 'remove diagnostic only')
    suffix = args.task + '/' + args.raw_episode_id + '/' + args.method
    require(args.out == OUT_PREFIX + suffix and args.runtime == RUNTIME_PREFIX + suffix,
            'separate fresh1111 output/runtime required')
    prerequisite = gate.require_main48_complete(ROOT)
    def combo(root, runtime, selected, events, *, permit):
        events['supplement'] = dict(schema=SCHEMA, source=dict(path=SOURCE, sha256=hashlib.sha256((root / SOURCE).read_bytes()).hexdigest()),
            mode_source=dict(path=MODE, sha256=hashlib.sha256((root / MODE).read_bytes()).hexdigest()),
            prerequisite=prerequisite, representative_selection=selection,
            selection_basis='preselected original TRAIN close-evidence representatives; no TEST or outcome filtering',
            original48_plan_extended=False)
        return observer_context_factory(root, runtime, selected, events, permit=permit)
    result = base.run_case(args, observer_context_factory=combo)
    print(json.dumps(dict(schema=SCHEMA, complete=True, mode=result['execution']['mode'],
                         selection=selection, formal_score_contribution=False), ensure_ascii=False))
    return result


if __name__ == '__main__':
    main()
