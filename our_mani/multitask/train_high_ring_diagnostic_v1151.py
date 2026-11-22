"""Two fixed high-ring TRAIN runs with passive public/Native-call evidence.

The separately locked1141/1140 controller, original Native loop and outcome
sampling are inherited. A new diagnostic identity/worker/source lock is explicit;
these cases never impersonate1141 invocation or grant formal score admission.
"""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import train_author_release_full16_overlay_v1141 as parent

SOURCE = 'our_mani/multitask/train_high_ring_diagnostic_v1151.py'
WORKER = 'scripts/diagnose_TRAIN_high_ring_v1151_r10.py'
TEST_SOURCE = 'tests/test_train_high_ring_diagnostic_v1151.py'
SCHEMA = 'r10_fixed_two_high_ring_diagnostic_v1151'
LOCK_SCHEMA = SCHEMA + '_source_lock'
CPU_EVIDENCE_SCHEMA = SCHEMA + '_CPU_integration_evidence'
NAME = 'TRAIN_high_ring_diagnostic_v1151'
ARTIFACT_PREFIX = 'artifacts/framework/multitask_r10/' + NAME + '/development_demo_only/'
SETUP_PREFIX = 'artifacts/framework/multitask_r10/' + NAME + '_setup/'
RUNTIME_PREFIX = 'runtime/framework/' + NAME + '/cases/'
PARENT_LOCK = 'artifacts/framework/multitask_r10/TRAIN_author_release_full16_v1141_setup/source_overlay_lock.json'
PARENT_LOCK_SHA = 'cbb0f900d27270b2c41974827224ce105dff0f48d7714e5f65837b98dc3a26c8'
PARENT_PINS = {
    'our_mani/multitask/train_author_release_full16_overlay_v1141.py': 'ec76994e889f9649a4702887cd01aac66e315bf8c2084beb0e3b4c6866c7bfd4',
    'scripts/run_TRAIN_author_release_full16_overlay_v1141_r10.py': '7ce09b69c1e5051062cc570065d7b9b24e19cff7ec2e30728948ef6d3def1baf',
    'tests/test_train_author_release_full16_overlay_v1141.py': '7efd2e5abb6ff7f2e794341d661223b687295ce4fe1a456ef639febbf6d39964',
}
PASSIVE_PINS = {
    'our_mani/multitask/train_public_execution_trace_v1126.py': 'ffa610f07d45fafc1ef43785e66b86f111af20408ab650b9a042e4da122921bc',
    'tests/test_train_public_execution_trace_v1126.py': '7d64bc3d5eb519a56f0113c934f745a23086d9c12a9d9bb7f92371de0145d743',
    'our_mani/multitask/train_public_readiness_probe_v1134.py': '38b85888f726efc887a74f8f0ec47e12824cf613b9e28ea9fd79f4166949d3ec',
    'tests/test_train_public_readiness_probe_v1134.py': '393f56c155ed3e3147a2fb08e79dbbd8cc67c27b46392c401449ddc163f95be0',
    'our_mani/multitask/train_ring_success_tap_v1125.py': '73f12acae977616c4f5f754d3a827eb9d28ca0356dae606878c681551912e003',
    'tests/test_train_ring_success_tap_v1125.py': 'b89f2def4ad5faae895e2ccbc5d1e7b804b3ace7fe3556044fd7370b0ada3c06',
    'our_mani/multitask/train_buzz_failure_tap_v1105.py': '798308727ef9d4f32e3f3247f867d5dc17a07e81bf09e77b2e040223d564bdc4',
    'tests/test_train_buzz_failure_tap_v1105.py': '7e294521ce2e12a172c91a4a2e92f399be07303b82bc107f71a51c6ba184afdd',
}
SELECTIONS = (
    ('insert_onto_rotating_peg_high_speed', 'episode102', 'cv_geometry'),
    ('insert_onto_rotating_peg_high_speed', 'episode102', 'context_geometry'),
)
OVERLAY_SOURCES = (
    SOURCE, WORKER, TEST_SOURCE,
    'our_mani/multitask/train_public_readiness_probe_v1134.py',
    'tests/test_train_public_readiness_probe_v1134.py',
    'our_mani/multitask/train_public_execution_trace_v1126.py',
    'tests/test_train_public_execution_trace_v1126.py',
    'our_mani/multitask/train_ring_success_tap_v1125.py',
    'tests/test_train_ring_success_tap_v1125.py',
    'our_mani/multitask/train_buzz_failure_tap_v1105.py',
    'tests/test_train_buzz_failure_tap_v1105.py',
)


def require(value, message):
    if not value:
        raise ValueError('TRAIN1151: ' + message)


def _parent_lock(root):
    for name, expected in PARENT_PINS.items():
        parent.raw(root, name, expected)
    parent.raw(root, PARENT_LOCK, PARENT_LOCK_SHA)
    value, reference = parent.read_overlay_lock(root, PARENT_LOCK)
    parent.require_native_ready(dict(source_overlay=value))
    return value, reference


def _cpu_evidence(root, reference, files):
    require(type(reference) is dict and set(reference) == {'path', 'sha256'}
        and reference['path'].startswith(SETUP_PREFIX), 'explicit new diagnostic CPU evidence pointer required')
    value = json.loads(parent.raw(root, reference['path'], reference['sha256']))
    require(value.get('schema') == CPU_EVIDENCE_SCHEMA and value.get('complete') is True
        and type(value.get('tests_exit_code')) is int and value['tests_exit_code'] == 0
        and value.get('native_started') is False and value.get('GPU_started') is False
        and value.get('source_hashes') == files
        and value.get('parent_source_overlay_lock') == dict(path=PARENT_LOCK, sha256=PARENT_LOCK_SHA),
        'complete CPU evidence must bind every diagnostic source and exact parent lock')


def write_diagnostic_lock(root, name, *, integration_complete=False, integration_cpu_evidence=None):
    """Fresh lock only; pending sources cannot acquire Native permission."""
    root = Path(root).resolve()
    path = parent._path(root, name, prefix=SETUP_PREFIX, must_exist=False)
    require(type(integration_complete) is bool, 'explicit integration state required')
    _, reference = _parent_lock(root)
    for source, expected in PASSIVE_PINS.items(): parent.raw(root, source, expected)
    files = {}; unpinned = []
    for source in OVERLAY_SOURCES:
        if (root/source).exists(): files[source] = parent.pointer(root, source)['sha256']
        else: unpinned.append(source)
    require(not integration_complete or not unpinned, 'all passive diagnostic sources must exist')
    if integration_complete:
        _cpu_evidence(root, integration_cpu_evidence, files)
    else:
        require(integration_cpu_evidence is None, 'pending integration cannot claim CPU completion')
    value = dict(schema=LOCK_SCHEMA, required_sources=list(OVERLAY_SOURCES), files=files,
        unpinned_sources=unpinned, integration_complete=integration_complete,
        integration_cpu_evidence=integration_cpu_evidence, parent_source_overlay_lock=reference,
        fixed_TRAIN_sources=[list(row) for row in SELECTIONS], diagnostic_only=True,
        automatic_retry=False, formal_score_contribution=False, formal_VAL_TEST_qualification=False,
        TEST_used_for_fitting=False, additional_native_getters_or_condition_calls=0)
    parent.write_json(path, value)
    return parent.pointer(root, name)


def read_diagnostic_lock(root, name):
    parent._path(root, name, prefix=SETUP_PREFIX)
    value = json.loads(parent.raw(root, name)); files = value.get('files')
    require(value.get('schema') == LOCK_SCHEMA and value.get('required_sources') == list(OVERLAY_SOURCES)
        and type(files) is dict and set(files) <= set(OVERLAY_SOURCES)
        and value.get('unpinned_sources') == [s for s in OVERLAY_SOURCES if s not in files],
        'honest diagnostic source roster required')
    _, reference = _parent_lock(root)
    for source, expected in PASSIVE_PINS.items(): parent.raw(root, source, expected)
    require(value.get('parent_source_overlay_lock') == reference
        and value.get('fixed_TRAIN_sources') == [list(row) for row in SELECTIONS]
        and value.get('diagnostic_only') is True and value.get('automatic_retry') is False
        and value.get('formal_score_contribution') is False
        and value.get('formal_VAL_TEST_qualification') is False
        and value.get('TEST_used_for_fitting') is False
        and value.get('additional_native_getters_or_condition_calls') == 0,
        'fixed source-only passive diagnostic contract required')
    for source, expected in files.items(): parent.raw(root, source, expected)
    require(type(value.get('integration_complete')) is bool, 'explicit integration state required')
    if value['integration_complete']:
        require(not value['unpinned_sources'], 'Native cannot use missing diagnostic sources')
        _cpu_evidence(root, value.get('integration_cpu_evidence'), files)
    else:
        require(value.get('integration_cpu_evidence') is None, 'pending integration cannot claim CPU completion')
    return value, parent.pointer(root, name)


def require_native_ready(prepared):
    require(prepared['diagnostic_overlay']['integration_complete'] is True,
            'diagnostic integration pending; Native disabled')
    parent.require_native_ready(prepared)


def selection(root, args):
    key = (args.task, args.raw_episode_id, args.method)
    require(key in SELECTIONS, 'only fixed TRAIN highring102 CV/Context; no replacement or outcome selection')
    return parent.selection(root, args)


def prepare(root, args):
    root = Path(root).resolve(); chosen = selection(root, args)
    lock, reference = read_diagnostic_lock(root, args.diagnostic_lock)
    require(all(s in lock['files'] for s in (SOURCE, WORKER, TEST_SOURCE)),
            'actual diagnostic worker/preparer/tests must be pinned even for CPU check')
    out = parent._path(root, args.out, prefix=ARTIFACT_PREFIX, must_exist=False)
    require(not out.exists(), 'fresh diagnostic output required; no overwrite/retry')
    forwarded = SimpleNamespace(**vars(args))
    forwarded.overlay_lock = PARENT_LOCK
    # Only the namespace validation constant is scoped; no source/control change.
    with patch.object(parent, 'ARTIFACT_PREFIX', ARTIFACT_PREFIX):
        prepared = parent.prepare(root, forwarded)
    inherited_identity = deepcopy(prepared['identity'])
    inherited_protocol = deepcopy(prepared['execution_protocol'])
    protocol = dict(schema=SCHEMA, diagnostic_only=True,
        actual_worker=parent.pointer(root, WORKER), actual_preparer=parent.pointer(root, SOURCE),
        diagnostic_source_overlay_lock=reference,
        parent_source_overlay_lock=prepared['source_overlay_pointer'],
        inherited_1141_execution_protocol=inherited_protocol,
        inherited_1141_identity_sha256=inherited_identity['sha256'],
        actual_worker_is_parent1141=False,
        scoped_parent_prepare_output_namespace=ARTIFACT_PREFIX,
        fixed_TRAIN_sources=[list(row) for row in SELECTIONS],
        passive_readiness_probe='v1134_original_ready_readiness_candidates_observed_reference_calls',
        passive_public_execution_trace='v1126_existing_inputs_and_path_results_only',
        passive_native_conditions='v1125_original_high_ring_success_and_four_condition_returns',
        sysprofile_installed_during_native_only=True, simultaneous_profile_scopes=1,
        additional_native_getters_or_success_or_condition_calls=0,
        Native_loop_policy_models_outcome_sampling_unchanged=True,
        formal_score_contribution=False, formal_VAL_TEST_qualification=False,
        original_745_execution_acceptance_granted=False, automatic_retry=False)
    identity = dict(inherited_identity); identity.pop('sha256')
    identity.update(execution_protocol_schema=SCHEMA, execution_protocol_sha256=parent.digest(protocol),
        source_overlay_lock_sha256=reference['sha256'],
        inherited_1141_identity_sha256=inherited_identity['sha256'])
    identity['sha256'] = parent.digest(identity)
    prepared.update(identity=identity, execution_protocol=protocol,
        diagnostic_overlay=lock, diagnostic_overlay_pointer=reference,
        diagnostic_lock_name=args.diagnostic_lock, diagnostic_selection=chosen,
        inherited_1141_identity=inherited_identity)
    return prepared


def verify(root, prepared):
    parent.verify(root, prepared)
    value, reference = read_diagnostic_lock(root, prepared['diagnostic_lock_name'])
    require(value == prepared['diagnostic_overlay'] and reference == prepared['diagnostic_overlay_pointer'],
            'diagnostic source overlay changed during case')
    require(prepared['execution_protocol']['diagnostic_source_overlay_lock'] == reference
        and prepared['execution_protocol']['parent_source_overlay_lock'] == prepared['source_overlay_pointer'],
        'honest nested diagnostic/parent lock identities required')


def annotated_output(prepared, path, value):
    """Accept the original parent annotation once, then tag before first write."""
    value = deepcopy(value)
    require(type(value) is dict, 'dictionary diagnostic output required')
    value.update(execution_protocol=prepared['execution_protocol'],
        source_overlay_lock=prepared['diagnostic_overlay_pointer'],
        parent_source_overlay_lock=prepared['source_overlay_pointer'], diagnostic_only=True,
        formal_score_contribution=False, new_protocol_formal_VAL_TEST_qualification=False)
    if Path(path).name == 'case.json':
        value['inherited_1141_diagnostic_parent_case_version'] = value.pop('version', None)
        value['version'] = SCHEMA + '_case'
    return value


def require_passive_quality(events, *, ticks, terminal=False):
    """Audit collection quality, never turn unavailable evidence into failure.

    Original readyFalse/empty candidates/helper or planner exceptions are data.
    A terminal counted tick has no final frontend.observe in frozen v7/744;
    preserve that omission rather than issuing a diagnostic observation.
    """
    require(type(ticks) is int and 0<=ticks<=200 and type(terminal) is bool,'actual counted ticks and terminal bool required')
    require(type(events) is dict and events.get('schema')==SCHEMA+'_observers','actual diagnostic observer schema required')
    require(all(events.get(key) is True for key in ('restored','factory_restored','writer_restored','parent_annotation_restored'))
        and events.get('observer_finalization_errors')==[] and events.get('sysprofile_installed') is True,
        'all diagnostic scopes and the single profiler must restore')
    require(type(events.get('additional_native_getters_or_success_or_condition_calls')) is int
        and events['additional_native_getters_or_success_or_condition_calls']==0,'no added Native or condition calls permitted')
    def count(value,key,*,maximum=None):
        number=value.get(key)
        require(type(number) is int and number>=0 and (maximum is None or number<=maximum),'actual bounded diagnostic counter: '+key)
        return number
    def zero(value,keys):
        for key in keys:require(count(value,key)==0,'recording/drop/extra-call counter must be zero: '+key)
    def restored(value,schema):
        require(type(value) is dict and value.get('schema')==schema and value.get('restored') is True
            and value.get('restoration_error') is None and value.get('scope_exception_type') is None
            and value.get('errors')==[] and type(value.get('restoration_checks')) is list
            and bool(value['restoration_checks']) and all(row.get('restored') is True for row in value['restoration_checks']),
            'complete passive stream and restoration checks required')
    public=events.get('public_execution_trace',{});sink=events.get('readiness_probe',{})
    require(public.get('restored') is True and sink.get('restored') is True,'both public stream scopes must restore')
    trace=public.get('trace_receipt');probe=sink.get('readiness_probe');profile=events.get('passive_native_conditions')
    restored(trace,'r10_evaluation_existing_public_execution_trace_v1126')
    restored(probe,'r10_evaluation_existing_public_readiness_probe_v1134')
    require(trace.get('sysprofile_installed') is False and probe.get('sysprofile_installed') is False
        and trace.get('original_return_or_exception_rewritten') is False
        and probe.get('original_return_or_exception_rewritten') is False,'public observers must preserve original calls/results')
    zero(trace,('recording_errors','frame_dropped','plan_dropped','plan_value_truncations','extra_env_observe_calls',
        'extra_native_getter_or_condition_calls','extra_prediction_calls','extra_path_step_or_FK_calls'))
    zero(probe,('recording_errors','dropped_events','extra_readiness_reference_forecast_or_native_calls'))
    expected_frames=ticks+(0 if terminal and ticks>0 else 1)
    frame_calls=count(trace,'frame_calls');frame_returns=count(trace,'frame_returns');frame_exceptions=count(trace,'frame_exceptions')
    require(frame_calls==frame_returns+frame_exceptions and frame_returns==expected_frames
        and type(trace.get('frames')) is list and len(trace['frames'])==frame_returns
        and trace.get('max_frames')==201 and trace.get('max_plans')==24,'all existing counted public frames must be retained')
    plans=count(trace,'plan_calls',maximum=24)
    require(plans==count(trace,'plan_returns')+count(trace,'plan_exceptions')
        and type(trace.get('plans')) is list and len(trace['plans'])==plans
        and count(trace,'geometry_calls')==1,'one original geometry factory and bounded original plans required')
    rows=probe.get('events');methods=('frontend.ready','collision_port.readiness','source_policy.candidates','generator.observed_reference')
    counts=probe.get('method_call_counts')
    require(type(rows) is list and len(rows)<=1000 and probe.get('max_events')==1000
        and type(counts) is dict and set(counts)==set(methods),'bounded original readiness/reference records required')
    require(count(probe,'geometry_calls')==1 and count(probe,'geometry_returns')==1
        and count(probe,'geometry_exceptions')==0,'original geometry construction must have returned once')
    for method in methods:
        totals=counts[method];calls=count(totals,'calls');returns=count(totals,'returns');exceptions=count(totals,'exceptions')
        records=[row for row in rows if row.get('method')==method]
        require(calls==returns+exceptions and len(records)==calls
            and sum(row.get('original_returned') is True for row in records)==returns
            and sum(row.get('original_returned') is False for row in records)==exceptions
            and all('original_return' in row if row.get('original_returned') is True else
                type(row.get('original_exception')) is dict and type(row['original_exception'].get('type')) is str for row in records),
            'original method call/return/exception records must agree: '+method)
    require(sum(counts[m]['calls'] for m in methods)==len(rows)
        and all(row.get('method') in methods and type(row.get('call_sequence')) is int and row['call_sequence']>0 for row in rows)
        and len({row['call_sequence'] for row in rows})==len(rows),'unique actual readiness call records required')
    require(type(profile) is dict and profile.get('schema')=='r10_fixed_TRAIN_ring_passive_original_success_tree_v1125'
        and profile.get('task')=='insert_onto_rotating_peg_high_speed' and profile.get('method') in ('cv_geometry','context_geometry')
        and profile.get('restored') is True and profile.get('scope_exception_type') is None
        and profile.get('existing_profiler_never_overwritten') is True
        and profile.get('physical_methods_patched') is False,'original high-ring passive success profile required')
    zero(profile,('dropped_events','observer_errors','exception_or_nonstandard_returns','registration_truncations','inner_sequence_drops',
        'extra_success_calls','extra_condition_calls','extra_physical_getter_calls'))
    success_calls=count(profile,'original_success_calls_observed');success_returns=count(profile,'original_success_returns_observed')
    condition_calls=count(profile,'original_condition_calls_observed');condition_returns=count(profile,'original_condition_returns_observed')
    require(success_calls==success_returns and success_returns>0 and condition_calls==condition_returns==5*success_returns,
        'original success/ConditionSet/four leaves call-return accounting required')
    records=profile.get('events');require(type(records) is list and len(records)<=5000,'bounded original success profile records required')
    successes=[row for row in records if row.get('event')=='original_task_success_return']
    conditions=[row for row in records if row.get('event')=='original_condition_met_return']
    require(len(successes)==success_returns and len(conditions)==condition_returns
        and sum(row.get('event')=='original_task_success_call' for row in records)==success_calls
        and sum(row.get('event')=='original_condition_met_call' for row in records)==condition_calls,
        'all observed original success/condition events must be retained')
    require(all(row.get('original_four_detector_registration') is True and type(row.get('leaf_returns')) is list
        and len(row['leaf_returns'])==4 and {leaf.get('ordinal') for leaf in row['leaf_returns']}=={0,1,2,3}
        and all(type(leaf.get('result')) is list and len(leaf['result'])==2
            and all(type(value) is bool for value in leaf['result']) for leaf in row['leaf_returns']) for row in successes)
        and {row.get('registered_leaf_ordinal') for row in conditions if row.get('registered_leaf_ordinal') is not None}=={0,1,2,3},
        'each original success sample must contain the four observed registered leaves')
    return dict(complete=True,restored=True,recording_errors=0,dropped=0,public_frames=frame_returns,
        expected_existing_public_frames=expected_frames,terminal_frame_omission_preserved=terminal and ticks>0,
        original_plans=plans,original_plan_exceptions=trace['plan_exceptions'],readiness_events=len(rows),
        original_method_counts=deepcopy(counts),unobserved_original_methods=[m for m in methods if counts[m]['calls']==0],
        observed_success_returns=success_returns,observed_condition_returns=condition_returns,
        readyFalse_empty_candidates_and_original_helper_exceptions_are_data=True,
        trace_unknown_reasons=deepcopy(trace.get('unknown_reasons',{})),probe_unknown_reasons=deepcopy(probe.get('unknown_reasons',{})),
        no_extra_calls=True,diagnostic_quality_is_not_native_success=True)


@contextmanager
def runtime_scope(root, prepared, runtime, events):
    """Parent→trace→readiness→one original-success profiler; never alter inputs."""
    require_native_ready(prepared); verify(root, prepared)
    original_geometry, original_writer = runtime._geometry, runtime.atomic_json
    original_annotation = parent.annotated_output
    trace_events = {}; probe_events = {}; original_error = None; finalization_errors = []; tap=None
    try:
        with parent.runtime_scope(root, prepared, runtime):
            def annotate(current, path, value):
                require(current is prepared, 'diagnostic writer escaped its prepared case')
                return annotated_output(current, path, original_annotation(current, path, value))
            with patch.object(parent, 'annotated_output', annotate):
                trace = importlib.import_module('our_mani.multitask.train_public_execution_trace_v1126')
                with trace.public_execution_trace_scope(runtime, trace_events, max_frames=201, max_plans=24):
                    probe = importlib.import_module('our_mani.multitask.train_public_readiness_probe_v1134')
                    with probe.readiness_probe_scope(runtime, probe_events, max_events=1000):
                        profile_module=importlib.import_module('our_mani.multitask.train_ring_success_tap_v1125')
                        tap=profile_module.PassiveRingSuccessTap(prepared['args'].task,prepared['args'].method,max_events=5000)
                        with tap:
                            try:
                                yield
                            except BaseException as error:
                                original_error = (error, error.__traceback__)
                                raise
        if original_error is not None:
            finalization_errors.append(dict(stage='observer_exit',reason='original_exception_was_suppressed'))
            raise original_error[0].with_traceback(original_error[1])
    except BaseException as error:
        if original_error is not None and error is not original_error[0]:
            finalization_errors.append(dict(stage='observer_or_parent_exit',type=type(error).__name__))
            raise original_error[0].with_traceback(original_error[1]) from error
        raise
    finally:
        profile=None
        if tap is not None:
            try:profile=tap.receipt()
            except BaseException as error:
                finalization_errors.append(dict(stage='passive_profile_receipt',type=type(error).__name__))
        events.update(schema=SCHEMA + '_observers', readiness_probe=probe_events,
            public_execution_trace=trace_events,passive_native_conditions=profile,
            observer_finalization_errors=finalization_errors,
            sysprofile_installed=tap is not None and vars(tap).get('_entered') is True,
            factory_restored=runtime._geometry is original_geometry,
            writer_restored=runtime.atomic_json is original_writer,
            parent_annotation_restored=parent.annotated_output is original_annotation,
            restored=(runtime._geometry is original_geometry and runtime.atomic_json is original_writer
                and parent.annotated_output is original_annotation
                and probe_events.get('restored') is True
                and trace_events.get('restored') is True
                and profile is not None and profile.get('restored') is True),
            additional_native_getters_or_success_or_condition_calls=0,
            native_held_identity_or_unobserved_condition_claimed=False)
    verify(root, prepared)
