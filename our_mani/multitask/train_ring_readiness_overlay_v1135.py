"""One immutable TRAIN source with passive public readiness/reference evidence.

The separately locked1132/1131 controller, original Native loop and outcome
sampling are inherited. A new diagnostic identity/worker/source lock is explicit;
these cases never impersonate1132 invocation or grant formal score admission.
"""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import train_public_phase_full16_overlay_v1132 as parent

SOURCE = 'our_mani/multitask/train_ring_readiness_overlay_v1135.py'
WORKER = 'scripts/diagnose_TRAIN_ring_readiness_v1135_r10.py'
TEST_SOURCE = 'tests/test_train_ring_readiness_overlay_v1135.py'
SCHEMA = 'r10_fixed_one_TRAIN_ring_readiness_diagnostic_v1135'
LOCK_SCHEMA = SCHEMA + '_source_lock'
CPU_EVIDENCE_SCHEMA = SCHEMA + '_CPU_integration_evidence'
NAME = 'TRAIN_ring_readiness_v1135'
ARTIFACT_PREFIX = 'artifacts/framework/multitask_r10/' + NAME + '/development_demo_only/'
SETUP_PREFIX = 'artifacts/framework/multitask_r10/' + NAME + '_setup/'
RUNTIME_PREFIX = 'runtime/framework/' + NAME + '/cases/'
PARENT_LOCK = 'artifacts/framework/multitask_r10/TRAIN_public_phase_full16_v1132_setup/source_overlay_lock.json'
PARENT_LOCK_SHA = '99a04a6434f90d588656a9b5b8b0f708e5c54848a80868703349f5cc3509d290'
PARENT_PINS = {
    'our_mani/multitask/train_public_phase_full16_overlay_v1132.py': 'be0f763aba7e23caddacd6fb6b6cdfbdb38ed273522f19004ee5ba61905e850a',
    'scripts/run_TRAIN_public_phase_full16_overlay_v1132_r10.py': 'ce5253dc95e0653155476e3f7dd6ef7608a8a7475c6d553769365edc49edcbb0',
    'tests/test_train_public_phase_full16_overlay_v1132.py': 'd04e762b51a2beff661adc9f8b3729c34b7086392f3aa9cbd429eb78714536ae',
}
PASSIVE_PINS = {
    'our_mani/multitask/train_public_execution_trace_v1126.py': 'ffa610f07d45fafc1ef43785e66b86f111af20408ab650b9a042e4da122921bc',
    'tests/test_train_public_execution_trace_v1126.py': '7d64bc3d5eb519a56f0113c934f745a23086d9c12a9d9bb7f92371de0145d743',
}
SELECTIONS = (
    ('insert_onto_rotating_peg', 'episode101', 'cv_geometry'),
)
OVERLAY_SOURCES = (
    SOURCE, WORKER, TEST_SOURCE,
    'our_mani/multitask/train_public_readiness_probe_v1134.py',
    'tests/test_train_public_readiness_probe_v1134.py',
    'our_mani/multitask/train_public_execution_trace_v1126.py',
    'tests/test_train_public_execution_trace_v1126.py',
)


def require(value, message):
    if not value:
        raise ValueError('TRAIN1135: ' + message)


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
    require(key in SELECTIONS, 'only fixed TRAIN ring101CV; no replacement or outcome selection')
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
        inherited_1132_execution_protocol=inherited_protocol,
        inherited_1132_identity_sha256=inherited_identity['sha256'],
        actual_worker_is_parent1132=False,
        scoped_parent_prepare_output_namespace=ARTIFACT_PREFIX,
        fixed_TRAIN_sources=[list(row) for row in SELECTIONS],
        passive_readiness_probe='v1134_original_ready_readiness_candidates_observed_reference_calls',
        passive_public_execution_trace='v1126_existing_inputs_and_path_results_only',
        sysprofile_installed=False,
        additional_native_getters_or_success_or_condition_calls=0,
        Native_loop_policy_models_outcome_sampling_unchanged=True,
        formal_score_contribution=False, formal_VAL_TEST_qualification=False,
        original_745_execution_acceptance_granted=False, automatic_retry=False)
    identity = dict(inherited_identity); identity.pop('sha256')
    identity.update(execution_protocol_schema=SCHEMA, execution_protocol_sha256=parent.digest(protocol),
        source_overlay_lock_sha256=reference['sha256'],
        inherited_1132_identity_sha256=inherited_identity['sha256'])
    identity['sha256'] = parent.digest(identity)
    prepared.update(identity=identity, execution_protocol=protocol,
        diagnostic_overlay=lock, diagnostic_overlay_pointer=reference,
        diagnostic_lock_name=args.diagnostic_lock, diagnostic_selection=chosen,
        inherited_1132_identity=inherited_identity)
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
        value['inherited_1132_diagnostic_parent_case_version'] = value.pop('version', None)
        value['version'] = SCHEMA + '_case'
    return value


@contextmanager
def runtime_scope(root, prepared, runtime, events):
    """Native only: parent factory, public trace, then passive readiness probe."""
    require_native_ready(prepared); verify(root, prepared)
    original_geometry, original_writer = runtime._geometry, runtime.atomic_json
    original_annotation = parent.annotated_output
    trace_events = {}; probe_events = {}; original_error = None; finalization_errors = []
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
        events.update(schema=SCHEMA + '_observers', readiness_probe=probe_events,
            public_execution_trace=trace_events,
            observer_finalization_errors=finalization_errors,sysprofile_installed=False,
            factory_restored=runtime._geometry is original_geometry,
            writer_restored=runtime.atomic_json is original_writer,
            parent_annotation_restored=parent.annotated_output is original_annotation,
            restored=(runtime._geometry is original_geometry and runtime.atomic_json is original_writer
                and parent.annotated_output is original_annotation
                and probe_events.get('restored') is True
                and trace_events.get('restored') is True),
            additional_native_getters_or_success_or_condition_calls=0,
            native_held_identity_or_unobserved_condition_claimed=False)
    verify(root, prepared)
