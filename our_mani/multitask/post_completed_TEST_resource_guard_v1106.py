"""Serial TRAIN diagnostic lease after the actual completed 1071 pipeline.

This guard does not grant a scientific release, reinterpret scores, traverse
source-data graphs, restart old jobs, or impersonate an obsolete completion
chain. Frozen924 supplies only its canonical same-process permit registry.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path

SOURCE = 'our_mani/multitask/post_completed_TEST_resource_guard_v1106.py'
HANDOFF = 'runtime/framework/latest_executor_continuation_v1102.json'
HANDOFF_SHA = '1d95a0137eecb0ed9f642ce60140028f0e56d9707ba89414afa56475c57523ed'
LEGACY = 'our_mani/multitask/post_TEST_resource_guard_v924.py'
LEGACY_SHA = '635d7e7a570982599ffb912a8cb65b5badb25907b4100677159955e4eb3afbed'
MOTION_STAGE = 'runtime/framework/post_complete_second_reboot_TEST_motion_v1076/stages/saved_motion_v1075/terminal.json'
MOTION_STAGE_SHA = '40048015d0c53d2bb85f6712b39c0fa8d2e90f668b12567cb7eec793454e47e9'
POLICY = 'eb8f768d99975a7c2de2c36de5283d64898c2296f33fb342e332310eb32db6ef'
LOCKS = (
    'runtime/framework/TRAIN_postcomplete_diagnostics_v1106.lock',
    'runtime/framework/fixed_TRAIN104_geometry_native.lock',
    'runtime/framework/ordered_second_reboot_TEST_v1073.lock',
    'runtime/framework/post_complete_second_reboot_TEST_motion_v1076.lock',
    'runtime/framework/ordered_reboot_TEST_v1058.lock',
    'runtime/framework/ordered_composite_TEST_v1043.lock',
    'runtime/framework/ordered_composite_TEST_v1013.lock',
    'runtime/framework/ordered_composite_TEST_v958.lock',
    'runtime/framework/ordered_concurrent_TEST_v907.lock',
    'runtime/framework/ordered_env_resume_VAL_TEST_v827.lock',
    'runtime/framework/ordered_resume_VAL_TEST_v775.lock',
)
STOP_PATHS = (
    'runtime/framework/multitask_r10_STOP',
    'runtime/framework/TRAIN_postcomplete_diagnostics_v1106/STOP',
    'runtime/framework/TRAIN_postcomplete_diagnostics_v1107/STOP',
    'runtime/framework/TRAIN_zero_success_diagnosis_v1107/STOP',
    'runtime/framework/formal_TEST_reboot_resume_v1071/STOP',
    'runtime/framework/ordered_second_reboot_TEST_v1073/STOP',
    'runtime/framework/post_complete_second_reboot_TEST_motion_v1076/STOP',
)


def require(value, message):
    if not value:
        raise ValueError('postcomplete1106: ' + message)


def raw(root, name, expected):
    root = Path(root).resolve()
    require(type(name) is str and name and not Path(name).is_absolute()
            and '..' not in Path(name).parts, 'canonical relative path required')
    path = root / name
    require(path.resolve(strict=True) == path, 'metadata/source alias refused')
    stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    before = stamp(path.stat())
    value = path.read_bytes()
    require(before == stamp(path.stat()) and hashlib.sha256(value).hexdigest() == expected,
            'fixed completion/source bytes changed: ' + name)
    return value


def _json(root, pointer):
    return json.loads(raw(root, pointer['path'], pointer['sha256']))


def _legacy(root):
    raw(root, LEGACY, LEGACY_SHA)
    module = importlib.import_module('our_mani.multitask.post_TEST_resource_guard_v924')
    require(Path(module.__file__).resolve() == Path(root).resolve() / LEGACY,
            'canonical924 registry required')
    return module


def require_complete_TEST_metadata(root):
    """Validate fixed completion metadata; ignore every success/motion value."""
    handoff = json.loads(raw(root, HANDOFF, HANDOFF_SHA))
    require(handoff.get('scientific_pipeline_complete') is True
            and handoff.get('full1920_independent_TEST_audit_complete') is True
            and handoff.get('saved_motion_summary_complete') is True
            and handoff.get('TEST_running') is False
            and handoff.get('goal_status') == 'complete', 'actual completed1102 required')
    counts = handoff['current_runtime_counts']
    require(counts['completed'] == counts['expected'] == 1920
            and all(counts[k] == 0 for k in ('failed', 'active_native', 'GPU_slots', 'CPU_checking', 'CPU_ready')),
            'no outstanding1071 queue permitted')
    require(handoff['groups'] == ['cv_geometry', 'context_geometry', 'physmani'], 'three frozen groups required')
    pointers = handoff['actual_completed_terminal_pointers']
    terminals = {p['path']: _json(root, p) for p in pointers}
    require(len(terminals) == 4 and all(v.get('complete') is True for v in terminals.values()),
            'four actual completed1071/1073/1074/1076 terminals required')
    owner = terminals['runtime/framework/formal_TEST_reboot_resume_v1071/terminal.json']
    require(owner.get('expected') == owner.get('completed') == 1920
            and owner.get('failed') == 0 and owner.get('owner_error') is None
            and owner.get('automatic_retry') is False, 'clean actual owner terminal required')
    audit = _json(root, handoff['independent_TEST_audit']['actual_stage_terminal'])
    motion_stage = json.loads(raw(root, MOTION_STAGE, MOTION_STAGE_SHA))
    require(all(v.get('complete') is True and v.get('returncode') == 0
                and v.get('automatic_retry') is False for v in (audit, motion_stage)),
            'actual1072/1075 clean stage exits required')
    result = _json(root, handoff['formal_TEST_result'])
    motion = _json(root, handoff['saved_motion_summary'])
    require(result.get('schema') == 'r10_seven_tier_second_reboot_formal_TEST_results_v1072'
            and result.get('complete') is True and result.get('source_cases') == 640
            and result.get('method_cases') == result.get('completed_method_cases') == 1920
            and result.get('policy_sha256') == POLICY and result.get('TEST_used_for_fitting') is False,
            'actual full1072 completion required')
    require(motion.get('schema') == 'r10_full1920_seven_tier_TEST_saved_motion_summary_v1075'
            and motion.get('complete') is True and motion.get('case_count') == 1920
            and motion.get('source_count') == 640 and motion.get('task_count') == 16
            and motion.get('policy_sha256') == POLICY
            and motion.get('test_results') == handoff['formal_TEST_result'], 'actual1075 completion required')
    require(terminals['runtime/framework/ordered_second_reboot_TEST_v1073/terminal.json']['TEST_results']
            == handoff['formal_TEST_result'], '1073 result pointer mismatch')
    post = terminals['runtime/framework/post_complete_second_reboot_TEST_motion_v1076/terminal.json']
    require(post.get('TEST_results') == handoff['formal_TEST_result']
            and post.get('motion_summary') == handoff['saved_motion_summary'], '1076 pointers mismatch')
    for pointer in handoff['current_sources']:
        raw(root, pointer['path'], pointer['sha256'])
    known = [dict(pid=v['pid'], start_ticks=v['start_ticks'])
             for v in handoff['historical_completed_processes'].values()]
    known.append(dict(pid=handoff['independent_TEST_audit']['actual_pid'],
                      start_ticks=handoff['independent_TEST_audit']['start_ticks']))
    return dict(complete=True, gate_chain='actual1071/1072/1073 then actual1075/1076',
                handoff=dict(path=HANDOFF, sha256=HANDOFF_SHA),
                TEST_results=handoff['formal_TEST_result'], motion_summary=handoff['saved_motion_summary'],
                terminals=pointers, known_processes=known, expected_cases=1920,
                scores_used_for_gate=False, motion_values_used_for_gate=False,
                new_release_granted=False, scientific_inputs_changed=False)


def _stop(root):
    if any((Path(root) / p).exists() for p in STOP_PATHS):
        raise InterruptedError('global/formal/new diagnostic STOP blocks a new TRAIN case')


@contextmanager
def _reserve(root):
    import fcntl
    with ExitStack() as stack:
        for name in LOCKS:
            path = Path(root) / name
            require(path.parent.resolve() == path.parent and not path.is_symlink(), 'canonical lock required')
            stream = stack.enter_context(path.open('a+'))
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError('another formal/diagnostic owner holds ' + name) from error
        yield


def require_permit(permit, root, *, allow_test_callbacks=False):
    return _legacy(root).require_permit(permit, root, allow_test_callbacks=allow_test_callbacks)


def _inventory(root, legacy):
    """Frozen924 inventory, with current owner names included independently."""
    snapshot = legacy._inventory(root)
    own = set(snapshot['excluded_self_and_ancestor_pids'])
    extras = ('run_second_reboot_TEST_', 'continue_second_reboot_TEST_',
              'audit_second_reboot_TEST_', 'continue_post_second_reboot_TEST_',
              'summarize_second_reboot_TEST_', 'run_TRAIN_postcomplete_',
              'run_TRAIN_zero_success_')
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) in own:
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] in ('Z', 'X'):
                continue
            tokens = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            matches = [Path(t).name for t in tokens if '\n' not in t
                       and Path(t).name.endswith('.py') and Path(t).name.startswith(extras)]
            if matches and Path(os.readlink(entry / 'cwd')).resolve() == Path(root):
                snapshot['live_project_native_processes'].append(
                    dict(pid=int(entry.name), start_ticks=int(fields[19]), scripts=matches))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, ValueError, IndexError) as error:
            snapshot['same_uid_inventory_incomplete'] = True
            snapshot['unresolved_processes'].append(dict(pid=int(entry.name), reason=type(error).__name__))
    return snapshot


@contextmanager
def reserved_resources_after_TEST(root, *, enabled=False):
    require(enabled is True, 'explicit enabled=True required')
    root = Path(root).resolve()
    legacy = _legacy(root)
    _stop(root)
    with _reserve(root):
        _stop(root)
        proof = require_complete_TEST_metadata(root)
        require(not any(legacy._is_alive(p) for p in proof['known_processes']), 'old actual owner/auditor still live')
        snapshot = _inventory(root, legacy)
        require(snapshot.get('host_visibility_available') is True
                and snapshot.get('ancestor_chain_complete') is True
                and snapshot.get('same_uid_inventory_incomplete') is False
                and snapshot.get('live_project_native_processes') == [], 'empty host-visible native inventory required')
        proof['host_inventory'] = snapshot
        _stop(root)
        permit = legacy._Permit()
        legacy._LEASES[permit] = dict(pid=os.getpid(), root=root, proof=deepcopy(proof), stop=_stop, test_only=False)
        try:
            yield permit
        finally:
            legacy._LEASES.pop(permit, None)
