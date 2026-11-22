"""Explicit parent-owned global lease and process-isolated child slots.

This module grants resource capabilities only. It neither admits a scientific
split nor changes a physical policy. Frozen1106 owns the global locks for the
whole run; each child gets its own canonical924 same-process capability. No
historical concurrency qualification is reused as a current measurement.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import secrets
from weakref import WeakKeyDictionary

SOURCE = 'our_mani/multitask/parallel_repaired_resource_guard_v1171.py'
PARENT_SOURCE = 'our_mani/multitask/post_completed_TEST_resource_guard_v1106.py'
BASE = 'runtime/framework/parallel_repaired_owner_v1173'
ART = 'artifacts/framework/multitask_r10/parallel_repaired_policy_v1172'
CASE_RUNTIME = 'runtime/framework/parallel_repaired_policy_v1172'
SCHEMA = 'r10_parent_owned_parallel_resource_session_v1171'
CLAIM_SCHEMA = 'r10_once_only_parallel_resource_claim_v1171'
RECEIPT_SCHEMA = 'r10_parallel_child_resource_receipt_v1171'
PROFILE_CEILING = dict(max_native_slots=12, max_GPU_slots=5,
    max_CPU_preflight_slots=6, ready_limit=24)
JOB_KEYS = {'job_id', 'task', 'method', 'raw_episode_id', 'split',
    'configuration', 'out', 'runtime', 'source_row_sha256'}
HASHES = ('physical_policy_sha256', 'control_graph_sha256', 'policy_sha256')
_ACTIVE = WeakKeyDictionary()
_RECEIPTS = WeakKeyDictionary()


def require(condition, message):
    if not condition:
        raise ValueError('parallel1171: ' + message)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()


def _same(a, b):
    return _bytes(a) == _bytes(b)


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _path(root, name):
    require(type(name) is str and name and not Path(name).is_absolute()
        and Path(name).as_posix() == name and '..' not in Path(name).parts
        and '\\' not in name, 'canonical relative path required')
    result = Path(root) / name
    require(result.resolve(strict=False) == result, 'symlink/alias path refused')
    return result


def _pointer(value):
    require(type(value) is dict and set(value) <= {'root', 'path', 'sha256'}
        and {'path', 'sha256'} <= set(value) and value.get('root', 'project') == 'project'
        and _sha(value['sha256']), 'project metadata pointer required')
    return dict(root='project', path=value['path'], sha256=value['sha256'])


def _stamp(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _read(root, pointer, *, prefix, basename):
    pointer = _pointer(pointer)
    require(type(pointer['path']) is str and pointer['path'].startswith(prefix)
        and Path(pointer['path']).name == basename, 'typed metadata domain required before read')
    file = _path(root, pointer['path'])
    before = _stamp(file.stat())
    data = file.read_bytes()
    require(before == _stamp(file.stat()) and hashlib.sha256(data).hexdigest() == pointer['sha256'],
        'immutable metadata bytes changed: ' + pointer['path'])
    def invalid(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    result = json.loads(data, parse_constant=invalid)
    require(type(result) is dict, 'metadata object required')
    return result


def _fresh(root, name, value):
    file = _path(root, name)
    file.parent.mkdir(parents=True, exist_ok=True)
    data = _bytes(value) + b'\n'
    with file.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(root='project', path=name, sha256=hashlib.sha256(data).hexdigest())


def _profile(value):
    require(type(value) is dict and set(value) == set(PROFILE_CEILING),
        'explicit four-field bounded profile required')
    for key, ceiling in PROFILE_CEILING.items():
        require(type(value[key]) is int and 0 < value[key] <= ceiling,
            'resource profile ceiling exceeded: ' + key)
    require(value['max_GPU_slots'] <= value['max_native_slots'],
        'GPU slots cannot exceed Native slots')
    return deepcopy(value)


def _process(pid=None):
    pid = os.getpid() if pid is None else pid
    require(type(pid) is int and pid > 0, 'positive actual PID required')
    fields = Path('/proc/' + str(pid) + '/stat').read_text().rsplit(')', 1)[1].split()
    require(fields[0] not in ('Z', 'X'), 'live process required')
    return dict(pid=pid, start_ticks=int(fields[19]))


def _alive(process):
    try:
        return _process(process['pid']) == process
    except (OSError, ValueError, IndexError):
        return False


def _boot():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def _descendant(owner):
    current = os.getpid()
    require(current != owner['pid'], 'child lease must be in a descendant process')
    seen = set()
    while current > 1 and current not in seen:
        seen.add(current)
        fields = Path('/proc/' + str(current) + '/stat').read_text().rsplit(')', 1)[1].split()
        current = int(fields[1])
        if current == owner['pid']:
            return _process(current) == owner
    return False


def _other_controllers(root):
    """Close the serial owner's gap between its separate child leases."""
    found = []
    prefixes = ('run_repaired_policy_owner_v', 'run_parallel_repaired_owner_v')
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            before = _process(int(entry.name))
            tokens = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')
            matched = [Path(t).name for t in tokens if '\n' not in t
                and Path(t).name.endswith('.py') and Path(t).name.startswith(prefixes)]
            if matched and Path(os.readlink(entry / 'cwd')).resolve() == root:
                require(_process(int(entry.name)) == before, 'controller PID reused during inventory')
                found.append(dict(**before, scripts=matched))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except ValueError:
            # Zombies have no resource capability; all other parsing errors fail closed.
            try:
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                if fields[0] in ('Z', 'X'):
                    continue
            except (FileNotFoundError, ProcessLookupError):
                continue
            raise
    return found


def _parent(root):
    module = importlib.import_module('our_mani.multitask.post_completed_TEST_resource_guard_v1106')
    require(Path(module.__file__).resolve() == root / PARENT_SOURCE,
        'canonical frozen1106 implementation required')
    return module


def _source(root, relative):
    file = _path(root, relative)
    return dict(root='project', path=relative, sha256=hashlib.sha256(file.read_bytes()).hexdigest())


def _graph(root, graph_ref):
    graph = _read(root, graph_ref, prefix='artifacts/framework/multitask_r10/', basename='policy_graph.json')
    require(graph.get('schema') == 'r10_frozen_repaired_policy_graph_v1143'
        and graph.get('roots', {}).get('project') == str(root)
        and all(_sha(graph.get(key)) for key in HASHES), 'actual typed policy graph required')
    require(Path(__file__).resolve() == root / SOURCE, 'actual imported1171 source differs')
    source = _source(root, SOURCE)
    control = graph.get('control', {})
    require(_same(control.get('components', {}).get('resource_guard'), source),
        'graph must explicitly declare the new1171 resource guard')
    values = list(control.get('sources_map', {}).values())
    require(source in values and _source(root, PARENT_SOURCE) in values,
        'new1171 and frozen parent1106 must both be source-bound')
    return graph


def _runtime(root, runtime, split):
    if isinstance(runtime, Path):
        runtime = str(runtime.relative_to(root)) if runtime.is_absolute() else runtime.as_posix()
    file = _path(root, runtime)
    require(split in ('train', 'validation') and file.parent == root / BASE / split
        and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*', file.name),
        'fresh explicit parallel owner namespace required')
    return file.relative_to(root).as_posix()


def _job(root, job, split, graph):
    require(type(job) is dict and set(job) == JOB_KEYS and job.get('split') == split,
        'exact declared job fields/split required; TEST refused')
    require(type(job['job_id']) is str and re.fullmatch('j[0-9]{4}', job['job_id']), 'fixed job ID required')
    require(job['method'] in ('cv_geometry', 'context_geometry', 'physmani')
        and job['task'] in graph.get('tasks', []) and type(job['raw_episode_id']) is str
        and re.fullmatch('episode[0-9]+', job['raw_episode_id']) and _sha(job['source_row_sha256']),
        'explicit task/method/source identity required')
    configuration = _pointer(job['configuration'])
    routes = [row for row in graph.get('routes', [])
        if row.get('task') == job['task'] and row.get('method') == job['method']]
    require(len(routes) == 1 and _same(routes[0].get('configuration'), configuration),
        'job configuration differs from exact graph route')
    for key, base in (('out', ART), ('runtime', CASE_RUNTIME)):
        file = _path(root, job[key])
        require(file.is_relative_to(root / base / split) and file != root / base / split,
            'case path outside actual parallel worker namespace')
    if split == 'train':
        require('development_demo_only' in Path(job['out']).parts, 'TRAIN development namespace required')
    else:
        require('development_demo_only' not in Path(job['out']).parts, 'validation cannot use TRAIN namespace')
    return deepcopy(job)


def _flock_rows(pid, file):
    stat = file.stat()
    expected = (os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino)
    rows = []
    for line in Path('/proc/locks').read_text().splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[1:4] != ['FLOCK', 'ADVISORY', 'WRITE']:
            continue
        try:
            device = parts[5].split(':')
            actual = (int(device[0], 16), int(device[1], 16), int(device[2]))
            if int(parts[4]) == pid and actual == expected and parts[6:8] == ['0', 'EOF']:
                rows.append(line)
        except (ValueError, IndexError):
            continue
    return rows


def _lock_proof(root, owner, names):
    rows = []
    descriptors = list(Path('/proc/' + str(owner['pid']) + '/fd').iterdir())
    for name in names:
        file = _path(root, name)
        stat = file.stat()
        fds = []
        for fd in descriptors:
            try:
                actual = fd.stat()
                if (actual.st_dev, actual.st_ino) == (stat.st_dev, stat.st_ino):
                    fds.append(int(fd.name))
            except FileNotFoundError:
                continue
        require(fds and _flock_rows(owner['pid'], file), 'actual owner FLOCK/FD missing: ' + name)
        rows.append(dict(path=name, device=stat.st_dev, inode=stat.st_ino, fd=min(fds)))
    return rows


def _check_locks(root, owner, proof):
    require(type(proof) is list and proof, 'actual parent lock proof required')
    for row in proof:
        file = _path(root, row['path'])
        try:
            stat = file.stat()
            fd = Path('/proc/' + str(owner['pid']) + '/fd/' + str(row['fd']))
            opened = fd.stat()
        except OSError as error:
            raise ValueError('parallel1171: actual parent/slot FLOCK descriptor disappeared') from error
        require((stat.st_dev, stat.st_ino) == (row['device'], row['inode'])
            == (opened.st_dev, opened.st_ino) and _flock_rows(owner['pid'], file),
            'parent no longer holds its actual global FLOCK: ' + row['path'])


def _stop(root, session):
    _parent(root)._stop(root)
    require(not (_path(root, session['runtime']) / 'STOP').exists(), 'parallel owner STOP blocks a new capability')


@contextmanager
def owner_scope(root, *, graph_ref, plan_ref, profile, runtime, enabled=False):
    require(enabled is True, 'explicit owner enabled=True required')
    root = Path(root).resolve()
    profile = _profile(profile)
    graph = _graph(root, graph_ref)
    plan = _read(root, plan_ref, prefix=BASE + '/', basename='plan.json')
    split = plan.get('split')
    runtime = _runtime(root, runtime, split)
    require(_pointer(plan_ref)['path'] == runtime + '/plan.json'
        and _same(plan.get('graph'), _pointer(graph_ref))
        and all(plan.get(key) == graph[key] for key in HASHES), 'owner plan/graph binding differs')
    require(_same(plan.get('profile'), profile), 'owner plan resource profile differs')
    jobs = [_job(root, row, split, graph) for row in plan.get('jobs', [])]
    require(jobs and len({row['job_id'] for row in jobs}) == len(jobs)
        and len({row['out'] for row in jobs}) == len(jobs)
        and len({row['runtime'] for row in jobs}) == len(jobs), 'unique complete declared job inventory required')
    resources = runtime + '/resources'
    require(not _path(root, resources).exists(), 'resource namespace already used; no restart or retry')
    require(_other_controllers(root) == [], 'prior serial/parallel owner must actually exit first')
    parent = _parent(root)
    with parent.reserved_resources_after_TEST(root, enabled=True) as parent_permit:
        require(_other_controllers(root) == [], 'another queue owner is still live')
        parent_proof = parent.require_permit(parent_permit, root)
        owner = _process()
        lock_proof = _lock_proof(root, owner, parent.LOCKS)
        for kind, count in (('CPU_check', profile['max_CPU_preflight_slots']),
                ('native', profile['max_native_slots']), ('GPU', profile['max_GPU_slots'])):
            for slot in range(count):
                file = _path(root, resources + '/locks/' + kind + '_' + str(slot) + '.lock')
                file.parent.mkdir(parents=True, exist_ok=True)
                with file.open('xb'):
                    pass
        session = dict(schema=SCHEMA, created_utc=_utc(), root=str(root), owner=owner,
            boot_id=_boot(), graph=_pointer(graph_ref), plan=_pointer(plan_ref),
            **{key: graph[key] for key in HASHES}, profile=profile, runtime=runtime,
            split=split, jobs=jobs, random_token=secrets.token_hex(32),
            parent_global_locks=lock_proof, parent_resource_proof=deepcopy(parent_proof),
            guard=_source(root, SOURCE), parent_guard=_source(root, PARENT_SOURCE),
            test_only=parent_proof.get('test_only') is True,
            resource_only=True, scientific_admission_granted=False,
            historical_concurrency_qualification_reused=False)
        session_ref = _fresh(root, resources + '/session.json', session)
        context = dict(session=session_ref, profile=deepcopy(profile),
            parent_resource_proof=deepcopy(parent_proof), process=owner,
            global_locks_held_by_actual_parent=True)
        _stop(root, session)
        owner_error = None
        try:
            yield context
        except BaseException as error:
            owner_error = error
            raise
        finally:
            # A completed parent may not release global locks with an owned child alive.
            # The scheduler must join all children before leaving this scope.
            live = []
            claims = _path(root, resources + '/claims')
            if claims.exists():
                for file in claims.glob('*/claim.json'):
                    file = _path(root, file.relative_to(root).as_posix())
                    claim = json.loads(file.read_bytes())
                    terminal = file.parent / 'terminal.json'
                    if not terminal.exists() or (claim['process'] != owner and _alive(claim['process'])):
                        live.append(claim['process'])
            if owner_error is None:
                require(not live, 'unclosed child resource claims at parent exit')
                _check_locks(root, owner, lock_proof)


def read_session(root, session_ref, *, existing_claim=False):
    root = Path(root).resolve()
    session = _read(root, session_ref, prefix=BASE + '/', basename='session.json')
    require(session.get('schema') == SCHEMA and session.get('root') == str(root)
        and session.get('resource_only') is True and session.get('scientific_admission_granted') is False,
        'actual resource-only1171 session required')
    require(session.get('split') in ('train', 'validation'), 'TEST session refused')
    runtime = _runtime(root, session['runtime'], session['split'])
    require(_pointer(session_ref)['path'] == runtime + '/resources/session.json', 'exact session path required')
    require(type(session.get('random_token')) is str and re.fullmatch('[0-9a-f]{64}', session['random_token']),
        'fresh session token required')
    require(_boot() == session['boot_id'] and _process(session['owner']['pid']) == session['owner'],
        'actual parent identity/boot changed')
    require(_descendant(session['owner']), 'actual parent must remain in child ancestry')
    _profile(session['profile'])
    graph = _graph(root, session['graph'])
    plan = _read(root, session['plan'], prefix=BASE + '/', basename='plan.json')
    require(session['plan']['path'] == runtime + '/plan.json'
        and _same(plan.get('graph'), session['graph']) and plan.get('split') == session['split']
        and _same(plan.get('profile'), session['profile']) and _same(plan.get('jobs'), session['jobs'])
        and all(session.get(key) == graph[key] == plan.get(key) for key in HASHES),
        'immutable graph/plan/session binding differs')
    require(_same(session.get('guard'), _source(root, SOURCE))
        and _same(session.get('parent_guard'), _source(root, PARENT_SOURCE)), 'actual guard bytes changed')
    for row in session['jobs']:
        _job(root, row, session['split'], graph)
    parent = _parent(root)
    require([row['path'] for row in session['parent_global_locks']] == list(parent.LOCKS),
        'complete actual1106 parent lock set required')
    _check_locks(root, session['owner'], session['parent_global_locks'])
    # Administrative STOP closes admission; an already claimed case drains.
    if not existing_claim:
        _stop(root, session)
    return session


def _slot(root, session, kind, slot, count):
    require(type(slot) is int and 0 <= slot < count, 'bounded zero-based ' + kind + ' slot required')
    file = _path(root, session['runtime'] + '/resources/locks/' + kind + '_' + str(slot) + '.lock')
    require(file.is_file(), 'parent-created slot lock required')
    stream = file.open('r+b')
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        stream.close()
        raise
    return stream


def require_permit(permit, root, *, allow_test_callbacks=False):
    root = Path(root).resolve()
    require(permit in _ACTIVE, 'active1171 child permit required')
    lease = _ACTIVE[permit]
    require(lease['process'] == _process() and lease['root'] == root,
        'same-process1171 capability required')
    legacy = _parent(root)._legacy(root)
    legacy.require_permit(permit, root, allow_test_callbacks=allow_test_callbacks)
    if lease['kind'] == 'metadata_only':
        _check_metadata(root, lease)
        return deepcopy(lease['proof'])
    session = read_session(root, lease['session'], existing_claim=True)
    require(session['random_token'] == lease['token'], 'child session token changed')
    _check_locks(root, lease['process'], lease['slot_locks'])
    return deepcopy(lease['proof'])


def permit_receipt(permit):
    require(permit in _RECEIPTS, 'actual owned1171 receipt required')
    return deepcopy(_RECEIPTS[permit])


def require_native_permit(permit, root):
    proof = require_permit(permit, root)
    require(proof.get('purpose') == 'child_slot' and proof.get('phase') == 'native'
        and proof.get('Native_authorized') is True, 'actual Native child-slot capability required')
    return proof


def _check_metadata(root, lease):
    require(lease['kind'] == 'metadata_only' and lease['process'] == _process()
        and lease['boot_id'] == _boot(), 'same-process live metadata capability required')
    _check_locks(root, lease['process'], lease['slot_locks'])


@contextmanager
def metadata_scope(root, *, enabled=False):
    """Short exclusive metadata capability; never a Native or child-slot lease."""
    require(enabled is True, 'explicit metadata enabled=True required')
    root = Path(root).resolve()
    require(_other_controllers(root) == [], 'prior serial/parallel owner must actually exit first')
    parent = _parent(root)
    with parent.reserved_resources_after_TEST(root, enabled=True) as parent_permit:
        require(_other_controllers(root) == [], 'another queue owner is still live')
        parent_proof = parent.require_permit(parent_permit, root)
        process = _process()
        locks = _lock_proof(root, process, parent.LOCKS)
        legacy = parent._legacy(root)
        permit = legacy._Permit()
        proof = dict(schema=RECEIPT_SCHEMA, purpose='metadata_only', process=process,
            boot_id=_boot(), parent_resource_proof=deepcopy(parent_proof),
            parent_global_locks_verified=True, Native_authorized=False,
            child_slots_authorized=False, resource_only=True,
            scientific_admission_granted=False, test_only=parent_proof.get('test_only') is True,
            body_completed=False, restored=False, errors=[])
        lease = dict(kind='metadata_only', root=root, process=process, boot_id=proof['boot_id'],
            slot_locks=locks, proof=proof)
        legacy._LEASES[permit] = dict(pid=process['pid'], root=root, proof=deepcopy(proof),
            stop=lambda owned_root: _check_metadata(owned_root, lease), test_only=proof['test_only'])
        _ACTIVE[permit] = lease
        _RECEIPTS[permit] = proof
        error = None
        try:
            yield permit
        except BaseException as exc:
            error = exc
            raise
        finally:
            errors = []
            try:
                _check_metadata(root, lease)
            except BaseException as exc:
                errors.append(dict(stage='metadata_final_check', type=type(exc).__name__, message=str(exc)))
            legacy._LEASES.pop(permit, None)
            _ACTIVE.pop(permit, None)
            proof.update(body_completed=error is None and not errors,
                original_exception_type=None if error is None else type(error).__name__,
                restored=True, leases_removed=True, errors=errors, finished_utc=_utc())
            _RECEIPTS[permit] = deepcopy(proof)
            if errors and error is None:
                raise RuntimeError('parallel1171: metadata resource finalization failed: ' + str(errors))


@contextmanager
def child_scope(root, *, session_ref, job, phase, slot, gpu_slot=None, enabled=False):
    require(enabled is True, 'explicit child enabled=True required')
    require(phase in ('CPU_check', 'native'), 'explicit CPU_check/native phase required')
    root = Path(root).resolve()
    session_ref = _pointer(session_ref)
    session = read_session(root, session_ref)
    matches = [row for row in session['jobs'] if row.get('job_id') == job.get('job_id')]
    require(len(matches) == 1 and _same(matches[0], job), 'child must exactly match one declared job')
    require(not (phase == 'native' and session.get('test_only') is True), 'CPU fake resource proof cannot authorize Native')
    uses_GPU = phase == 'native' and job['method'] == 'physmani'
    require(uses_GPU == (gpu_slot is not None), 'GPU slot required only for actual native PhysMani')
    resources = session['runtime'] + '/resources'
    claim_dir = resources + '/claims/' + job['job_id'] + '_' + phase
    if phase == 'native':
        terminal = _path(root, resources + '/claims/' + job['job_id'] + '_CPU_check/terminal.json')
        require(terminal.is_file(), 'same job CPU resource scope must close before Native')
        previous = json.loads(terminal.read_bytes())
        require(previous.get('schema') == RECEIPT_SCHEMA and previous.get('body_completed') is True
            and previous.get('restored') is True and previous.get('errors') == []
            and previous.get('phase') == 'CPU_check'
            and _same(previous.get('session'), session_ref) and previous.get('job_id') == job['job_id'],
            'clean same-session CPU resource scope required; actual CPU quality remains owner responsibility')
    process = _process()
    original_error = None
    original_traceback = None
    errors = []
    parent = _parent(root)
    legacy = parent._legacy(root)
    permit = None
    with ExitStack() as slots:
        count = session['profile']['max_CPU_preflight_slots' if phase == 'CPU_check' else 'max_native_slots']
        opened = [slots.enter_context(_slot(root, session, phase, slot, count))]
        names = [resources + '/locks/' + phase + '_' + str(slot) + '.lock']
        if uses_GPU:
            opened.append(slots.enter_context(_slot(root, session, 'GPU', gpu_slot, session['profile']['max_GPU_slots'])))
            names.append(resources + '/locks/GPU_' + str(gpu_slot) + '.lock')
        read_session(root, session_ref)
        claim = dict(schema=CLAIM_SCHEMA, created_utc=_utc(), session=session_ref,
            job=deepcopy(job), phase=phase, slot=slot, gpu_slot=gpu_slot, process=process,
            automatic_retry=False, test_only=session.get('test_only') is True)
        claim_ref = _fresh(root, claim_dir + '/claim.json', claim)
        lock_proof = _lock_proof(root, process, names)
        proof = dict(schema=RECEIPT_SCHEMA, session=session_ref, claim=claim_ref,
            actual_parent=session['owner'], parent_global_locks_verified=True,
            parent1106_empty_inventory_checked_at_session_start_only=True,
            child_empty_inventory_claimed=False, graph=session['graph'],
            **{key: session[key] for key in HASHES}, profile=session['profile'],
            job_id=job['job_id'], phase=phase, slot=slot, gpu_slot=gpu_slot,
            process=process, test_only=session.get('test_only') is True,
            purpose='child_slot', Native_authorized=phase == 'native', child_slots_authorized=True,
            scientific_admission_granted=False, resource_only=True,
            body_completed=False, restored=False, errors=[])
        permit = legacy._Permit()
        lease = dict(pid=process['pid'], root=root, proof=deepcopy(proof),
            stop=lambda owned_root: read_session(owned_root, session_ref, existing_claim=True),
            test_only=session.get('test_only') is True)
        legacy._LEASES[permit] = lease
        _ACTIVE[permit] = dict(kind='child_slot', root=root, process=process, session=session_ref,
            token=session['random_token'], proof=proof, slot_locks=lock_proof)
        _RECEIPTS[permit] = proof
        try:
            yield permit
        except BaseException as error:
            original_error = error
            original_traceback = error.__traceback__
        finally:
            try:
                read_session(root, session_ref, existing_claim=True)
                _check_locks(root, process, lock_proof)
            except BaseException as error:
                errors.append(dict(stage='parent_or_slot_final_check', type=type(error).__name__, message=str(error)))
            legacy._LEASES.pop(permit, None)
            _ACTIVE.pop(permit, None)
            proof.update(body_completed=original_error is None and not errors,
                original_exception_type=None if original_error is None else type(original_error).__name__,
                leases_removed=True, restored=False, errors=errors, finished_utc=_utc())
    # All slot descriptors are closed before the terminal receipt claims restoration.
    proof['restored'] = True
    try:
        _fresh(root, claim_dir + '/terminal.json', proof)
    except BaseException as error:
        errors.append(dict(stage='terminal_write', type=type(error).__name__, message=str(error)))
    _RECEIPTS[permit] = deepcopy(proof)
    if original_error is not None:
        raise original_error.with_traceback(original_traceback)
    if errors and original_error is None:
        raise RuntimeError('parallel1171: child resource finalization failed: ' + str(errors))
