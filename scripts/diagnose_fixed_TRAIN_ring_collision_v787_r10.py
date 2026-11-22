"""Unenabled standalone fixed-TRAIN ring diagnostic; default is CPU check only.

Never queue this entry in the formal controller.  A future independent process
may run ONE original fixed-four TRAIN source, after 775/772 and other native
owners have exited.  --execute-native additionally requires an explicit small
diagnostic source lock, empty CUDA_VISIBLE_DEVICES and an available X11 display.
Use the existing run_local_sim.sh environment only after resources are free.

All physical/model/source checks are the original742/743/744 checks.  The only
new hook wraps744._geometry after its actor and public ghosts exist, installing
783 before the first policy observation/action.  No native target pose is read.
This file is not referenced by the formal chain and creates no request/queue.
Native operation of this candidate has NOT been validated.
"""
import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCHEMA = 'r10_standalone_development_TRAIN_ring_collision_v787'
INPUT_SCHEMA = 'r10_standalone_TRAIN_public_collision_inputs_v787'
WORKER = 'scripts/run_task_executor_smoke_author_outcome_v743_r10.py'
OUTPUT = 'artifacts/framework/multitask_r10/train_public_collision_diagnostic_v787/development_demo_only'
TRAIN_LOCK = 'artifacts/framework/multitask_r10/author_outcome_qualification_v745_setup/frozen_inputs.json'
CONFIG_BASE = 'artifacts/framework/multitask_r10/post_training_qualification_v680_setup'
DIAGNOSTIC_LOCK = 'artifacts/framework/multitask_r10/train_public_collision_diagnostic_v787_setup/diagnostic_inputs.json'
SOURCES = (
    'scripts/diagnose_fixed_TRAIN_ring_collision_v787_r10.py',
    'our_mani/multitask/train_public_collision_diagnostics_v783.py',
    'tests/test_fixed_TRAIN_ring_collision_entry_v787.py',
    'tests/test_train_public_collision_diagnostics_v783.py',
)
LOCKS = ('runtime/framework/ordered_resume_VAL_TEST_v775.lock',
         'runtime/framework/fixed_TRAIN104_geometry_native.lock')
RUNTIMES = ('runtime/framework/ordered_resume_VAL_TEST_v775',
            'runtime/framework/formal_VAL_resume_v772',
            'runtime/framework/formal_TEST_after_resume_v753')
_RESOURCE_TOKEN = object()


def utc(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'manifest', 'frozen-inputs', 'raw-episode-id', 'out'):
        result.add_argument('--'+name, required=True)
    result.add_argument('--task', required=True, choices=('insert_onto_rotating_peg', 'insert_onto_rotating_peg_high_speed'))
    result.add_argument('--method', required=True, choices=('cv_geometry', 'context_geometry'))
    result.add_argument('--split', required=True, choices=('train',))
    result.add_argument('--development-only', required=True, action='store_true')
    mode = result.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Default: CPU preparation only, no simulator')
    mode.add_argument('--execute-native', action='store_true', help='Explicit future standalone TRAIN execution')
    result.add_argument('--diagnostic-inputs', help='Required for native execution; freeze only the four diagnostic sources')
    result.add_argument('--max-events', type=int, default=10000)
    return result


def validate_request(root, args):
    root = Path(root).resolve()
    if (args.split != 'train' or args.development_only is not True
            or args.task not in ('insert_onto_rotating_peg', 'insert_onto_rotating_peg_high_speed')
            or args.method not in ('cv_geometry', 'context_geometry')):
        raise ValueError('standalone development-only fixed TRAIN ring CV/Context diagnostic required')
    if args.max_events < 1:
        raise ValueError('positive max-events required')
    if (root/args.manifest).resolve() != root/'artifacts/framework/protocol/train_manifest.json':
        raise ValueError('exact original TRAIN manifest only; no VAL/TEST source')
    if ((root/args.config).resolve() != root/CONFIG_BASE/(args.task+'_ours_configuration.json')
            or (root/args.frozen_inputs).resolve() != root/TRAIN_LOCK):
        raise ValueError('exact frozen680 ring configuration and745 TRAIN input lock required')
    out = (root/args.out).resolve()
    base = root/OUTPUT
    if not out.is_relative_to(base) or out == base or out.exists():
        raise ValueError('fresh isolated v787 development output required; no reuse/overwrite')
    if args.execute_native and not args.diagnostic_inputs:
        raise ValueError('native opt-in also requires an explicit diagnostic source lock')
    return out


def _process(record, proc=Path('/proc')):
    if (type(record.get('pid')) is not int or type(record.get('start_ticks')) is not int
            or record['pid'] < 1 or record['start_ticks'] < 1):
        raise RuntimeError('invalid scheduler PID/start identity; fail closed')
    path = Path(proc)/str(record['pid'])/'stat'
    try:
        fields = path.read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[0] not in ('Z', 'X') and int(fields[19]) == record['start_ticks']


def resource_snapshot(root, proc=Path('/proc')):
    """Read process/launch metadata only, including stopped schedulers as live."""
    root, proc = Path(root).resolve(), Path(proc)
    if (proc/'1/comm').read_text().strip() not in ('systemd', 'init'):
        raise RuntimeError('host process visibility unavailable; diagnostic remains unlaunched')
    known = []
    for index, runtime in enumerate(RUNTIMES):
        path = root/runtime/'launch.json'
        if not path.exists():
            if index < 2:
                raise RuntimeError('cannot establish original775/772 scheduler identity: '+runtime)
            continue
        record = json.loads(path.read_text())
        live = _process(record, proc)
        known.append(dict(runtime=runtime, pid=record['pid'], start_ticks=record['start_ticks'], live=live))
        if live:
            raise RuntimeError('formal scheduler/owner still live; fail closed: '+runtime)
        if not (path.parent/'terminal.json').is_file():
            raise RuntimeError('formal scheduler/owner lacks terminal evidence: '+runtime)
    prefixes = ('run_task_executor_smoke_', 'run_formal_case_', 'run_formal_suite16_owner_',
        'run_resumed_formal_VAL_owner_', 'continue_resumed_VAL_TEST_',
        'continue_author_outcome_VAL_TEST_',
        'run_native_', 'run_geometry_', 'collect_multitask_', 'evaluate_multitask_case_',
        'run_PhysMani_task_', 'run_continuous_reference_pair_', 'launch_multitask_',
        'run_ready_candidate_', 'run_multitask_fresh_queue', 'run_trained_task_', 'run_task_formal_')
    live_native = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            before = (entry/'stat').read_text().rsplit(')', 1)[1].split()
            tokens = (entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
            matches = [Path(token).name for token in tokens if '\n' not in token
                and Path(token).name.endswith('.py') and Path(token).name.startswith(prefixes)]
            if matches and (entry/'cwd').resolve() == root:
                after = (entry/'stat').read_text().rsplit(')', 1)[1].split()
                if before[19] == after[19] and after[0] not in ('Z', 'X'):
                    live_native.append(dict(pid=int(entry.name), start_ticks=int(after[19]), scripts=matches))
        except FileNotFoundError:
            continue
        except PermissionError as error:
            try:
                same_uid = entry.stat().st_uid == os.getuid()
            except FileNotFoundError:
                continue
            if same_uid:
                raise RuntimeError('incomplete same-UID process inventory; fail closed') from error
    if live_native:
        raise RuntimeError('existing project native/controller processes prevent diagnostic: '+repr(live_native))
    return dict(host_visibility_available=True, known_formal_processes=known,
        live_project_native_processes=[], same_uid_inventory_incomplete=False)


def _stop_guard(root):
    paths = [Path(root)/'runtime/framework/multitask_r10_STOP',
             *[Path(root)/name/'STOP' for name in RUNTIMES]]
    if any(path.exists() for path in paths):
        raise InterruptedError('existing global/formal STOP prevents a new diagnostic')


@dataclass
class _ResourcePermit:
    root: Path
    pid: int
    inventory: dict
    _token: object
    active: bool = True


@contextmanager
def reserved_resources(root):
    """Hold the SAME kernel locks throughout preparation/native cleanup."""
    root = Path(root).resolve()
    _stop_guard(root)
    permit = None
    with ExitStack() as stack:
        for name in LOCKS:
            stream = stack.enter_context((root/name).open('a+'))
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError('formal controller/global native resource lock busy: '+name) from error
        _stop_guard(root)
        inventory = resource_snapshot(root)
        permit = _ResourcePermit(root, os.getpid(), inventory, _RESOURCE_TOKEN)
        try:
            yield permit
        finally:
            permit.active = False


def require_permit(permit, root):
    if (not isinstance(permit, _ResourcePermit) or permit._token is not _RESOURCE_TOKEN
            or not permit.active or permit.pid != os.getpid() or permit.root != Path(root).resolve()):
        raise RuntimeError('live standalone global resource reservation required')
    _stop_guard(root)


def source_fingerprints(root):
    return {str((Path(root)/name).resolve()):sha(Path(root)/name) for name in SOURCES}


def verify_diagnostic_inputs(root, path):
    root, path = Path(root).resolve(), (Path(root)/path).resolve()
    if path != root/DIAGNOSTIC_LOCK:
        raise ValueError('isolated exact v787 diagnostic source lock required')
    document = json.loads(path.read_text())
    expected = source_fingerprints(root)
    if (document.get('schema') != INPUT_SCHEMA or document.get('split') != 'train'
            or document.get('approved_for_development_only') is not True
            or document.get('files') != expected):
        raise ValueError('exact four-source development-only diagnostic lock required')
    return dict(path=str(path.relative_to(root)), sha256=sha(path), files=expected)


def prepare(root, args, out, worker=None):
    # Importing this entry and --help require only stdlib.  This import/preflight
    # is still CPU-only and uses the actual742 gate with frozen743 identity.
    if worker is None:
        from scripts import run_task_executor_smoke_author_outcome_v743_r10 as worker
    original_args = SimpleNamespace(config=args.config, manifest=args.manifest,
        frozen_inputs=args.frozen_inputs, task=args.task, raw_episode_id=args.raw_episode_id,
        method=args.method, split='train', out=str((Path(out)/'case').relative_to(root)),
        replica=0, exploration_seed=None, development_demo=True, record_demo=False,
        demo_only=False, check=not args.execute_native, suite=None, suite_lock=None,
        validation_release=None, source_inputs=None)
    prepared = worker.contract.prepare(Path(root), original_args, Path(worker.__file__))
    from our_mani.multitask.rml_component_domain_v640 import ComponentDomain
    domain = prepared['component_domain']
    if (not isinstance(domain, ComponentDomain) or domain.require(args.task, args.method) != 'train_qualification'
            or domain.split != 'train' or prepared.get('formal') is not False):
        raise ValueError('original fixed-TRAIN sealed capability required before diagnostics')
    return prepared, worker


@contextmanager
def original_cpu_scopes():
    # Match743.__main__ for BOTH CPU preparation and execution.  These import
    # CPU validation code only; no native environment is constructed here.
    from our_mani.multitask.frozen_input_edges_v463 import evaluator_scope, declared_edge_scope
    from scripts import evaluate_multitask_case_r10 as evaluator
    with declared_edge_scope(), evaluator_scope(evaluator):
        yield


def collection_snapshot(env):
    """Never substitute an arm tree for collection membership.

    Installed PyRep Arm has check_arm_collision and a collection handle, but
    backend.sim exposes no get-collection-members/overrideProperties wrapper.
    Do not add CFFI, Lua, scene serialization, or a new collision predicate.
    get_grasped_objects is the existing gripper's cached attachment list.
    """
    return dict(collection_handle=getattr(env.arm, '_collision_collection', None),
        membership_status='unknown_no_installed_PyRep_read_only_collection_inventory_API',
        collection_members=None, member_names=None, member_raw_flags=None,
        overrideProperties=None, member_robot_or_native_attachment=None,
        arm_tree_is_not_assumed_collection=True,
        cached_native_grasped_handles=[int(obj.get_handle()) for obj in env.gripper.get_grasped_objects()],
        native_attachment_pose_read=False, native_target_pose_read=False)


def _atomic(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name+'.', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_prepared(root, prepared, worker, out, permit, *, diagnostic_lock, max_events=10000):
    """Internal one-case path; caller must already own the live resource permit."""
    root, out = Path(root).resolve(), Path(out).resolve()
    require_permit(permit, root)
    # Revalidate explicit supplement at actual execution boundary.
    verified = verify_diagnostic_inputs(root, diagnostic_lock['path'])
    if verified != diagnostic_lock:
        raise ValueError('diagnostic lock changed after CPU preparation')
    from our_mani.multitask.train_public_collision_diagnostics_v783 import TrainPublicCollisionDiagnostics
    from our_mani.multitask.rml_component_domain_v640 import ComponentDomain
    domain = prepared['component_domain']
    if (not isinstance(domain, ComponentDomain)
            or domain.require(prepared['args'].task, prepared['args'].method) != 'train_qualification'
            or domain.split != 'train' or prepared.get('formal') is not False
            or prepared['args'].split != 'train' or prepared['args'].method not in ('cv_geometry','context_geometry')
            or prepared['args'].task not in ('insert_onto_rotating_peg','insert_onto_rotating_peg_high_speed')):
        raise ValueError('TRAIN-only diagnostic execution')
    if out.exists() or prepared['out'].resolve() != out/'case':
        raise ValueError('fresh standalone diagnostic root/case required')
    out.mkdir(parents=True, exist_ok=False)
    traces, observations, diagnostic_errors = [], [], []
    launch = dict(schema=SCHEMA, created_utc=utc(), development_only=True,
        participates_in_formal_qualification=False, participates_in_VAL_TEST=False,
        native_requested=True, original_collision_policy_unchanged=True,
        native_candidate_validated=False, automatic_retry=False,
        component_domain=domain.receipt(), diagnostic_inputs=diagnostic_lock,
        resource_inventory=permit.inventory, source_sha256=sha(root/SOURCES[0]),
        original_worker=dict(path=WORKER, sha256=sha(root/WORKER)),
        case_output=str(prepared['out'].relative_to(root)), native_target_pose_queries=False)
    _atomic(out/'diagnostic_entry_v787.json', launch)
    error = None
    try:
        with ExitStack() as stack:
            original_geometry = worker.runtime._geometry

            def geometry(*args, **kwargs):
                value = original_geometry(*args, **kwargs)
                actor = value[0]
                if actor is None or actor.map is not args[1].map_service or traces:
                    raise ValueError('exact single public ring actor/map required')
                observer = stack.enter_context(TrainPublicCollisionDiagnostics(actor.map,
                    port=actor.collision_port, domain=domain, enabled=True,
                    development_only=True, max_events=max_events))
                traces.append(observer)
                env = args[1]
                def snapshot(label, query_sequence=None):
                    try:
                        observations.append(dict(label=label, query_sequence=query_sequence,
                            created_utc=utc(), collection=collection_snapshot(env)))
                    except BaseException as problem:
                        diagnostic_errors.append(dict(label=label, type=type(problem).__name__, message=str(problem)))
                snapshot('after_actor_creation_before_first_policy_observation_or_action')
                model, port = actor.collision_port.model, actor.collision_port
                original_future = model.collision_query
                captured_lift = False
                def future(*a, **k):
                    nonlocal captured_lift
                    before = len(observer.events)
                    result = original_future(*a, **k)
                    # A TRUE map result in an actual lift query, not an
                    # unknown-scene fail-closed query that never called map.
                    if (not captured_lift and len(observer.events)>before
                            and (port.bound or {}).get('phase')=='lift'
                            and observer.events[-1].get('result') is True):
                        captured_lift = True
                        snapshot('first_actual_lift_public_map_rejection', observer.events[-1]['sequence'])
                    return result
                stack.enter_context(patch.object(model, 'collision_query', future))
                return value  # Preserve exact original geometry return tuple.

            stack.enter_context(patch.object(worker.base_factory, 'create_actor', worker.extension.create_actor))
            stack.enter_context(patch.object(worker.base_policy, 'construct', worker.latest_policy.construct))
            stack.enter_context(patch.object(worker.runtime, '_geometry', geometry))
            result = worker.runtime.run(prepared)  # Original744 cleanup/failure semantics.
        if verify_diagnostic_inputs(root, diagnostic_lock['path']) != diagnostic_lock:
            raise ValueError('diagnostic source supplement changed during execution')
        return result
    except BaseException as problem:
        error = problem
        raise
    finally:
        try:
            value = dict(launch, finished_utc=utc(), complete=error is None,
                original_exception=None if error is None else dict(type=type(error).__name__, message=str(error)),
                public_query_traces=[trace.receipt() for trace in traces],
                collection_observations=observations, diagnostic_errors=diagnostic_errors,
                no_trace_is_not_evidence_of_no_collision=not bool(traces))
            _atomic(out/'public_collision_diagnostics_v787.json', value)
        except BaseException as write_error:
            if error is None:
                raise
            if hasattr(error, 'add_note'):
                error.add_note('diagnostic sidecar write failed: '+repr(write_error))


def main(argv=None):
    args = parser().parse_args(argv)
    out = validate_request(ROOT, args)
    with reserved_resources(ROOT) as permit, original_cpu_scopes():
        prepared, worker = prepare(ROOT, args, out)
        sources = source_fingerprints(ROOT)
        if not args.execute_native:
            print(json.dumps(dict(schema=SCHEMA, complete=True, development_only=True,
                native_started=False, GPU_started=False, native_candidate_validated=False,
                participates_in_formal_qualification=False, participates_in_VAL_TEST=False,
                component_domain=prepared['component_domain'].receipt(), model_preflight=prepared['model_preflight'],
                resource_inventory=permit.inventory, source_sha256=sources,
                future_diagnostic_input_path=DIAGNOSTIC_LOCK,
                future_diagnostic_input_template=dict(schema=INPUT_SCHEMA, split='train',
                    approved_for_development_only=True, files=sources)), ensure_ascii=False))
            return
        diagnostic_lock = verify_diagnostic_inputs(ROOT, args.diagnostic_inputs)
        from scripts.native_display_guard_r10 import readiness
        display = readiness(os.environ)
        if not display['ready']:
            raise RuntimeError('original CPU/X11 native environment not ready: '+json.dumps(display))
        result = run_prepared(ROOT, prepared, worker, out, permit,
            diagnostic_lock=diagnostic_lock, max_events=args.max_events)
        print(json.dumps(dict(complete=True, development_only=True,
            native_success=result['official_task_success'], diagnostic=str(out/'public_collision_diagnostics_v787.json'),
            participates_in_formal_qualification=False, participates_in_VAL_TEST=False)))


if __name__ == '__main__':main()
