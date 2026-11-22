"""Prepare a new control graph with byte-identical1168 physical declarations.

Only explicitly supplied source/profile/CPU metadata is read. No model is
constructed, simulator launched, or VAL/TEST roster/sample read. The original
742 frozen-input lookup is retained; new controls are checked by this graph.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from our_mani.multitask import repaired_policy_graph_v1143 as graphs

SOURCE = 'scripts/prepare_parallel_repaired_graph_v1202_r10.py'
TEST_SOURCE = 'tests/test_prepare_parallel_repaired_graph_v1202.py'
SCHEMA = 'r10_parallel_control_graph_preparation_v1202'
OLD_GRAPH = dict(root='project',
    path='artifacts/framework/multitask_r10/parallel_repaired_control_v1194_release/policy_graph.json',
    sha256='0e5b27016b2a79800f5dedee824351c54950e47d926c67d6175740be11f1897f')
PHYSICAL_SHA = '12c96b9b7818cc2cc76e2d185233b47042a3ef5063c5ddb25ecb20f63f21e470'
WORKER = 'scripts/run_parallel_repaired_case_v1196_r10.py'
PREPARER = 'our_mani/multitask/parallel_repaired_case_v1196.py'
CASE_SCHEMA = 'r10_parallel_graph_bound_repaired_policy_case_v1196'
COMPONENTS = dict(
    train_owner='scripts/run_parallel_repaired_owner_v1197_r10.py',
    evaluation_owner='scripts/run_parallel_repaired_owner_v1197_r10.py',
    train_auditor='our_mani/multitask/parallel_repaired_readiness_v1198.py',
    evaluation_auditor='our_mani/multitask/parallel_repaired_validation_audit_v1201.py',
    admission_guard='our_mani/multitask/parallel_repaired_admission_v1199.py',
    admission_driver='our_mani/multitask/parallel_repaired_admission_v1199.py',
    formal_source_reader='our_mani/multitask/parallel_repaired_admission_v1199.py',
    split_source_lock_writer='our_mani/multitask/parallel_repaired_validation_inputs_v1200.py',
    resource_guard='our_mani/multitask/parallel_repaired_resource_guard_v1195.py',
    graph_preparation_helper=SOURCE)
CONTRACTS = dict(readiness_schema='r10_mixed_control_TRAIN192_readiness_v1198',
    admission_schema='r10_mixed_control_validation_admission_v1199',
    split_source_lock_schema='r10_full16_post_admission_split_source_lock_v1')
PROBE_SOURCE = 'scripts/probe_parallel_parent_locks_v1195_r10.py'
NEW_SOURCES = tuple(dict.fromkeys((*COMPONENTS.values(), WORKER, PREPARER, PROBE_SOURCE)))
NEW_TESTS = (
    'tests/test_parallel_repaired_resource_guard_v1195.py',
    'tests/test_parallel_repaired_case_v1196.py',
    'tests/test_parallel_repaired_owner_v1197.py',
    'tests/test_parallel_repaired_readiness_v1198.py',
    'tests/test_parallel_repaired_admission_v1199.py',
    'tests/test_parallel_repaired_validation_inputs_v1200.py',
    'tests/test_parallel_repaired_validation_audit_v1201.py', TEST_SOURCE)
HARD_LIMITS = dict(max_native_slots=12, max_GPU_slots=5, max_CPU_preflight_slots=6, ready_limit=24)
INPUT_PREFIX = 'artifacts/framework/multitask_r10/parallel_repaired_control_v1202_setup/'
AFFINITY_NAME = INPUT_PREFIX + 'parallel_CPU_affinity_profile.json'
PROBE_CHECKS = ('actual820_EACCES_reproduced', 'actual820_broker_passed', 'actual820_namespace_verified',
    'fresh_admission_final_nonces', 'parent_unlock_rejected',
    'wrong_server_PID_rejected', 'nonce_mismatch_rejected',
    'actual820_per_FD_fdinfo_verified', 'actual820_independent_conflicts_verified',
    'actual820_churn_per_FD_and_conflicts_stable')


def capture_CPU_affinity():
    allowed = sorted(os.sched_getaffinity(0))
    groups = {}
    for cpu in allowed:
        base = Path('/sys/devices/system/cpu') / ('cpu' + str(cpu))
        package = int((base / 'topology/physical_package_id').read_text())
        core = int((base / 'topology/core_id').read_text())
        size = (base / 'cache/index3/size').read_text().strip()
        units = {'K': 1024, 'M': 1024 * 1024}
        cache = int(size[:-1]) * units[size[-1]] if size[-1] in units else int(size)
        group = groups.setdefault((package, core), dict(package=package, core=core, cpus=[], L3_bytes=cache))
        group['cpus'].append(cpu)
    values = [groups[key] for key in sorted(groups)]
    require(len(values) > HARD_LIMITS['max_native_slots'], 'separate physical cores for preflight are required')
    native = [row['cpus'] for row in values[:HARD_LIMITS['max_native_slots']]]
    pool = sorted(cpu for row in values[HARD_LIMITS['max_native_slots']:] for cpu in row['cpus'])
    tool = Path('/usr/bin/taskset')
    return dict(schema='r10_parallel_repaired_CPU_affinity_v1202', allowed_logical_cpus=allowed,
        physical_core_count=len(values), core_groups=values, native_slot_cpus=native,
        control_and_CPU_preflight_cpus=pool, max_native_slots=HARD_LIMITS['max_native_slots'],
        taskset_root=str(tool.parent), taskset_binary=dict(root='host_tools', path=tool.name,
            sha256=hashlib.sha256(tool.read_bytes()).hexdigest()),
        taskset_runs_outside820_as_control_tool=True, bitwise_equivalence_claimed=False)


def validate_CPU_affinity(value):
    require(value.get('schema') == 'r10_parallel_repaired_CPU_affinity_v1202'
            and value.get('max_native_slots') == HARD_LIMITS['max_native_slots']
            and value.get('bitwise_equivalence_claimed') is False,
            'explicit control-only CPU affinity profile required')
    allowed, groups = value.get('allowed_logical_cpus'), value.get('core_groups')
    native, pool = value.get('native_slot_cpus'), value.get('control_and_CPU_preflight_cpus')
    require(type(allowed) is list and allowed == sorted(set(allowed)) and bool(allowed)
            and all(type(cpu) is int and cpu >= 0 for cpu in allowed)
            and type(groups) is list and value.get('physical_core_count') == len(groups)
            and type(native) is list and len(native) == HARD_LIMITS['max_native_slots']
            and type(pool) is list and bool(pool), 'complete CPU/core/slot partition required')
    flattened = [cpu for row in groups for cpu in row['cpus']]
    require(sorted(flattened) == allowed and len(flattened) == len(set(flattened))
            and native == [row['cpus'] for row in groups[:HARD_LIMITS['max_native_slots']]]
            and pool == sorted(cpu for row in groups[HARD_LIMITS['max_native_slots']:] for cpu in row['cpus'])
            and all(type(row['package']) is int and type(row['core']) is int and bool(row['cpus']) for row in groups),
            'Native physical-core slots and preflight pool must be disjoint and exhaustive')
    tool = value.get('taskset_binary', {})
    require(tool.get('root') == 'host_tools' and tool.get('path') == 'taskset'
            and type(value.get('taskset_root')) is str and value.get('taskset_runs_outside820_as_control_tool') is True,
            'explicitly pinned outer control taskset binary required')
    return value


def require(value, message):
    if not value:
        raise ValueError('parallel_graph1202: ' + message)


def pointer(root, name):
    require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts
            and str(Path(name)) == name, 'canonical relative path required')
    file = Path(root) / name
    require(file.resolve(strict=True) == file and file.is_file(), 'canonical existing file required: ' + name)
    return dict(root='project', path=name, sha256=hashlib.sha256(file.read_bytes()).hexdigest())


def read(root, reference):
    value, normalized = graphs.read_project_json(root, reference)
    return value, normalized


def typed_input(reference, basename):
    name = reference if type(reference) is str else reference.get('path') if type(reference) is dict else None
    require(type(name) is str and name.startswith(INPUT_PREFIX)
            and Path(name).name == basename and str(Path(name)) == name
            and '..' not in Path(name).parts and '\\' not in name,
            'role-specific parallel control input required before reading bytes')
    if type(reference) is dict:
        require(reference.get('root', 'project') == 'project', 'project-only control input required')
    return name


def source_inventory(root):
    """Actual AST declarations and hashes only; never import control modules."""
    names = tuple(dict.fromkeys((*NEW_SOURCES, *NEW_TESTS)))
    result = {}
    for name in names:
        reference = pointer(root, name)
        tree = ast.parse((Path(root) / name).read_bytes())
        result[name] = dict(source=reference,
            functions=[node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))])
    return result


def control_spec(root, *, profile_ref):
    typed_input(profile_ref, 'parallel_resource_profile.json')
    affinity_ref = pointer(root, AFFINITY_NAME)
    affinity, affinity_ref = read(root, affinity_ref)
    validate_CPU_affinity(affinity)
    old = graphs.read_policy_graph(root, OLD_GRAPH)
    require(old['physical_policy_sha256'] == PHYSICAL_SHA, 'exact frozen1168 physical policy required')
    profile, profile_ref = read(root, profile_ref)
    require(profile.get('hard_limits') == HARD_LIMITS and profile.get('target_utilization') == .9
            and profile.get('initial_limits') == dict(native=2, GPU=1, CPU=6)
            and all(type(value) is int for value in profile.get('hard_limits', {}).values())
            and all(type(value) is int for value in profile.get('initial_limits', {}).values())
            and profile.get('automatic_retry') is False and profile.get('source_outcome_filtering') is False,
            'explicit bounded parallel profile and unchanged outcome accounting required')
    inventory = source_inventory(root)
    spec = {key: deepcopy(old[key]) for key in
            ('tasks', 'methods', 'train_source_manifest', 'physical', 'control', 'routes')}
    spec['schema'] = graphs.SPEC_SCHEMA
    spec['external_roots'] = {key: value for key, value in old['roots'].items() if key != 'project'}
    require(spec['external_roots'].get('host_tools') in (None, affinity['taskset_root']),
            'existing control tool root must keep its exact original path')
    spec['external_roots']['host_tools'] = affinity['taskset_root']
    control = spec['control']
    for name, item in inventory.items():
        control['sources_map']['parallel_' + Path(name).stem] = item['source']
    control['components'].update({name: inventory[source]['source'] for name, source in COMPONENTS.items()})
    control['contracts'] = dict(CONTRACTS)
    control['static_inputs'].update(parallel_resource_profile=profile_ref, physical_predecessor_graph=OLD_GRAPH)
    control['static_inputs'].update(parallel_CPU_affinity_profile=affinity_ref,
                                    parallel_taskset_binary=affinity['taskset_binary'])
    for route in spec['routes']:
        for split, suffix in (('train', '_TRAIN_case'), ('evaluation', '_validation_case')):
            route[split].update(worker=inventory[WORKER]['source'], preparer=inventory[PREPARER]['source'],
                                case_schema=CASE_SCHEMA + suffix)
    return spec, inventory, old


def prepare_parallel_graph(root, *, profile_ref, cpu_evidence_ref, namespace, reviewed=False):
    require(reviewed is True, 'explicit source/CPU review required')
    typed_input(cpu_evidence_ref, 'control_CPU_evidence.json')
    root = Path(root).resolve()
    spec, inventory, old = control_spec(root, profile_ref=profile_ref)
    evidence, evidence_ref = read(root, cpu_evidence_ref)
    require(evidence.get('actual_tool_exit_observed') is True and evidence.get('complete') is True
            and type(evidence.get('returncode')) is int and evidence['returncode'] == 0
            and type(evidence.get('tests_run')) is int and evidence['tests_run'] > 0
            and type(evidence.get('skipped')) is int and evidence['skipped'] == 0 and evidence.get('native_started') is False
            and evidence.get('GPU_started') is False, 'actual complete CPU-only test exit required')
    expected = {name: item['source']['sha256'] for name, item in inventory.items()}
    require(evidence.get('source_hashes_before') == expected and evidence.get('source_hashes_after') == expected,
            'every new control/test byte must be stable around CPU execution')
    probe_ref = evidence.get('namespace_probe')
    typed_input(probe_ref, 'parent_lock_namespace_probe.json')
    probe, probe_ref = read(root, probe_ref)
    require(probe.get('schema') == 'r10_pure_OS_parent_lock_broker_probe_v1195'
            and probe.get('actual_tool_exit_observed') is True and probe.get('complete') is True
            and type(probe.get('returncode')) is int and probe['returncode'] == 0
            and set(probe.get('checks', {})) == set(PROBE_CHECKS)
            and all(probe['checks'][key] is True for key in PROBE_CHECKS)
            and probe.get('Native_started') is False and probe.get('GPU_started') is False
            and probe.get('global_permissions_changed') is False
            and probe.get('scientific_admission_granted') is False and probe.get('test_only') is True
            and probe.get('probe_source') == pointer(root, PROBE_SOURCE)
            and probe.get('guard_source') == pointer(root, COMPONENTS['resource_guard'])
            and probe.get('frozen_environment_manifest') == old['physical']['environment_manifest'],
            'actual frozen820 broker and rejection probes must pass on these exact control bytes')
    compiled = graphs.compile_policy_graph(root, spec)
    require(compiled['physical'] == old['physical']
            and compiled['physical_policy_sha256'] == old['physical_policy_sha256'] == PHYSICAL_SHA,
            'all physical declarations and their fingerprint must remain identical')
    old_routes = [{k: v for k, v in route.items() if k not in ('train', 'evaluation')} for route in old['routes']]
    new_routes = [{k: v for k, v in route.items() if k not in ('train', 'evaluation')} for route in compiled['routes']]
    require(new_routes == old_routes, 'configuration, lookup locks, models and checkpoints cannot change')
    graph_ref = graphs.write_policy_graph(root, spec, namespace=namespace)
    actual = graphs.read_policy_graph(root, graph_ref)
    require(actual == compiled, 'written graph differs from reviewed compilation')
    receipt = dict(schema=SCHEMA, graph=graph_ref, predecessor_graph=OLD_GRAPH,
        physical_policy_sha256=PHYSICAL_SHA, control_graph_sha256=actual['control_graph_sha256'],
        policy_sha256=actual['policy_sha256'], CPU_evidence=evidence_ref, namespace_probe=probe_ref,
        physical_declarations_byte_identical=True, original742_lookup_retained=True,
        new_control_sources_verified_by_graph=True, inherited742_preflight_entry='scripts/run_repaired_policy_case_v1167_r10.py',
        actual_worker=WORKER, route_count=48, source_inventory=inventory,
        Native_started=False, GPU_started=False, VAL_read=False, TEST_read=False,
        readiness_granted=False, execution_started=False)
    file = Path(root) / namespace / 'parallel_preparation_receipt.json'
    with file.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return dict(graph=graph_ref, receipt=pointer(root, str(file.relative_to(root))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--profile')
    parser.add_argument('--cpu-evidence')
    parser.add_argument('--namespace')
    parser.add_argument('--reviewed', action='store_true')
    args = parser.parse_args(argv)
    if not args.prepare:
        return dict(schema=SCHEMA, execution_started=False, default_inert=True)
    require(all((args.profile, args.cpu_evidence, args.namespace)), 'explicit preparation inputs required')
    profile_name = typed_input(args.profile, 'parallel_resource_profile.json')
    cpu_name = typed_input(args.cpu_evidence, 'control_CPU_evidence.json')
    return prepare_parallel_graph(ROOT, profile_ref=pointer(ROOT, profile_name),
        cpu_evidence_ref=pointer(ROOT, cpu_name), namespace=args.namespace, reviewed=args.reviewed)


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False))
