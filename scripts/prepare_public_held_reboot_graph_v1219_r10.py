"""Freeze a control-only host-reboot recovery; never read VAL/TEST samples.

The complete physical graph, models, configurations and fixed TRAIN roster
remain identical to1212. Old cases retain their original graph identities.
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
from scripts import prepare_public_held_repaired_graph_v1212_r10 as previous

SOURCE = 'scripts/prepare_public_held_reboot_graph_v1219_r10.py'
TEST_SOURCE = 'tests/test_prepare_public_held_reboot_graph_v1219.py'
SCHEMA = 'r10_public_held_host_reboot_control_preparation_v1219'
OLD_GRAPH = dict(root='project',
    path='artifacts/framework/multitask_r10/public_held_repaired_v1212_release/policy_graph.json',
    sha256='854d9b75ac91faa3751224a8abcd62b39c52adab026624e68cd99f5a7b087ece')
PHYSICAL_SHA = '2a2ce641a073d4c816bf2be7a8459622ab2f8c3c971a542376e0aab5af4fadbf'
INHERITED_CPU_EVIDENCE = dict(root='project',
    path='artifacts/framework/multitask_r10/public_held_repaired_control_v1212_setup/cpu_root_b/control_CPU_evidence.json',
    sha256='e93e2fdbd2678953ed7b4a222b8bf755ec872904264acb5dcc229776e308e16a')
WORKER = 'scripts/run_public_held_reboot_case_v1220_r10.py'
PREPARER = 'our_mani/multitask/public_held_reboot_case_v1220.py'
CASE_SCHEMA = 'r10_parallel_graph_bound_public_held_reboot_case_v1220'
COMPONENTS = dict(
    train_owner='scripts/run_public_held_reboot_owner_v1214_r10.py',
    evaluation_owner='scripts/run_public_held_reboot_owner_v1214_r10.py',
    train_auditor='our_mani/multitask/public_held_reboot_readiness_v1215.py',
    evaluation_auditor='our_mani/multitask/public_held_reboot_validation_audit_v1218.py',
    admission_guard='our_mani/multitask/public_held_reboot_admission_v1216.py',
    admission_driver='our_mani/multitask/public_held_reboot_admission_v1216.py',
    formal_source_reader='our_mani/multitask/public_held_reboot_admission_v1216.py',
    split_source_lock_writer='our_mani/multitask/public_held_reboot_validation_inputs_v1217.py',
    resource_guard='our_mani/multitask/public_held_repaired_resource_guard_v1213.py',
    graph_preparation_helper=SOURCE)
CONTRACTS = dict(readiness_schema='r10_reboot_public_held_TRAIN192_readiness_v1215',
    admission_schema='r10_reboot_public_held_validation_admission_v1216',
    split_source_lock_schema='r10_full16_post_admission_split_source_lock_v1')
NEW_SOURCES = tuple(dict.fromkeys((*COMPONENTS.values(), WORKER, PREPARER)))
NEW_TESTS = (
    'tests/test_public_held_reboot_owner_v1214.py',
    'tests/test_public_held_reboot_readiness_v1215.py',
    'tests/test_public_held_reboot_admission_v1216.py',
    'tests/test_public_held_reboot_validation_inputs_v1217.py',
    'tests/test_public_held_reboot_validation_audit_v1218.py',
    'tests/test_public_held_reboot_case_v1220.py',
    'tests/test_run_public_held_reboot_case_v1220_r10.py', TEST_SOURCE)
INPUT_PREFIX = 'artifacts/framework/multitask_r10/public_held_reboot_control_v1219_setup/'
AFFINITY_NAME = previous.AFFINITY_NAME
PROBE_SOURCE = previous.PROBE_SOURCE
PROBE_CHECKS = previous.PROBE_CHECKS
HARD_LIMITS = previous.HARD_LIMITS
pointer = previous.pointer
read = previous.read


def require(value, message):
    if not value:
        raise ValueError('reboot_graph1219: ' + message)


def persist(file):
    file = Path(file)
    with file.open('rb') as stream:
        os.fsync(stream.fileno())
    descriptor = os.open(file.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def typed_input(reference, basename):
    name = reference if type(reference) is str else reference.get('path') if type(reference) is dict else None
    require(type(name) is str and name.startswith(INPUT_PREFIX)
        and Path(name).name == basename and Path(name).as_posix() == name
        and '..' not in Path(name).parts and '\\' not in name,
        'role-specific recovery control input required before reading bytes')
    if type(reference) is dict:
        require(reference.get('root', 'project') == 'project', 'project-only control metadata')
    return name


def source_inventory(root):
    result = {}
    for name in dict.fromkeys((*NEW_SOURCES, *NEW_TESTS)):
        ref = pointer(root, name)
        tree = ast.parse((Path(root) / name).read_bytes())
        result[name] = dict(source=ref, functions=[node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))])
    return result


def inherited_CPU_sources(root, old):
    evidence, ref = read(root, INHERITED_CPU_EVIDENCE)
    require(evidence.get('complete') is True and evidence.get('actual_tool_exit_observed') is True
        and type(evidence.get('returncode')) is int and evidence['returncode'] == 0
        and evidence.get('native_started') is False and evidence.get('GPU_started') is False,
        'actual completed predecessor CPU proof required')
    require(evidence.get('source_hashes_before') == evidence.get('source_hashes_after')
        and evidence.get('inherited_source_hashes_before') == evidence.get('inherited_source_hashes_after'),
        'predecessor source maps changed around CPU run')
    result = dict(evidence['inherited_source_hashes_before'])
    for name, value in evidence['source_hashes_before'].items():
        require(name not in result or result[name] == value, 'conflicting predecessor source')
        result[name] = value
    require(bool(result), 'nonempty inherited source closure required')
    allowed = [Path(p) for p in old['roots'].values()]
    for name, expected in result.items():
        file = Path(name) if Path(name).is_absolute() else Path(root) / name
        require(file.resolve(strict=True) == file and file.suffix in ('.py', '.sh')
            and any(file.is_relative_to(p) for p in allowed)
            and hashlib.sha256(file.read_bytes()).hexdigest() == expected,
            'frozen inherited source drift: ' + name)
    return result, ref


def require_physical_equivalence(old, new):
    require(old['physical_policy_sha256'] == PHYSICAL_SHA
        and new['physical_policy_sha256'] == PHYSICAL_SHA and old['physical'] == new['physical'],
        'the entire physical declaration and fingerprint must remain identical')
    for key in ('tasks', 'methods', 'train_source_manifest', 'roots'):
        require(old[key] == new[key], 'fixed source grid or roots changed: ' + key)
    require(len(old['routes']) == len(new['routes']) == 48, 'complete48 routes required')
    for before, after in zip(old['routes'], new['routes']):
        require(set(before) == set(after), 'route keys changed')
        for key in before:
            if key not in ('train', 'evaluation'):
                require(before[key] == after[key], 'route model/configuration changed: ' + key)
            else:
                allowed = {'worker', 'preparer', 'case_schema'}
                require(set(before[key]) == set(after[key])
                    and all(before[key][k] == after[key][k] for k in before[key] if k not in allowed),
                    'physical runtime/consumer route declarations changed')
    return True


def control_spec(root, *, profile_ref):
    root = Path(root).resolve()
    typed_input(profile_ref, 'parallel_resource_profile.json')
    old = graphs.read_policy_graph(root, OLD_GRAPH)
    require(old['physical_policy_sha256'] == PHYSICAL_SHA, 'exact1212 physical predecessor required')
    _, inherited_ref = inherited_CPU_sources(root, old)
    profile, profile_ref = read(root, profile_ref)
    require(profile.get('hard_limits') == HARD_LIMITS and profile.get('target_utilization') == .9
        and profile.get('initial_limits') == dict(native=2, GPU=1, CPU=6)
        and profile.get('automatic_retry') is False and profile.get('source_outcome_filtering') is False,
        'unchanged bounded resource and outcome policy required')
    affinity_ref = pointer(root, AFFINITY_NAME)
    affinity, _ = read(root, affinity_ref)
    previous.validate_CPU_affinity(affinity)
    inventory = source_inventory(root)
    spec = {key: deepcopy(old[key]) for key in
        ('tasks', 'methods', 'train_source_manifest', 'physical', 'control', 'routes')}
    spec['schema'] = graphs.SPEC_SCHEMA
    spec['external_roots'] = {key: value for key, value in old['roots'].items() if key != 'project'}
    control = spec['control']
    for name, item in inventory.items():
        control['sources_map']['reboot_' + Path(name).stem] = item['source']
    control['components'].update({key: inventory[name]['source'] for key, name in COMPONENTS.items()})
    control['contracts'] = dict(CONTRACTS)
    control['static_inputs'].update(parallel_resource_profile=profile_ref,
        recovery_physical_predecessor_graph=OLD_GRAPH, inherited_recovery_CPU_evidence=inherited_ref,
        parallel_CPU_affinity_profile=affinity_ref, parallel_taskset_binary=affinity['taskset_binary'])
    for route in spec['routes']:
        for split, suffix in (('train', '_TRAIN_case'), ('evaluation', '_validation_case')):
            route[split].update(worker=inventory[WORKER]['source'], preparer=inventory[PREPARER]['source'],
                case_schema=CASE_SCHEMA + suffix)
    return spec, inventory, old


def validate_CPU_evidence(evidence, expected, inherited, inherited_ref):
    require(evidence.get('actual_tool_exit_observed') is True and evidence.get('complete') is True
        and type(evidence.get('returncode')) is int and evidence['returncode'] == 0
        and type(evidence.get('tests_run')) is int and evidence['tests_run'] > 0
        and type(evidence.get('skipped')) is int and evidence['skipped'] == 0
        and evidence.get('native_started') is False and evidence.get('GPU_started') is False,
        'actual complete CPU-only exit required')
    require(evidence.get('source_hashes_before') == expected and evidence.get('source_hashes_after') == expected
        and evidence.get('inherited_source_hashes_before') == inherited
        and evidence.get('inherited_source_hashes_after') == inherited
        and previous._canonical(evidence.get('inherited_CPU_evidence')) == previous._canonical(inherited_ref),
        'all new and inherited source bytes must remain stable during tests')


def prepare_parallel_graph(root, *, profile_ref, cpu_evidence_ref, namespace, reviewed=False):
    require(reviewed is True, 'explicit Root source/CPU review required')
    typed_input(cpu_evidence_ref, 'control_CPU_evidence.json')
    root = Path(root).resolve()
    spec, inventory, old = control_spec(root, profile_ref=profile_ref)
    evidence, evidence_ref = read(root, cpu_evidence_ref)
    inherited, inherited_ref = inherited_CPU_sources(root, old)
    validate_CPU_evidence(evidence, {name: item['source']['sha256'] for name, item in inventory.items()},
        inherited, inherited_ref)
    probe_ref = evidence.get('namespace_probe')
    typed_input(probe_ref, 'parent_lock_namespace_probe.json')
    probe, probe_ref = read(root, probe_ref)
    require(probe.get('schema') == 'r10_pure_OS_parent_lock_broker_probe_v1213'
        and probe.get('actual_tool_exit_observed') is True and probe.get('complete') is True
        and type(probe.get('returncode')) is int and probe['returncode'] == 0
        and set(probe.get('checks', {})) == set(PROBE_CHECKS)
        and all(probe['checks'][key] is True for key in PROBE_CHECKS)
        and probe.get('Native_started') is False and probe.get('GPU_started') is False
        and probe.get('global_permissions_changed') is False and probe.get('test_only') is True
        and probe.get('scientific_admission_granted') is False
        and probe.get('probe_source') == pointer(root, PROBE_SOURCE)
        and probe.get('guard_source') == pointer(root, COMPONENTS['resource_guard'])
        and probe.get('frozen_environment_manifest') == old['physical']['environment_manifest'],
        'actual unchanged820/1213 pureOS probes required after reboot')
    compiled = graphs.compile_policy_graph(root, spec)
    require_physical_equivalence(old, compiled)
    graph_ref = graphs.write_policy_graph(root, spec, namespace=namespace)
    require(graphs.read_policy_graph(root, graph_ref) == compiled, 'written graph differs from reviewed graph')
    persist(root / graph_ref['path'])
    receipt = dict(schema=SCHEMA, graph=graph_ref, predecessor_graph=OLD_GRAPH,
        CPU_evidence=evidence_ref, namespace_probe=probe_ref, source_inventory=inventory,
        physical_declarations_byte_identical=True, original_case_graphs_relabelled=False,
        new_control_graph=True, physical_policy_sha256=PHYSICAL_SHA,
        Native_started=False, GPU_started=False, VAL_read=False, TEST_read=False,
        readiness_granted=False, execution_started=False)
    file = root / namespace / 'parallel_preparation_receipt.json'
    with file.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    persist(file)
    return dict(graph=graph_ref, receipt=pointer(root, file.relative_to(root).as_posix()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    for name in ('profile', 'cpu-evidence', 'namespace'):
        parser.add_argument('--' + name)
    parser.add_argument('--reviewed', action='store_true')
    args = parser.parse_args(argv)
    if not args.prepare:
        return dict(schema=SCHEMA, execution_started=False, default_inert=True)
    require(all((args.profile, args.cpu_evidence, args.namespace)), 'explicit preparation inputs required')
    return prepare_parallel_graph(ROOT, profile_ref=pointer(ROOT, typed_input(args.profile, 'parallel_resource_profile.json')),
        cpu_evidence_ref=pointer(ROOT, typed_input(args.cpu_evidence, 'control_CPU_evidence.json')),
        namespace=args.namespace, reviewed=args.reviewed)


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False))
