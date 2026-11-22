"""Prepare an explicitly reviewed new public-held physical and control graph.

Only explicitly supplied source/profile/CPU metadata is read. No model is
constructed, simulator launched, or VAL/TEST roster/sample read. The original
742 lookup remains a declared inherited preflight domain. The new physical
source overlay and graph verify the added factory independently. Default inert.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from our_mani.multitask import repaired_policy_graph_v1143 as graphs

SOURCE = 'scripts/prepare_public_held_repaired_graph_v1212_r10.py'
TEST_SOURCE = 'tests/test_prepare_public_held_repaired_graph_v1212.py'
SCHEMA = 'r10_public_held_physical_control_graph_preparation_v1212'
OLD_GRAPH = dict(root='project',
    path='artifacts/framework/multitask_r10/parallel_repaired_control_v1202_release/policy_graph.json',
    sha256='79949876953d9e732de31eb2ae37af2459661e0d36314e0ccd18d41113db400b')
PREDECESSOR_PHYSICAL_SHA = '12c96b9b7818cc2cc76e2d185233b47042a3ef5063c5ddb25ecb20f63f21e470'
INHERITED_CPU_EVIDENCE = dict(root='project',
    path='artifacts/framework/multitask_r10/parallel_repaired_control_v1202_setup/cpu_r2/control_CPU_evidence.json',
    sha256='aab51ddf04be02676eba357bffa1f2281a9d9fc3d7e5f31e24389ffa0ff69719')
WORKER = 'scripts/run_public_held_repaired_case_v1206_r10.py'
PREPARER = 'our_mani/multitask/public_held_repaired_case_v1206.py'
CASE_SCHEMA = 'r10_parallel_graph_bound_public_held_repaired_case_v1206'
COMPONENTS = dict(
    train_owner='scripts/run_public_held_repaired_owner_v1207_r10.py',
    evaluation_owner='scripts/run_public_held_repaired_owner_v1207_r10.py',
    train_auditor='our_mani/multitask/public_held_repaired_readiness_v1208.py',
    evaluation_auditor='our_mani/multitask/public_held_repaired_validation_audit_v1211.py',
    admission_guard='our_mani/multitask/public_held_repaired_admission_v1209.py',
    admission_driver='our_mani/multitask/public_held_repaired_admission_v1209.py',
    formal_source_reader='our_mani/multitask/public_held_repaired_admission_v1209.py',
    split_source_lock_writer='our_mani/multitask/public_held_repaired_validation_inputs_v1210.py',
    resource_guard='our_mani/multitask/public_held_repaired_resource_guard_v1213.py',
    graph_preparation_helper=SOURCE)
CONTRACTS = dict(readiness_schema='r10_fresh_public_held_TRAIN192_readiness_v1208',
    admission_schema='r10_fresh_public_held_validation_admission_v1209',
    split_source_lock_schema='r10_full16_post_admission_split_source_lock_v1')
PROBE_SOURCE = 'scripts/probe_public_held_parent_locks_v1213_r10.py'
PHYSICAL_OVERLAY_MODULE = 'our_mani.multitask.public_held_repaired_overlay_v1205'
PHYSICAL_FACTORY = 'our_mani/multitask/public_held_block_phase_v1203.py'
PHYSICAL_FACTORY_SHA = '8f2670d18ffb387f41becb56ba7a16e72111419c562f5af706f3be7d3445c77e'
DEVELOPMENT_MODULE = 'our_mani/multitask/train_public_held_block_development_v1204.py'
DEVELOPMENT_WORKER = 'scripts/run_TRAIN_public_held_block_development_v1204_r10.py'
DEVELOPMENT_TEST = 'tests/test_train_public_held_block_development_v1204.py'
DEVELOPMENT_SOURCE_LOCK = dict(root='project',
    path='artifacts/framework/multitask_r10/TRAIN_public_held_block_v1204_setup/source_overlay_lock.json',
    sha256='9dd8d33129842ef2274604ad8eb86e0a4cec130b949869ffd043c9e1bf9217d3')
NEW_SOURCES = tuple(dict.fromkeys((*COMPONENTS.values(), WORKER, PREPARER, PROBE_SOURCE,
    'our_mani/multitask/public_held_repaired_overlay_v1205.py', PHYSICAL_FACTORY,
    DEVELOPMENT_MODULE, DEVELOPMENT_WORKER,
    'our_mani/multitask/evaluation.py',
    'our_mani/multitask/frozen_input_edges_v463.py',
    'our_mani/multitask/post_TEST_resource_guard_v924.py',
    'scripts/evaluate_multitask_case_r10.py')))
NEW_TESTS = (
    'tests/test_public_held_repaired_resource_guard_v1213.py',
    'tests/test_public_held_repaired_case_v1206.py',
    'tests/test_public_held_repaired_owner_v1207.py',
    'tests/test_public_held_repaired_readiness_v1208.py',
    'tests/test_public_held_repaired_admission_v1209.py',
    'tests/test_public_held_repaired_validation_inputs_v1210.py',
    'tests/test_public_held_repaired_validation_audit_v1211.py',
    'tests/test_public_held_repaired_overlay_v1205.py',
    'tests/test_public_held_block_phase_v1203.py', DEVELOPMENT_TEST, TEST_SOURCE)
HARD_LIMITS = dict(max_native_slots=12, max_GPU_slots=5, max_CPU_preflight_slots=6, ready_limit=24)
INPUT_PREFIX = 'artifacts/framework/multitask_r10/public_held_repaired_control_v1212_setup/'
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
    return dict(schema='r10_parallel_repaired_CPU_affinity_v1212', allowed_logical_cpus=allowed,
        physical_core_count=len(values), core_groups=values, native_slot_cpus=native,
        control_and_CPU_preflight_cpus=pool, max_native_slots=HARD_LIMITS['max_native_slots'],
        taskset_root=str(tool.parent), taskset_binary=dict(root='host_tools', path=tool.name,
            sha256=hashlib.sha256(tool.read_bytes()).hexdigest()),
        taskset_runs_outside820_as_control_tool=True, bitwise_equivalence_claimed=False)


def validate_CPU_affinity(value):
    require(value.get('schema') == 'r10_parallel_repaired_CPU_affinity_v1212'
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
        raise ValueError('public_held_graph1212: ' + message)


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


def _canonical(reference):
    require(type(reference) is dict and set(reference) <= {'root', 'path', 'sha256'}
            and set(reference) >= {'path', 'sha256'},
            'plain hash-bound project reference required')
    require(reference.get('root', 'project') == 'project', 'project-only physical source overlay required')
    return dict(root='project', path=reference['path'], sha256=reference['sha256'])


def inherited_CPU_sources(root, old):
    evidence, evidence_ref = read(root, INHERITED_CPU_EVIDENCE)
    require(evidence.get('complete') is True and evidence.get('actual_tool_exit_observed') is True
            and type(evidence.get('returncode')) is int and evidence['returncode'] == 0
            and evidence.get('native_started') is False and evidence.get('GPU_started') is False,
            'complete actual frozen predecessor CPU evidence required')
    before, after = evidence.get('source_hashes_before'), evidence.get('source_hashes_after')
    inherited_before = evidence.get('inherited_source_hashes_before')
    inherited_after = evidence.get('inherited_source_hashes_after')
    require(type(before) is dict and bool(before) and before == after
            and type(inherited_before) is dict and inherited_before == inherited_after,
            'stable complete inherited CPU source maps required')
    combined = dict(inherited_before)
    for name, expected in before.items():
        require(name not in combined or combined[name] == expected, 'inherited CPU source conflict')
        combined[name] = expected
    roots = [Path(value) for value in old['roots'].values()]
    aliases = set()
    for name, expected in combined.items():
        file = Path(name) if Path(name).is_absolute() else Path(root) / name
        require(file.resolve(strict=True) == file and file.suffix in ('.py', '.sh')
                and any(file.is_relative_to(directory) for directory in roots)
                and str(file) not in aliases, 'canonical distinct inherited CPU source-only path required')
        aliases.add(str(file))
        require(hashlib.sha256(file.read_bytes()).hexdigest() == expected,
                'frozen predecessor CPU source drift: ' + name)
    return combined, evidence_ref


def apply_physical_overlay(root, old, reference):
    module = importlib.import_module(PHYSICAL_OVERLAY_MODULE)
    physical, overlay_ref = module.read_overlay_lock(root, reference)
    overlay_ref = _canonical(overlay_ref)
    require(physical.get('integration_complete') is True, 'completed independent physical CPU envelope required')
    result = deepcopy(old['physical'])
    block = physical.get('block_factory', {})
    require(block.get('source') == PHYSICAL_FACTORY and block.get('sha256') == PHYSICAL_FACTORY_SHA
            and block.get('function') == 'integrated_block_components', 'exact reviewed1203 block factory required')
    core = physical.get('core_factory', {})
    original_core = result['factories']['core_geometry']
    require(core.get('source') == original_core['source']['path']
            and core.get('sha256') == original_core['source']['sha256']
            and core.get('function') == original_core['callable'], 'frozen1164 core factory must remain unchanged')
    require(pointer(root, PHYSICAL_FACTORY)['sha256'] == PHYSICAL_FACTORY_SHA, 'reviewed1203 bytes changed')
    for name, expected in physical['files'].items():
        actual = pointer(root, name)
        require(actual['sha256'] == expected, 'physical envelope byte drift: ' + name)
        if Path(name).suffix in graphs.SOURCE_SUFFIXES and not name.startswith('tests/'):
            key = 'project:' + name
            require(key not in result['sources_map'] or result['sources_map'][key] == actual,
                    'existing physical source may not be replaced: ' + name)
            result['sources_map'][key] = actual
    result['source_overlay_lock'] = overlay_ref
    result['factories']['block_geometry'] = dict(source=pointer(root, PHYSICAL_FACTORY),
                                                callable='integrated_block_components')
    worker = importlib.import_module(PREPARER[:-3].replace('/', '.'))
    require(Path(worker.__file__).resolve() == Path(root) / PREPARER, 'actual registered new preparer required')
    require(_canonical(worker.PHYSICAL_SOURCE_OVERLAY) == overlay_ref, 'new worker must pin this physical envelope')
    expected = {key: (value['source']['path'], value['callable']) for key, value in result['factories'].items()}
    require(worker.FACTORIES == expected, 'new worker registry must match all actual selected physical factories')
    require(worker.CASE_SCHEMAS == dict(train=CASE_SCHEMA + '_TRAIN_case',
            validation=CASE_SCHEMA + '_validation_case'), 'registered worker case schemas changed')
    return result, overlay_ref


def require_expected_physical_change(old, new):
    before = deepcopy(old['physical']); after = deepcopy(new['physical'])
    before.pop('source_overlay_lock'); after.pop('source_overlay_lock')
    old_sources = before.pop('sources_map'); new_sources = after.pop('sources_map')
    require(all(new_sources.get(key) == value for key, value in old_sources.items()),
            'every inherited physical source must remain exact')
    before['factories'].pop('block_geometry'); after['factories'].pop('block_geometry')
    require(before == after, 'core/PhysMani/environment/RML/outcome/checkpoint physical declarations changed')
    require(new['physical']['factories']['block_geometry']['source']['sha256'] == PHYSICAL_FACTORY_SHA
            and new['physical_policy_sha256'] != old['physical_policy_sha256'],
            'new1203 physical graph must receive its own different fingerprint')


def verify_development_validator_sources(root, spec):
    value, reference = read(root, DEVELOPMENT_SOURCE_LOCK)
    require(value.get('schema') == 'r10_fixed_TRAIN_public_held_block_development_v1204_source_lock'
            and value.get('integration_complete') is True and type(value.get('files')) is dict,
            'exact frozen1204 development source envelope required')
    factory = value.get('factory', {})
    require(factory.get('source') == PHYSICAL_FACTORY and factory.get('sha256') == PHYSICAL_FACTORY_SHA,
            'development evidence must concern the same reviewed1203 physical factory')
    available = [_canonical(ref) for mapping in (spec['physical']['sources_map'], spec['control']['sources_map'])
                 for ref in mapping.values() if ref.get('root', 'project') == 'project']
    for name, expected in value['files'].items():
        if Path(name).suffix not in ('.py', '.sh') or name.startswith('tests/'):
            continue
        actual = pointer(root, name)
        require(actual['sha256'] == expected and actual in available,
                'development validator dependency is not frozen in this graph: ' + name)
    return reference


def control_spec(root, *, profile_ref, physical_overlay_ref):
    typed_input(profile_ref, 'parallel_resource_profile.json')
    affinity_ref = pointer(root, AFFINITY_NAME)
    affinity, affinity_ref = read(root, affinity_ref)
    validate_CPU_affinity(affinity)
    old = graphs.read_policy_graph(root, OLD_GRAPH)
    require(old['physical_policy_sha256'] == PREDECESSOR_PHYSICAL_SHA, 'exact completed1202 physical predecessor required')
    _, inherited_CPU_ref = inherited_CPU_sources(root, old)
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
    spec['physical'], selected_overlay_ref = apply_physical_overlay(root, old, physical_overlay_ref)
    spec['external_roots'] = {key: value for key, value in old['roots'].items() if key != 'project'}
    require(spec['external_roots'].get('host_tools') in (None, affinity['taskset_root']),
            'existing control tool root must keep its exact original path')
    spec['external_roots']['host_tools'] = affinity['taskset_root']
    control = spec['control']
    for name, item in inventory.items():
        control['sources_map']['parallel_' + Path(name).stem] = item['source']
    control['components'].update({name: inventory[source]['source'] for name, source in COMPONENTS.items()})
    development_ref = verify_development_validator_sources(root, spec)
    control['contracts'] = dict(CONTRACTS)
    control['static_inputs'].update(parallel_resource_profile=profile_ref, physical_predecessor_graph=OLD_GRAPH,
                                    independently_closed_physical_overlay=selected_overlay_ref,
                                    inherited_control_CPU_evidence=inherited_CPU_ref,
                                    development_validator_source_lock=development_ref)
    control['static_inputs'].update(parallel_CPU_affinity_profile=affinity_ref,
                                    parallel_taskset_binary=affinity['taskset_binary'])
    for route in spec['routes']:
        for split, suffix in (('train', '_TRAIN_case'), ('evaluation', '_validation_case')):
            route[split].update(worker=inventory[WORKER]['source'], preparer=inventory[PREPARER]['source'],
                                case_schema=CASE_SCHEMA + suffix)
    return spec, inventory, old


def prepare_parallel_graph(root, *, profile_ref, physical_overlay_ref, cpu_evidence_ref, namespace, reviewed=False):
    require(reviewed is True, 'explicit source/CPU review required')
    typed_input(cpu_evidence_ref, 'control_CPU_evidence.json')
    root = Path(root).resolve()
    spec, inventory, old = control_spec(root, profile_ref=profile_ref, physical_overlay_ref=physical_overlay_ref)
    evidence, evidence_ref = read(root, cpu_evidence_ref)
    require(evidence.get('actual_tool_exit_observed') is True and evidence.get('complete') is True
            and type(evidence.get('returncode')) is int and evidence['returncode'] == 0
            and type(evidence.get('tests_run')) is int and evidence['tests_run'] > 0
            and type(evidence.get('skipped')) is int and evidence['skipped'] == 0 and evidence.get('native_started') is False
            and evidence.get('GPU_started') is False, 'actual complete CPU-only test exit required')
    expected = {name: item['source']['sha256'] for name, item in inventory.items()}
    require(evidence.get('source_hashes_before') == expected and evidence.get('source_hashes_after') == expected,
            'every new control/test byte must be stable around CPU execution')
    inherited, inherited_ref = inherited_CPU_sources(root, old)
    require(evidence.get('inherited_source_hashes_before') == inherited
            and evidence.get('inherited_source_hashes_after') == inherited
            and _canonical(evidence.get('inherited_CPU_evidence')) == _canonical(inherited_ref),
            'every inherited control/physical/test source must remain stable around the new CPU run')
    probe_ref = evidence.get('namespace_probe')
    typed_input(probe_ref, 'parent_lock_namespace_probe.json')
    probe, probe_ref = read(root, probe_ref)
    require(probe.get('schema') == 'r10_pure_OS_parent_lock_broker_probe_v1213'
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
    require_expected_physical_change(old, compiled)
    old_routes = [{k: v for k, v in route.items() if k not in ('train', 'evaluation')} for route in old['routes']]
    new_routes = [{k: v for k, v in route.items() if k not in ('train', 'evaluation')} for route in compiled['routes']]
    require(new_routes == old_routes, 'configuration, lookup locks, models and checkpoints cannot change')
    graph_ref = graphs.write_policy_graph(root, spec, namespace=namespace)
    actual = graphs.read_policy_graph(root, graph_ref)
    require(actual == compiled, 'written graph differs from reviewed compilation')
    receipt = dict(schema=SCHEMA, graph=graph_ref, predecessor_graph=OLD_GRAPH,
        physical_policy_sha256=actual['physical_policy_sha256'], control_graph_sha256=actual['control_graph_sha256'],
        policy_sha256=actual['policy_sha256'], CPU_evidence=evidence_ref, namespace_probe=probe_ref,
        physical_declarations_byte_identical=False, new_public_held_block_factory=True,
        inherited_sources_unchanged=True, original742_lookup_retained=True,
        lookup_is_inherited_preflight_domain_not_new_factory_qualification=True,
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
    parser.add_argument('--physical-overlay')
    parser.add_argument('--namespace')
    parser.add_argument('--reviewed', action='store_true')
    args = parser.parse_args(argv)
    if not args.prepare:
        return dict(schema=SCHEMA, execution_started=False, default_inert=True)
    require(all((args.profile, args.physical_overlay, args.cpu_evidence, args.namespace)), 'explicit preparation inputs required')
    profile_name = typed_input(args.profile, 'parallel_resource_profile.json')
    cpu_name = typed_input(args.cpu_evidence, 'control_CPU_evidence.json')
    return prepare_parallel_graph(ROOT, profile_ref=pointer(ROOT, profile_name),
        physical_overlay_ref=args.physical_overlay,
        cpu_evidence_ref=pointer(ROOT, cpu_name), namespace=args.namespace, reviewed=args.reviewed)


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False))
