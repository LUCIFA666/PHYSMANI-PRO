"""Source-only full16 physical overlay for the separately identified1203 block.

The complete frozen1165/core1164 closure is inherited. Only the block factory
changes. An observed independent CPU run must close before this module writes
an active source lock. Future worker/control/graph artifacts are not inputs to
that physical evidence, avoiding a physical/control hash cycle.
"""
from copy import deepcopy
import importlib
import json
from pathlib import Path

from . import train_native_release_full16_overlay_v1165 as parent
from . import public_held_block_phase_v1203 as block

SOURCE = 'our_mani/multitask/public_held_repaired_overlay_v1205.py'
TEST_SOURCE = 'tests/test_public_held_repaired_overlay_v1205.py'
SCHEMA = 'r10_full16_public_held_repaired_physical_overlay_v1205'
LOCK_SCHEMA = SCHEMA + '_source_lock'
CPU_EVIDENCE_SCHEMA = SCHEMA + '_observed_CPU_evidence'
NAME = 'public_held_repaired_physical_v1205'
SETUP_PREFIX = 'artifacts/framework/multitask_r10/' + NAME + '_setup/'
PARENT_OVERLAY_LOCK = 'artifacts/framework/multitask_r10/TRAIN_native_release_full16_v1165_setup/source_overlay_lock.json'
PARENT_OVERLAY_LOCK_SHA = '7d1b53b6a56bd48012e0850320b859d765b7c4b6d32f94f6aa607a22f8eef929'
BLOCK_MODULE = 'our_mani.multitask.public_held_block_phase_v1203'
BLOCK_SOURCE = block.SOURCE
BLOCK_TEST = 'tests/test_public_held_block_phase_v1203.py'
BLOCK_SHA = '8f2670d18ffb387f41becb56ba7a16e72111419c562f5af706f3be7d3445c77e'
BLOCK_TEST_SHA = '1a2c1c0dabc38769e30dd9273630cfc212d871dae690fe2c10df50f435065995'
CORE_INTEGRATION_MODULE = parent.CORE_INTEGRATION_MODULE
CORE_INTEGRATION_SOURCE = parent.CORE_INTEGRATION_SOURCE
CHECKPOINT = deepcopy(parent.CHECKPOINT)
TASKS, METHODS, BLOCKS = parent.TASKS, parent.METHODS, parent.BLOCKS
ROUTES, ROUTES_SHA = parent.ROUTES, parent.ROUTES_SHA
MANIFEST, MANIFEST_SHA = parent.MANIFEST, parent.MANIFEST_SHA
TRAIN_INPUTS, TRAIN_INPUTS_SHA = parent.TRAIN_INPUTS, parent.TRAIN_INPUTS_SHA
OVERLAY_SOURCES = tuple(dict.fromkeys((*parent.OVERLAY_SOURCES,
    PARENT_OVERLAY_LOCK, SOURCE, TEST_SOURCE, *block.SOURCES, BLOCK_TEST)))


def require(value, message):
    if not value:
        raise ValueError('physical1205: ' + message)


digest = parent.digest
raw = parent.raw
pointer = parent.pointer
_path = parent._path
write_json = parent.write_json


def _parent_lock(root):
    raw(root, PARENT_OVERLAY_LOCK, PARENT_OVERLAY_LOCK_SHA)
    value, reference = parent.read_overlay_lock(root, PARENT_OVERLAY_LOCK)
    require(reference == dict(path=PARENT_OVERLAY_LOCK, sha256=PARENT_OVERLAY_LOCK_SHA)
            and value['integration_complete'] is True,
            'complete frozen1165 physical substrate required')
    return value, reference


def collect_source_hashes(root):
    """Read the independent physical roster, without a future worker or graph."""
    inherited, _ = _parent_lock(root)
    require(set(inherited['files']) <= set(OVERLAY_SOURCES), 'complete1165 closure must be retained')
    files = {name: pointer(root, name)['sha256'] for name in OVERLAY_SOURCES}
    require(all(files[name] == expected for name, expected in inherited['files'].items()),
            'inherited frozen1165 physical source changed')
    require(files[BLOCK_SOURCE] == BLOCK_SHA and files[BLOCK_TEST] == BLOCK_TEST_SHA,
            'frozen1203 source/test changed; allocate a new version')
    return files


def _core_binding(files):
    return dict(module=CORE_INTEGRATION_MODULE, function='integrated_components',
        source=CORE_INTEGRATION_SOURCE, sha256=files[CORE_INTEGRATION_SOURCE],
        matched_policy_variant=deepcopy(parent.CORE_POLICY_VARIANT))


def _block_binding(files):
    return dict(module=BLOCK_MODULE, function='integrated_block_components',
        source=BLOCK_SOURCE, sha256=files[BLOCK_SOURCE],
        matched_policy_variant=dict(changed_factor='public_grasp_phase_and_inhand_pose_admission',
            original1153_preclose_and1114_RRT_RML_gripper_preserved=True,
            new_numeric_thresholds=False, context671_domain_unchanged=True,
            core1164_and_baseline744_unchanged=True))


def _cpu_evidence(root, reference, files):
    require(type(reference) is dict and set(reference) == {'path', 'sha256'},
            'explicit observed physical CPU evidence pointer required')
    _path(root, reference['path'], prefix=SETUP_PREFIX)
    require(Path(reference['path']).name == 'physical_CPU_evidence.json',
            'typed independent physical CPU evidence basename required')
    evidence = json.loads(raw(root, reference['path'], reference['sha256']))
    require(evidence.get('schema') == CPU_EVIDENCE_SCHEMA and evidence.get('complete') is True
            and evidence.get('actual_tool_exit_observed') is True
            and type(evidence.get('returncode')) is int and evidence['returncode'] == 0
            and type(evidence.get('tests_exit_code')) is int and evidence['tests_exit_code'] == 0
            and type(evidence.get('tests_run')) is int and evidence['tests_run'] > 0
            and type(evidence.get('skipped')) is int and evidence['skipped'] == 0
            and evidence.get('native_started') is False and evidence.get('GPU_started') is False
            and evidence.get('source_hashes') == files
            and evidence.get('source_hashes_before') == files
            and evidence.get('source_hashes_after') == files
            and evidence.get('parent_source_overlay_lock') == dict(
                path=PARENT_OVERLAY_LOCK, sha256=PARENT_OVERLAY_LOCK_SHA),
            'actual successful CPU-only evidence must bind the complete independent physical closure')
    return evidence


def _value(files, evidence):
    return dict(schema=LOCK_SCHEMA, required_sources=list(OVERLAY_SOURCES), files=files,
        core_factory=_core_binding(files), block_factory=_block_binding(files),
        parent_source_overlay_lock=dict(path=PARENT_OVERLAY_LOCK, sha256=PARENT_OVERLAY_LOCK_SHA),
        unpinned_sources=[], integration_complete=True, integration_cpu_evidence=evidence,
        original_TRAIN_input_lock=dict(path=TRAIN_INPUTS, sha256=TRAIN_INPUTS_SHA),
        original_TRAIN_manifest=dict(path=MANIFEST, sha256=MANIFEST_SHA),
        original_745_routes=dict(path=ROUTES, sha256=ROUTES_SHA), shared671_checkpoint=deepcopy(CHECKPOINT),
        changed_factor='public_grasp_phase_and_inhand_pose_admission',
        core1164_and_baseline744_unchanged=True, original48_routes_and_models_unchanged=True,
        borrowed_frozen_physical_graph_identity=False,
        physical_CPU_evidence_excludes_future_worker_control_and_graph=True,
        original_745_execution_acceptance_granted=False, formal_VAL_TEST_qualification=False,
        qualification_granted=False, automatic_retry=False, TEST_used_for_fitting=False)


def write_overlay_lock(root, name, *, integration_cpu_evidence):
    """First-write only after actual independent physical CPU evidence closes."""
    root = Path(root).resolve()
    destination = _path(root, name, prefix=SETUP_PREFIX, must_exist=False)
    files = collect_source_hashes(root)
    _cpu_evidence(root, integration_cpu_evidence, files)
    write_json(destination, _value(files, integration_cpu_evidence))
    return pointer(root, name)


def read_overlay_lock(root, name):
    expected = None
    if type(name) is dict:
        require(set(name) <= {'root', 'path', 'sha256'} and name.get('root', 'project') == 'project'
                and set(name) >= {'path', 'sha256'}, 'project overlay pointer required')
        expected, name = name['sha256'], name['path']
    _path(root, name, prefix=SETUP_PREFIX)
    value = json.loads(raw(root, name, expected))
    files = collect_source_hashes(root)
    evidence = value.get('integration_cpu_evidence')
    _cpu_evidence(root, evidence, files)
    require(value == _value(files, evidence), 'complete separately identified1205 physical source lock required')
    return value, pointer(root, name)


def phase_component_binding(root, config, task, method):
    if method != 'physmani' and task in BLOCKS:
        module = importlib.import_module(BLOCK_MODULE)
        return dict(block_integration=module.binding(root, config['baseline_repo']))
    return parent.phase_component_binding(root, config, task, method)
