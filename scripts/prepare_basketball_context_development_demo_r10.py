"""Metadata-only preparation for two fixed TRAIN Context/geometry demos.

This does not start a simulator, load Torch, approve execution, or freeze a
launcher that has not yet been supplied. Preserve this namespace once written.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from evaluate_multitask_case_r10 import required_local_sources, FFMPEG
from collect_multitask_recovery_r10 import OWNED_SOURCES
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.factory import continuation_identity
from our_mani.multitask.forecasting import ForecastConfig
from our_mani.multitask.interaction import CandidateConfig, MPCConfig, InteractionLibrary
from our_mani.multitask.protocol import validate_protocol

BASE = ROOT / 'artifacts/framework/multitask_r10'
TASKS = ('moving_basketball_in_hoop', 'moving_basketball_in_hoop_high_speed')
OUT = BASE / 'basketball_Context_geometry_TRAIN_development_demo_v1_setup'


def read(path):
    return json.loads(Path(path).read_text())


def pointer(path):
    path = Path(path).resolve()
    return {'path': str(path.relative_to(ROOT)), 'sha256': file_sha(path)}


def local_import_superset(seeds):
    """Bind every local AST import, including inactive branches, plus parents.

    This intentionally overbinds local Python code. It is a preparation
    inventory, not a claim that arbitrary dynamic/native dependencies are done.
    """
    found, pending = set(), [Path(path).resolve() for path in seeds]
    while pending:
        path = pending.pop()
        if path in found or not path.is_file():
            continue
        found.add(path)
        if path.suffix != '.py':
            continue
        package = path.parent.relative_to(ROOT).parts
        for parent in path.parents:
            if parent == ROOT:
                break
            init = parent / '__init__.py'
            if init.is_file() and init not in found:
                pending.append(init)
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split('.') for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    prefix = list(package[:len(package)-node.level+1])
                else:
                    prefix = []
                stem = prefix + (node.module.split('.') if node.module else [])
                names = [stem] + [stem + alias.name.split('.') for alias in node.names]
            for name in names:
                if not name or name[0] != 'our_mani':
                    continue
                base = ROOT.joinpath(*name)
                for target in (base.with_suffix('.py'), base / '__init__.py'):
                    if target.is_file() and target not in found:
                        pending.append(target)
    return sorted(found)


def main():
    if OUT.exists():
        raise FileExistsError('preserve previous metadata; choose a fresh version')
    base_path = BASE / 'PhysMani_early_VAL_approved_v12/early_config.json'
    train_path = ROOT / 'artifacts/framework/protocol/train_manifest.json'
    report_path = BASE / 'TRAIN_artifacts_full_goalv6_semanticv9_3100_v1/report.json'
    config = copy.deepcopy(read(base_path))
    fit_report, original_manifest = read(report_path), read(train_path)
    rows = [copy.deepcopy(next(row for row in original_manifest['episodes']
                             if row['task'] == task)) for task in TASKS]
    candidate = CandidateConfig(arrival_offsets_seconds=(.3,.5,.7,.9,1.1), dt=.05,
        maximum_linear_speed=.8, maximum_angular_speed=2., maximum_observation_age=.1,
        minimum_observation_confidence=.2, geometry_time_penalty=.1,
        geometry_joint_effort_penalty=.02, ranking_minimum_gain=.02,
        ranking_disagreement_multiplier=1.)
    mpc = MPCConfig(dt=.05, horizon_steps=8, maximum_linear_speed=.8,
        maximum_linear_acceleration=3., maximum_angular_speed=2.,
        maximum_angular_acceleration=8., position_weight=50., rotation_weight=10.,
        smoothness_weight=.05, optimizer_iterations=50)
    config.update(execution_scope='fixed_two_TRAIN_development_demos_not_formal_metrics',
        formal_comparison_ready=False, candidate_config=asdict(candidate),
        mpc_config=asdict(mpc), forecast_config=asdict(ForecastConfig()),
        context_checkpoint=pointer(BASE/'context_se3_full_v6/dynamics.pt'))
    config['development_demo_scope'] = dict(
        version='r10_fixed_TRAIN_basketball_Context_geometry_development_demo_v1',
        tasks=list(TASKS), method='context_geometry', split='train', replica=0,
        source_selection='first exact original TRAIN manifest row per task, independent of outcomes',
        training_or_critic_optimization_source=False, formal_VAL=False, formal_TEST=False,
        formal_metrics_included=False, performance_estimate=False, exploration_seed=None,
        numerical_configuration_origin='ROOT-declared engineering initial values; not tuned on VAL/TEST',
        public_completion_metric_calibrated=False,
        native_outcome_is_parent_only=True, native_success_is_actor_observation=False,
        preserved_fourteen_task_public_permissions=True,
        periodic_collection_cases_started=1, periodic_collection_expansion_authorized=False,
        source_config=pointer(base_path), fitted_library_report=pointer(report_path))
    library_summary = {}
    continuation = {}
    for task in TASKS:
        for key in ('interaction', 'rigid_layout'):
            config['learned_artifacts'][task][key] = copy.deepcopy(fit_report['artifacts'][task][key])
        config['learned_artifacts'][task]['colour'] = None
        config['learned_artifacts'][task]['cv_value'] = None
        config['learned_artifacts'][task]['context_value'] = None
        library = InteractionLibrary.load(Path(config['learned_artifacts'][task]['interaction']['path']))
        assert library.metadata['development_support_override'] is False
        assert library.metadata['fit_configuration']['development_only'] is False
        assert library.metadata['fit_configuration']['minimum_completed_parent_sources'] >= 8
        assert all(t.reference_kind in ('translation_only','phase_origin','world') for t in library.templates)
        library_summary[task] = dict(phase_order=list(library.phase_order),
            templates=[{key:getattr(t,key) for key in ('phase','reference_role','manipulated_role',
                'reference_kind','reference_mode','reference_time_basis','position_tolerance_m',
                'orientation_tolerance_rad','duration_seconds','duration_budget_seconds')}
                for t in library.templates],
            colour_required=False, public_ball_rotation_required=False,
            geometry=config['learned_artifacts'][task]['geometry'],
            interaction=config['learned_artifacts'][task]['interaction'],
            rigid_layout=config['learned_artifacts'][task]['rigid_layout'])
        continuation[task] = continuation_identity(ROOT,config,task,'context',library,candidate,mpc)
    config['continuation_identities'] = {
        task:{'context': continuation[task]['sha256']} for task in TASKS}
    validate_protocol(config)
    manifest = dict(split='train', episodes=rows,
        version='r10_fixed_two_basketball_development_demo_source_manifest_v1',
        original_manifest=pointer(train_path), original_rows_unchanged=True,
        source_selection_outcome_independent=True,
        formal_VAL=False, formal_TEST=False, fitting_source=False)
    OUT.mkdir(parents=True)
    config_path, manifest_path = OUT/'configuration.json', OUT/'manifest.json'
    atomic_json(config_path,config)
    atomic_json(manifest_path,manifest)
    atomic_json(OUT/'continuation_identities.json',continuation)
    seeds = set(required_local_sources('context_geometry',split='train'))
    seeds.update(ROOT/name for name in OWNED_SOURCES)
    for item in continuation.values():
        seeds.update(ROOT/name for name in item['observation_control_sources'])
    sources = local_import_superset(seeds)
    inputs = {str(path):file_sha(path) for path in sources}
    artifacts = set()
    def visit(value):
        if isinstance(value,dict):
            if {'path','sha256'} <= set(value) and isinstance(value['path'],str):
                path=(ROOT/value['path']).resolve()
                if path.is_file() and path.is_relative_to(ROOT):
                    assert file_sha(path)==value['sha256'],str(path)
                    artifacts.add(path)
            for inner in value.values():visit(inner)
        elif isinstance(value,list):
            for inner in value:visit(inner)
    visit(config)
    # Source-bound physical reset needs its permission's exact native inputs.
    physical_path=(ROOT/config['development_reset_profile']['path']).resolve()
    physical=read(physical_path)
    for name,expected in physical['native_inputs_sha256'].items():
        path=(Path(config['baseline_repo'])/name).resolve()
        assert file_sha(path)==expected
        inputs[str(path)]=expected
    native_tasks = Path(config['baseline_repo'])/'third_party/RLBench/rlbench/tasks'
    for task in TASKS:
        path=native_tasks/(task+'.py')
        inputs[str(path)]=file_sha(path)
    # Existing r09 gate and recording codec are explicit, not altered for demos.
    for path in (config_path,manifest_path,train_path,report_path,FFMPEG,
            ROOT/'config/framework/demo_policy.json',
            ROOT/'artifacts/framework/aligned_r09/final_report.json',*artifacts):
        inputs[str(path.resolve())]=file_sha(path)
    for row in rows:
        path=Path(row['path']);stat=path.stat()
        library_summary[row['task']]['source_HDF5_stat'] = dict(path=str(path),
            size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,inode=stat.st_ino,device=stat.st_dev)
    dependencies = dict(version='r10_basketball_development_demo_dependency_preparation_v1',
        files=inputs, approved_for_execution=False, final_lock=False,
        local_AST_import_superset=True, local_source_count=len(sources),
        frozen_hook_owned_sources=list(OWNED_SOURCES),
        final_ROOT_wrapper_source_pending=True,
        remaining_to_freeze=['ROOT development wrapper source',
            'ROOT-selected exact native/runtime/environment launcher closure',
            'actual raw source HDF5 lock policy'],
        runtime_closure_claimed_complete=False, native_started=False,GPU_started=False)
    atomic_json(OUT/'source_lock_dependencies_preparation.json',dependencies)
    summary=dict(version='r10_basketball_development_demo_readiness_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),configuration=pointer(config_path),
        manifest=pointer(manifest_path),source_dependencies=pointer(OUT/'source_lock_dependencies_preparation.json'),
        sources=library_summary, native_started=False,GPU_started=False,
        formal_VAL_or_TEST_started=False, periodic_collection_expanded=False,
        pipeline_colour_is_not_a_basketball_gate=True,
        public_position_tracking_and_completion_still_require_actual_closed_loop_demonstration=True)
    atomic_json(OUT/'preparation.json',summary)
    print(json.dumps(dict(preparation=pointer(OUT/'preparation.json'),
        configuration=pointer(config_path),manifest=pointer(manifest_path),
        source_count=len(inputs),native_started=False),ensure_ascii=False))


if __name__ == '__main__':
    main()
