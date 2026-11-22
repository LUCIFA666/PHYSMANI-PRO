"""Independent V70 public controller worker; default check never starts Native.

Original Native construction, random seeding, 50 ms executor and posthoc label
records are reused without replacement. Execution additionally requires a real
passing V10 scope12 position certificate and an externally frozen complete source map.
"""
import argparse
import json
import os
from pathlib import Path
import re
import traceback

from ..local_mpc_20261102 import worker as original
from ..local_mpc_20261102.records import (
    ROOT, archive_sources, dependency_receipt, freeze_sources, sha, utc,
    verify_sources, write_new,
)
from ..local_mpc_20261102.recording import FrameRecorder

FAMILY = ROOT/'runtime/framework/local_mpc_20261102'
REFERENCE_PLANS = {'fixed6':ROOT/'config/framework/P4_position_response_v69_development_canaries_v1/plan.json'}
CANARY_PLAN = ROOT/'config/framework/P4_position_response_v70_development_canaries_v1/plan.json'
CASE_IDS = ('00_val_static', '01_val_static', '03_val_uniform', '04_val_uniform', 'uniform_002', 'static_no_proxy')
OWN_FILES = (
    'our_mani/local_mpc_20261103/position_response_v70_worker.py',
    'scripts/run_local_mpc_position_response_v70_20261103.py',
    'scripts/launch_local_mpc_position_response_v70_20261103.py',
    'scripts/run_local_mpc_position_response_v70_20261103.sh',
    'scripts/run_position_response_v70_fixed_development_20261103.py',
)
ALLOWED_CONFIG_CHANGES = frozenset(('schema','information_setting','single_change','paired_reference','candidate_response_seed_v70','candidate_far_pixel_centre_v70','v70_execution_source_freeze','v70_development_reference','v70_preparation_only'))


def _reference(config):
    """Six exact closed V69 assignments; never use task outcomes as selection."""
    ref=config.get('v70_development_reference',{});ident=ref.get('case_id')
    if ident not in CASE_IDS:raise ValueError('One of the fixed six V69 development parents required')
    group='fixed6'
    plan_path=REFERENCE_PLANS[group]
    if ref.get('plan')!=str(plan_path.relative_to(ROOT))or ref.get('plan_sha256')!=sha(plan_path):raise ValueError('Exact closed reference plan required')
    job=next(j for j in json.loads(plan_path.read_text())['jobs']if j['id']==ident)
    path=(ROOT/job['config']).resolve(strict=True)
    if ref.get('config')!=job['config']or ref.get('sha256')!=sha(path):raise ValueError('Exact reference config identity required')
    baseline=json.loads(path.read_text())
    if {k:v for k,v in config.items()if k not in ALLOWED_CONFIG_CHANGES}!={k:v for k,v in baseline.items()if k not in ALLOWED_CONFIG_CHANGES}:raise ValueError('V70 must preserve all original physics/seed/models/proof/limits/budget')
    return job,path


def _source_closure(config_path, producer, position_tracker):
    sources = freeze_sources(config_path, 'p1')
    sources.update({str((ROOT/name).relative_to(ROOT)): sha(ROOT/name) for name in OWN_FILES})
    _, reference = _reference(json.loads(Path(config_path).read_text()))
    sources[str(reference.relative_to(ROOT))] = sha(reference)
    for reference_plan in REFERENCE_PLANS.values():sources[str(reference_plan.relative_to(ROOT))]=sha(reference_plan)
    # One common code/model/six-config map is frozen before the first case.
    plan = json.loads(CANARY_PLAN.read_text())
    if tuple(row['id'] for row in plan['jobs']) != CASE_IDS:
        raise ValueError('Fixed complete six-case development plan required')
    for row in plan['jobs']:
        path = (ROOT/row['config']).resolve(strict=True)
        config = json.loads(path.read_text());_reference(config)
        sources.update(freeze_sources(path, 'p1'))
        sources[str(path.relative_to(ROOT))] = sha(path)
        prior = (ROOT/config['v70_development_reference']['config']).resolve(strict=True)
        sources[str(prior.relative_to(ROOT))] = sha(prior)
    sources[str(CANARY_PLAN.relative_to(ROOT))] = sha(CANARY_PLAN)
    judge = ROOT/'our_mani/local_mpc_control_20261103/native_task_evaluation.py'
    sources[str(judge.relative_to(ROOT))] = sha(judge)
    extra = producer.pixel_centre_response_producer_source_map()
    extra.update(position_tracker.model_sources())
    for name, expected in extra.items():
        path = Path(name).resolve(strict=True)
        relative = str(path.relative_to(ROOT))
        if sha(path) != expected:
            raise ValueError('V70 source/model evidence changed: '+relative)
        sources[relative] = expected
    return sources


def check_config(config_path, scene, *, require_ready=False, source_freeze=None,
                 expected_source_freeze_sha256=None, collect_sources=False):
    """Read-only preparation check, or strict CPU certificate/source admission."""
    path = Path(config_path).resolve(strict=True)
    path.relative_to(ROOT)
    config = json.loads(path.read_text())
    job, _ = _reference(config)
    if scene != job['scene']:
        raise ValueError('Fixed development scene cannot change')
    required_true = ('candidate_response_seed_v70','candidate_far_pixel_centre_v70','candidate_response_seed_v69','candidate_response_seed_v68','candidate_scope12_position_v67', 'hold_micro_orientation_local_nominal_v66', 'public_reverification_epoch_v65', 'candidate_position_measurement_v64', 'public_grasp_candidate_transformer_v5',
                     'candidate_goal_sphere_filter_v56', 'postconfirmed_position_maintenance_v56',
                     'require_public_goal_completion')
    if any(config.get(key) is not True for key in required_true) or config.get('candidate_darkred_target_v59') is not False:
        raise ValueError('Only explicit independent V70 public candidate route is permitted')
    for key in ('public_grasp_active_visual_v4', 'public_grasp_active_visual_v41',
                'public_grasp_active_visual_v42', 'public_grasp_cold_probe_v43',
                'public_grasp_prototype', 'public_pregrasp_control', 'public_motion_shadow'):
        if config.get(key, False) is not False:
            raise ValueError('No competing public/truth controller route permitted')
    if (config.get('physics_dt') != .05 or config.get('control_dt') != .1
            or config.get('decision_ticks') != 2 or config.get('horizon_steps') != 8
            or config.get('cem_sequences_total') != 256
            or config.get('cem_iteration_sequences') != [88, 88, 80]):
        raise ValueError('Original 50ms/100ms/2tick/256-H8 protocol required')
    artifact = config.get('position_refiner_checkpoint')
    digest = config.get('position_refiner_checkpoint_sha256')
    missing = []
    if not isinstance(artifact, str) or not artifact: missing.append('passing_V10_scope12_position_checkpoint')
    if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None: missing.append('real_calibrated_checkpoint_SHA256')
    if (source_freeze is None or not isinstance(expected_source_freeze_sha256, str)
            or re.fullmatch('[0-9a-f]{64}', expected_source_freeze_sha256) is None):
        missing.append('final_source_freeze_and_external_SHA256')
    if config.get('v70_preparation_only') is not False:
        missing.append('final_case_config')
    report = dict(schema='V70_independent_worker_preparation_check_v1', complete=True,
                  case_id=job['id'], scene=scene, ready=False, missing_admission=missing,
                  Native_started=False, model_forward_calls=0, no_training=True,
                  task_judge_changed=False, old_worker_or_records_changed=False)
    if missing:
        can_collect = collect_sources and missing == ['final_source_freeze_and_external_SHA256']
        if not can_collect:
            if require_ready or collect_sources: raise ValueError('V70 execution is blocked: '+', '.join(missing))
            return report
    checkpoint = (ROOT/artifact).resolve(strict=True)
    checkpoint.relative_to(ROOT)
    if sha(checkpoint) != digest: raise ValueError('Real calibrated V10 scope12 checkpoint SHA mismatch')
    import torch
    torch.set_num_threads(2)
    from ..local_mpc_control_20261103 import public_candidate_response_seed_v70 as producer
    from ..local_mpc_control_20261103.depth_pixel_centre_goal_perception_v70 import CurrentDepthPixelCentreTrackerV70
    from ..cube_position_measurement_20261103_v10.runtime import CurrentCubePositionMeasurementV10
    position_tracker = CurrentDepthPixelCentreTrackerV70(checkpoint, digest)
    runtime = position_tracker.refiner
    if (type(runtime) is not CurrentCubePositionMeasurementV10
            or runtime.position_measurement_qualified is not True
            or any(getattr(runtime, key, None) is not False for key in ('qualified', 'model_qualified', 'raw_mask_model_qualified'))
            or tuple(runtime.proposal_thresholds) != (.995, .9)
            or runtime.scope12_EF_qualified is not True or runtime.axis_general_qualified is not False
            or runtime.scope_radius_m != .12 or runtime.active_certified_routes != ('legacy_v8','scope12_EF')):
        raise ValueError('A genuinely passing V10 scope12 position-only certificate is mandatory')
    sources = _source_closure(path, producer, position_tracker)
    if collect_sources:
        # Read-only prospective content for the parent to write/freeze after
        # the real certificate and final case config exist; no Native starts.
        return dict(schema='V70_development_execution_source_freeze_v1',
                    position_measurement_qualified=True,
                    checkpoint=dict(path=artifact,sha256=digest),files_sha256=sources,
                    Native_started=False,model_forward_calls=0)
    # The expected freeze SHA is a launcher argument, outside the config and
    # its source map. This avoids a config<->source-freeze hash cycle.
    source_path = Path(source_freeze).resolve(strict=True); source_path.relative_to(ROOT)
    if sha(source_path) != expected_source_freeze_sha256:
        raise ValueError('Final execution source-freeze SHA mismatch')
    freeze = json.loads(source_path.read_text())
    if (freeze.get('schema') != 'V70_development_execution_source_freeze_v1'
            or freeze.get('position_measurement_qualified') is not True
            or freeze.get('checkpoint') != {'path': artifact, 'sha256': digest}
            or freeze.get('files_sha256') != sources):
        raise ValueError('Exact complete prospectively frozen V70 dependency/model/config closure required')
    verify_sources(sources)
    report.update(ready=True, missing_admission=[], checkpoint_sha256=digest,
                  position_measurement_qualified=True, model_qualified=False,
                  raw_mask_model_qualified=False, source_sha256=sources,
                  source_freeze=dict(path=str(source_path),sha256=expected_source_freeze_sha256))
    return report


def execute(config_path, scene, output, *, source_freeze=None, expected_source_freeze_sha256=None):
    admission = check_config(config_path, scene, require_ready=True,
        source_freeze=source_freeze,expected_source_freeze_sha256=expected_source_freeze_sha256)
    config = json.loads(Path(config_path).read_text())
    output = Path(output).resolve()
    if FAMILY.resolve() not in output.parents or output.exists():
        raise ValueError('Fresh local MPC development output required; no resumption')
    if any(p.exists() for p in (output/'STOP', output.parent/'STOP', output.parent/(output.name+'.STOP'), FAMILY/'STOP')):
        raise ValueError('Administrative STOP is preserved')
    # Admission, certificate and source checks all precede mkdir and Native.
    proof = original.environment_proof()
    sources = admission['source_sha256']; native_sources = original.native_source_snapshot(config)
    output.mkdir(parents=True, exist_ok=False)
    started = utc(); backend = recorder = executor = None; code = 1
    try:
        archive_sources(output, sources)
        with (output/'execution_source_freeze.json').open('xb') as stream:
            stream.write(Path(admission['source_freeze']['path']).read_bytes())
        if sha(output/'execution_source_freeze.json') != admission['source_freeze']['sha256']:
            raise ValueError('Execution freeze raced during archival')
        write_new(output/'run_inputs.json', dict(started_utc=started, mode='p1', scene=scene,
            config=config, source_sha256=sources, dependencies=dependency_receipt(),
            frozen_environment=proof, pid=os.getpid(), CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
            native_sources_sha256=native_sources, old_experiments_resumed=False, TEST_read=False,
            information_setting='public_candidate_response_seed_v70_development',
            position_admission=admission, Native_labels_used_by_policy=False,
            label_record_paths=['initial_snapshot.json','native_ticks.jsonl','public_shadow/evaluation_labels.jsonl'],
            posthoc_judge='unchanged native_task_evaluation.evaluate_case; never called by policy'))
        backend = original.make_backend(config, scene)
        write_new(output/'initial_snapshot.json', backend.initial_snapshot)
        if config.get('diagnostic_no_proxy_attachment', False):
            if config.get('counterfactual_not_benchmark') is not True:
                raise ValueError('NoProxy requires explicit non-benchmark negative declaration')
            from ..local_mpc_control_20261103.no_proxy_attachment import NoProxyAttachmentBackend
            backend = NoProxyAttachmentBackend(backend)
        if config.get('gripper_completion_latch', False):
            from ..local_mpc_control_20261103.completed_gripper import CompletedGripperBackend
            backend = CompletedGripperBackend(backend)
            write_new(output/'gripper_completion_protocol.json', backend.gripper_completion_receipt())
        from ..local_mpc_control_20261103.public_candidate_response_seed_v70 import (
            ObservedPublicCandidateResponseSeedV70Backend, run_public_candidate_response_seed_v70)
        from ..local_mpc_control_20261103.public_grasp_v41 import ActiveVisualGraspV41Config
        from ..local_mpc_control_20261103.active_visual_attachment import ActiveVisualConfig
        recorder = FrameRecorder()
        backend = ObservedPublicCandidateResponseSeedV70Backend(backend, output/'public_shadow',
            refiner_checkpoint=ROOT/config['position_refiner_checkpoint'],
            refiner_checkpoint_sha256=config['position_refiner_checkpoint_sha256'],
            goal_sphere_filter=True, postconfirmed_maintenance=True,
            active_config=ActiveVisualConfig(**config.get('active_visual_settings', {})),
            recorder=recorder, save_frames=config.get('public_shadow_save_frames', True),
            grasp_config=ActiveVisualGraspV41Config(**config.get('public_grasp_settings', {})),
            checkpoint=ROOT/config['candidate_transformer_checkpoint'],
            checkpoint_sha256=config['candidate_transformer_checkpoint_sha256'],
            require_public_goal_completion=config['require_public_goal_completion'])
        backend.sample()
        executor = original.make_executor(backend, config)
        result = run_public_candidate_response_seed_v70(backend, executor, output, config, recorder=recorder)
        # Privileged labels are flushed only after the public controller returns.
        original.append_ticks(backend, output)
        recorder.close(); recorder = None
        if original.native_source_snapshot(config) != native_sources:
            raise RuntimeError('Original Native source changed')
        verify_sources(sources)
        if sha(admission['source_freeze']['path']) != admission['source_freeze']['sha256']:
            raise ValueError('Execution freeze changed during the case')
        if config.get('gripper_completion_latch', False):
            result['gripper_completion_protocol'] = backend.gripper_completion_receipt()
            result['finger_response_requalified_for_runtime_protocol'] = False
        if config.get('diagnostic_no_proxy_attachment', False):
            result['counterfactual_no_proxy_attachment'] = backend.no_proxy_attachment_receipt()
            result['positive_benchmark_score_eligible'] = False
        result.update(started_utc=started, finished_utc=utc(), source_unchanged=True)
        write_new(output/'result.json', result); code = 0
    except BaseException as error:
        if backend is not None: original.append_ticks(backend, output)
        if executor is not None and executor.last_receipt is not None:
            write_new(output/'last_execution_receipt.json', executor.last_receipt)
        write_new(output/'failure.json', dict(started_utc=started, finished_utc=utc(),
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
            source_sha256=sources, stage='p1_public_response_seed_v70', formal_evaluation=False))
        traceback.print_exc()
    finally:
        try:
            if recorder is not None: recorder.close()
        finally:
            closed = False
            try:
                if backend is not None:
                    backend.close(); closed = True
            finally:
                write_new(output/'worker_exit.json', dict(finished_utc=utc(), exit_code=code,
                    result_recorded=(output/'result.json').exists(), native_closed=closed))
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--scene', choices=('static','uniform','speed_change'), required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--source-freeze', type=Path)
    parser.add_argument('--source-freeze-sha256')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--print-source-freeze', action='store_true')
    args = parser.parse_args(argv)
    if args.print_source_freeze:
        if args.execute: parser.error('prospective source inspection cannot execute')
        print(json.dumps(check_config(args.config,args.scene,collect_sources=True),allow_nan=False));return 0
    if not args.execute:
        print(json.dumps(check_config(args.config,args.scene,source_freeze=args.source_freeze,
            expected_source_freeze_sha256=args.source_freeze_sha256), allow_nan=False)); return 0
    if args.output is None: parser.error('--execute requires --output')
    return execute(args.config,args.scene,args.output,source_freeze=args.source_freeze,
        expected_source_freeze_sha256=args.source_freeze_sha256)


if __name__ == '__main__': raise SystemExit(main())
