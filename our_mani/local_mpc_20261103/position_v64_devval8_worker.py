"""Independent V64 public controller worker; default check never starts Native.

Original Native construction, random seeding, 50 ms executor and posthoc label
records are reused without replacement. Execution additionally requires a real
passing V8 position certificate and an externally frozen complete source map.
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
REFERENCE_PLAN = ROOT/'config/framework/P4_student_closedloop_devVAL_v56_v1/plan.json'
CANARY_PLAN = ROOT/'config/framework/P4_student_closedloop_devVAL_v64_v1/plan.json'
CASE_IDS = ('00_val_static', '01_val_static', '02_val_uniform', '03_val_uniform', '04_val_uniform', '05_val_uniform', '06_val_speed_change', '07_val_speed_change')
OWN_FILES = (
    'our_mani/local_mpc_20261103/position_v64_devval8_worker.py',
    'scripts/run_local_mpc_position_v64_devval8_20261103.py',
    'scripts/launch_local_mpc_position_v64_devval8_20261103.py',
    'scripts/run_local_mpc_position_v64_devval8_20261103.sh',
    'scripts/run_position_v64_full_devval8_20261103.py',
)
ALLOWED_CONFIG_CHANGES = frozenset((
    'schema', 'information_setting', 'single_change', 'paired_reference',
    'candidate_darkred_target_v59', 'candidate_position_measurement_v64',
    'position_refiner_checkpoint', 'position_refiner_checkpoint_sha256',
    'v64_execution_source_freeze', 'v64_development_reference',
    'v64_preparation_only', 'candidate_goal_sphere_filter_v56', 'finger_response_fastpath',
))


def _reference(config):
    """Only eight pre-existing V56 development assignments, never TEST results."""
    ref = config.get('v64_development_reference', {})
    if ref.get('case_id') not in CASE_IDS:
        raise ValueError('One of the eight fixed V56 development parents required')
    jobs = json.loads(REFERENCE_PLAN.read_text())['jobs']
    job = next(j for j in jobs if j['id'] == ref['case_id'])
    path = (ROOT/job['config']).resolve(strict=True)
    if (ref.get('config') != job['config'] or ref.get('sha256') != sha(path)
            or sha(path) != job['config_sha256']):
        raise ValueError('Exact existing V56 development config identity required')
    baseline = json.loads(path.read_text())
    if {k:v for k,v in config.items() if k not in ALLOWED_CONFIG_CHANGES} != {
            k:v for k,v in baseline.items() if k not in ALLOWED_CONFIG_CHANGES}:
        raise ValueError('V64 must preserve original scene/RNG/physics/control/model/evidence configuration')
    return job, path


def _source_closure(config_path, producer, position_tracker):
    sources = freeze_sources(config_path, 'p1')
    sources.update({str((ROOT/name).relative_to(ROOT)): sha(ROOT/name) for name in OWN_FILES})
    _, reference = _reference(json.loads(Path(config_path).read_text()))
    sources[str(reference.relative_to(ROOT))] = sha(reference)
    sources[str(REFERENCE_PLAN.relative_to(ROOT))] = sha(REFERENCE_PLAN)
    # One common code/model/eight-config map is frozen before the first case.
    plan = json.loads(CANARY_PLAN.read_text())
    if tuple(row['id'] for row in plan['jobs']) != CASE_IDS:
        raise ValueError('Fixed complete eight-case development plan required')
    for row in plan['jobs']:
        path = (ROOT/row['config']).resolve(strict=True)
        config = json.loads(path.read_text());_reference(config)
        sources.update(freeze_sources(path, 'p1'))
        sources[str(path.relative_to(ROOT))] = sha(path)
        prior = (ROOT/config['v64_development_reference']['config']).resolve(strict=True)
        sources[str(prior.relative_to(ROOT))] = sha(prior)
    sources[str(CANARY_PLAN.relative_to(ROOT))] = sha(CANARY_PLAN)
    judge = ROOT/'our_mani/local_mpc_control_20261103/native_task_evaluation.py'
    sources[str(judge.relative_to(ROOT))] = sha(judge)
    extra = producer.position_producer_source_map()
    extra.update(position_tracker.model_sources())
    for name, expected in extra.items():
        path = Path(name).resolve(strict=True)
        relative = str(path.relative_to(ROOT))
        if sha(path) != expected:
            raise ValueError('V64 source/model evidence changed: '+relative)
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
    required_true = ('candidate_position_measurement_v64', 'public_grasp_candidate_transformer_v5',
                     'candidate_goal_sphere_filter_v56', 'postconfirmed_position_maintenance_v56',
                     'require_public_goal_completion', 'finger_response_fastpath')
    if any(config.get(key) is not True for key in required_true) or config.get('candidate_darkred_target_v59') is not False:
        raise ValueError('Only explicit independent V64 public candidate route is permitted')
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
    if not isinstance(artifact, str) or not artifact: missing.append('passing_V8_position_checkpoint')
    if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None: missing.append('real_calibrated_checkpoint_SHA256')
    if (source_freeze is None or not isinstance(expected_source_freeze_sha256, str)
            or re.fullmatch('[0-9a-f]{64}', expected_source_freeze_sha256) is None):
        missing.append('final_source_freeze_and_external_SHA256')
    if config.get('v64_preparation_only') is not False:
        missing.append('final_case_config')
    report = dict(schema='V64_independent_worker_preparation_check_v1', complete=True,
                  case_id=job['id'], scene=scene, ready=False, missing_admission=missing,
                  Native_started=False, model_forward_calls=0, no_training=True,
                  task_judge_changed=False, old_worker_or_records_changed=False)
    if missing:
        can_collect = collect_sources and missing == ['final_source_freeze_and_external_SHA256']
        if not can_collect:
            if require_ready or collect_sources: raise ValueError('V64 execution is blocked: '+', '.join(missing))
            return report
    checkpoint = (ROOT/artifact).resolve(strict=True)
    checkpoint.relative_to(ROOT)
    if sha(checkpoint) != digest: raise ValueError('Real calibrated V8 checkpoint SHA mismatch')
    import torch
    torch.set_num_threads(2)
    from ..local_mpc_control_20261103 import public_candidate_position_v64 as producer
    from ..local_mpc_control_20261103.depth_position_goal_perception_v64 import CurrentDepthPositionTrackerV64
    from ..cube_position_measurement_20261103_v8.runtime import CurrentCubePositionMeasurementV8
    position_tracker = CurrentDepthPositionTrackerV64(checkpoint, digest)
    runtime = position_tracker.refiner
    if (type(runtime) is not CurrentCubePositionMeasurementV8
            or runtime.position_measurement_qualified is not True
            or any(getattr(runtime, key, None) is not False for key in ('qualified', 'model_qualified', 'raw_mask_model_qualified'))
            or tuple(runtime.proposal_thresholds) != (.995, .9)):
        raise ValueError('A genuinely passing V8 position-only certificate is mandatory')
    sources = _source_closure(path, producer, position_tracker)
    if collect_sources:
        # Read-only prospective content for the parent to write/freeze after
        # the real certificate and final case config exist; no Native starts.
        return dict(schema='V64_devVAL8_execution_source_freeze_v1',
                    position_measurement_qualified=True,
                    checkpoint=dict(path=artifact,sha256=digest),files_sha256=sources,
                    Native_started=False,model_forward_calls=0)
    # The expected freeze SHA is a launcher argument, outside the config and
    # its source map. This avoids a config<->source-freeze hash cycle.
    source_path = Path(source_freeze).resolve(strict=True); source_path.relative_to(ROOT)
    if sha(source_path) != expected_source_freeze_sha256:
        raise ValueError('Final execution source-freeze SHA mismatch')
    freeze = json.loads(source_path.read_text())
    if (freeze.get('schema') != 'V64_devVAL8_execution_source_freeze_v1'
            or freeze.get('position_measurement_qualified') is not True
            or freeze.get('checkpoint') != {'path': artifact, 'sha256': digest}
            or freeze.get('files_sha256') != sources):
        raise ValueError('Exact complete prospectively frozen V64 dependency/model/config closure required')
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
    if any(p.exists() for p in (output/'STOP', output.parent/(output.name+'.STOP'), FAMILY/'STOP')):
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
            information_setting='public_candidate_position_v64_development',
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
        from ..local_mpc_control_20261103.public_candidate_position_v64 import (
            ObservedPublicCandidatePositionV64Backend, run_public_candidate_position_v64)
        from ..local_mpc_control_20261103.public_grasp_v41 import ActiveVisualGraspV41Config
        from ..local_mpc_control_20261103.active_visual_attachment import ActiveVisualConfig
        recorder = FrameRecorder()
        backend = ObservedPublicCandidatePositionV64Backend(backend, output/'public_shadow',
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
        result = run_public_candidate_position_v64(backend, executor, output, config, recorder=recorder)
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
            source_sha256=sources, stage='p1_public_position_v64', formal_evaluation=False))
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
