"""Finish the fixed 5-method x 2-task R09 demo inventory after formal400.

Two Context+value videos come from their original formal rollouts. Six CPU
and two GPU replays use new demo-only directories and never replace formal
records. Recording is selected by manifest position, never by task outcome.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from our_mani.benchmark_protocol import file_sha256, verify_lock, validate_manifest
import publish_aligned_r09_acceptance as delivery

BASE = ROOT / 'artifacts/framework/aligned_r09'
DEMO_BASE = BASE / 'demo_only/supplement'
TASKS = ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed')
METHODS = ('cv_geometry', 'cv_value', 'context_geometry', 'context_value', 'physmani')
FIRST_RAW = dict(zip(TASKS, ('episode2', 'episode0')))
OURS_EVALUATOR = 'scripts/evaluate_aligned_r09.py'
BASELINE_EVALUATOR = 'scripts/evaluate_physmani_aligned_r09_v6.py'
BASELINE_HARNESS = 'scripts/record_physmani_demo_r09.py'
CONFIG = 'config/framework/aligned_protocol_r09.json'
BASELINE_CONFIG = 'config/framework/aligned_protocol_r09_baseline_v6.json'
PROGRESS_LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def formal_directory(method):
    return BASE / ('test_physmani_v6' if method == 'physmani' else 'test_' + method)


def fixed_specs(manifest):
    specs = []
    for method in METHODS:
        for task in TASKS:
            selected = [row for row in manifest if row['task'] == task]
            require(selected and selected[0]['raw_episode_id'] == FIRST_RAW[task], 'fixed first TEST case differs')
            raw = selected[0]['raw_episode_id']
            specs.append(dict(method=method, task=task, raw_episode_id=raw,
                name=task + '__' + raw, queue='gpu' if method == 'physmani' else 'cpu',
                reuse_formal_video=method == 'context_value'))
    require(len(specs) == 10, 'expected canonical ten demo entries')
    return specs


def prepare():
    """Read-only gate. No output creation or child launch before formal400."""
    report_path, release_path = BASE / 'final_report.json', BASE / 'fixed_candidate_evaluation_release.json'
    require(report_path.is_file(), 'formal400 final report is not available yet')
    report, release, gate = read(report_path), read(release_path), read(BASE / 'success_acceptance.json')
    require(report.get('complete') is True and report.get('protocol_acceptance_passed') is True
            and report.get('total_test_runs') == report.get('expected_test_runs') == 400
            and not report.get('acceptance_errors'), 'wait for complete accepted formal400')
    require(report.get('fixed_candidate_evaluation_release') == release
            and report.get('development_performance_screen_passed') is False
            and gate.get('passed') is False
            and release.get('original_performance_screen_sha256') == file_sha256(BASE / 'success_acceptance.json')
            and release.get('original_performance_failures') == gate.get('failures')
            and release.get('original_VAL_results') == gate.get('results')
            and release.get('evaluation_allowed') is True
            and release.get('deployment_or_benefit_validated') is False
            and release.get('model_or_threshold_retuned_after_VAL') is False
            and release.get('TEST_success_labels_used_for_release') is False,
            'fixed candidate/failed VAL disclosure is not preserved')
    lock_paths = [BASE / name for name in ('frozen_inputs.json', 'comparison_v6_inputs.json',
                                          'fixed_evaluation_extra_inputs.json')]
    locked = {}
    for path in lock_paths:
        lock = read(path);verify_lock(ROOT, lock)
        for name, digest in lock['files'].items():
            require(name not in locked or locked[name] == digest, 'conflicting frozen identity: ' + name)
            locked[name] = digest
    parallel_path, parallel, _ = delivery.parallel_evidence(report, read(lock_paths[-1]))
    if parallel_path is not None:
        lock_paths.append(parallel_path)
        for name, digest in parallel['files'].items():
            require(name not in locked or locked[name] == digest, 'conflicting parallel identity')
            locked[name] = digest
    for name in (OURS_EVALUATOR, BASELINE_EVALUATOR, CONFIG, BASELINE_CONFIG,
                 'scripts/run_local_sim.sh', 'scripts/run_aligned_gpu.sh'):
        require(locked.get(name) == file_sha256(ROOT / name), 'unfrozen evaluator/config/launcher: ' + name)
    manifest_path = ROOT / 'artifacts/framework/pipeline_v1/official_test_manifest.json'
    manifest = read(manifest_path)['episodes'];validate_manifest(manifest)
    expected = {(row['task'], row['raw_episode_id']) for row in manifest}
    counts, formal_hashes = delivery.collect_cases('test', METHODS, expected, read(ROOT / CONFIG),
                                                  read(ROOT / BASELINE_CONFIG), locked)
    for method in METHODS:
        for task in TASKS:
            actual, saved = counts[method][task], report['variants'][method][task]
            require(actual['episodes'] == saved['episodes'] == 40 and actual['successes'] == saved['successes'],
                    'formal coverage/counts differ from final summary')
    usage_path = BASE / 'success_selection_usage_test.json'
    delivery.check_usage(read(usage_path), 'test', counts)
    conditions = read(BASE / 'test_conditions.json')
    require(conditions.get('passed') is True and conditions.get('split') == 'test'
            and not conditions.get('errors') and len(conditions.get('checks', [])) == 800,
            'full five-way official TEST condition checks are incomplete')
    policy_path = ROOT / 'config/framework/demo_policy.json'
    policy = read(policy_path)
    require(policy.get('methods') == list(METHODS) and policy.get('videos_per_method_per_task') == 1
            and policy.get('current_two_tasks_expected') == 10, 'demo policy differs')
    import imageio_ffmpeg
    from record_physmani_demo_r09 import FFMPEG
    require(Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve() == FFMPEG.resolve(),
            'video decoder and baseline recorder must use the same project-local FFmpeg binary')
    source_paths = [Path(__file__), ROOT / BASELINE_HARNESS, FFMPEG, ROOT / 'scripts/publish_aligned_r09_acceptance.py',
                    ROOT / 'scripts/summarize_aligned_r09_v6.py', policy_path,
                    *[ROOT / name for name in (OURS_EVALUATOR, BASELINE_EVALUATOR, CONFIG, BASELINE_CONFIG,
                                               'scripts/run_local_sim.sh', 'scripts/run_aligned_gpu.sh')]]
    snapshot = {**formal_hashes,
        **{relative(path): file_sha256(path) for path in [report_path, release_path, usage_path,
           BASE / 'test_conditions.json', *lock_paths, *source_paths]}}
    return dict(report=report, specs=fixed_specs(manifest), locked=locked, snapshot=snapshot,
                report_sha256=file_sha256(report_path), lock_paths=[relative(path) for path in lock_paths])


def verify_snapshot(context, *, full=False):
    for name, digest in context['snapshot'].items():
        if full or not name.startswith('artifacts/framework/aligned_r09/test_'):
            require(file_sha256(ROOT / name) == digest, 'formal/demo input changed during recording: ' + name)
    if full:
        verify_lock(ROOT, {'files': context['locked']})


def formal_source(spec):
    path = formal_directory(spec['method']) / (spec['name'] + '.json')
    record = read(path)
    require(record.get('task') == spec['task'] and record.get('raw_episode_id') == spec['raw_episode_id']
            and record.get('split') == 'test' and type(record.get('official_task_success')) is bool,
            'wrong formal source for demo')
    receipt = record['case_process_receipt']
    inputs_path = ROOT / receipt['child_inputs_path']
    child_path = ROOT / receipt['child_record_path']
    require(file_sha256(inputs_path) == receipt['child_inputs_sha256']
            and file_sha256(child_path) == receipt['child_record_sha256'], 'formal child source changed')
    return path, record, read(inputs_path)


def result(record):
    return dict(official_task_success=record['official_task_success'],
        failure=record.get('policy_failure', record.get('failure')),
        simulation_seconds=record.get('simulation_seconds',
            record.get('alignment_receipt', {}).get('clock', {}).get('simulation_seconds')))


def validated_video(path, record, method):
    decoded = delivery.decode_video(path)
    expected = (record['alignment_receipt']['clock']['ticks'] + 1 if method == 'physmani'
                else len(record['steps']))
    require(decoded['decoded_frame_count'] == expected > 0, 'video/frame/rollout length differs')
    require(decoded['size'] == [512, 512] and abs(decoded['fps'] - 20.) < 1e-6,
            'demo must contain four 256x256 views at the common 20Hz')
    return decoded


def entry(spec, *, video, validation, formal_path, formal, replay_path=None, replay=None,
          recording_receipt=None):
    return dict(method=spec['method'], method_label=delivery.LABELS[spec['method']],
        task=spec['task'], raw_episode_id=spec['raw_episode_id'],
        selection='fixed first official TEST manifest case, regardless of outcome',
        source='formal_rollout_video' if replay is None else 'independent_demo_only_replay',
        formal_metrics_included=replay is None, replay_included_in_formal_statistics=False,
        formal_record=relative(formal_path), formal_record_sha256=file_sha256(formal_path),
        original_formal_child_record= formal['case_process_receipt']['child_record_path'],
        original_formal_child_record_sha256=formal['case_process_receipt']['child_record_sha256'],
        formal_result=result(formal), replay_record=relative(replay_path) if replay_path else None,
        replay_record_sha256=file_sha256(replay_path) if replay_path else None,
        replay_result=result(replay) if replay else None,
        recording_receipt=relative(recording_receipt) if recording_receipt else None,
        recording_receipt_sha256=file_sha256(recording_receipt) if recording_receipt else None,
        video=relative(video), video_sha256=file_sha256(video), video_validation=validation,
        result_note='formal and replay outcomes are separate; replay nondeterminism is not used to replace statistics')


def reused_formal_entry(spec):
    require(spec['reuse_formal_video'] and spec['method'] == 'context_value', 'only the two saved formal videos are reused')
    formal_path, formal, _ = formal_source(spec)
    video = formal_path.with_suffix('.mp4')
    child_video = (ROOT / formal['case_process_receipt']['child_record_path']).with_suffix('.mp4')
    require(child_video.is_file() and file_sha256(video) == file_sha256(child_video),
            'formal copied video differs from preserved child video')
    return entry(spec, video=video, validation=validated_video(video, formal, spec['method']),
                 formal_path=formal_path, formal=formal)


def command_for(spec, child):
    require(child.resolve().is_relative_to(DEMO_BASE.resolve()), 'replay output must stay in demo-only directory')
    baseline = spec['method'] == 'physmani'
    command = ['bash', 'scripts/run_aligned_gpu.sh' if baseline else 'scripts/run_local_sim.sh',
        BASELINE_HARNESS if baseline else OURS_EVALUATOR]
    if baseline:
        command += ['--video-out', relative(child / (spec['name'] + '.mp4'))]
    command += ['--config', BASELINE_CONFIG if baseline else CONFIG, '--split', 'test',
        '--task', spec['task'], '--raw-episode-id', spec['raw_episode_id'],
        '--episodes-per-task', '0', '--out', relative(child)]
    if not baseline:
        predictor, ranking = spec['method'].split('_', 1)
        command += ['--predictor', predictor, '--ranking', ranking, '--videos', '1']
    return command


def validate_replay(spec, child, formal, formal_inputs):
    path = child / (spec['name'] + '.json');record = read(path)
    inputs_path = child / (spec['task'] + '_inputs.json' if spec['method'] == 'physmani' else 'inputs.json')
    inputs = read(inputs_path)
    require(record.get('task') == spec['task'] and record.get('raw_episode_id') == spec['raw_episode_id']
            and record.get('split') == 'test' and record.get('raw_episode_filter') == spec['raw_episode_id']
            and type(record.get('official_task_success')) is bool
            and type(record.get('evaluation_process_pid')) is int and record['evaluation_process_pid'] > 0
            and not record.get('one_decision_diagnostic'), 'replay returned wrong/partial source')
    require(inputs == formal_inputs, 'demo replay must use the original frozen input identity')
    for field in ('checkpoint_sha256', 'controller_sha256', 'value_checkpoint_sha256',
                  'selection_policy_sha256', 'sensor_protocol_sha256'):
        if field in formal:
            require(record.get(field) == formal[field], 'replay method/model changed: ' + field)
    from summarize_aligned_r09_v6 import compare_conditions
    parity = compare_conditions(formal, record, read(BASE / 'acceptance_tolerances.json'))
    require(parity['accepted'], 'demo reset differs under unchanged formal tolerances: ' + str(parity['problems']))
    video = child / (spec['name'] + '.mp4')
    validation = validated_video(video, record, spec['method'])
    if spec['method'] == 'physmani':
        receipt = read(child / 'demo_receipt.json')
        require(receipt.get('demo_only') is True and receipt.get('formal_metrics_included') is False
                and receipt.get('original_formal_statistics_replaced') is False
                and receipt.get('task') == spec['task'] and receipt.get('raw_episode_id') == spec['raw_episode_id']
                and receipt.get('video_sha256') == file_sha256(video)
                and receipt.get('output_record_sha256') == file_sha256(path)
                and receipt.get('original_formal_record_sha256') == file_sha256(
                    formal_directory(spec['method']) / (spec['name'] + '.json'))
                and receipt.get('observer', {}).get('native_clock_q_target_before_after_checks_passed') is True
                and receipt.get('observer', {}).get('extra_observe_or_render_or_physics_calls') is False,
                'baseline observer receipt differs')
    return path, record, inputs_path, video, validation, parity


def run_replay(spec, context, cancelled):
    verify_snapshot(context)
    formal_path, formal, formal_inputs = formal_source(spec)
    directory = DEMO_BASE / spec['method'] / spec['name']
    receipt_path = directory / 'recording_receipt.json'
    if receipt_path.exists():
        saved = read(receipt_path)
        require(saved.get('formal_record_sha256') == file_sha256(formal_path)
                and saved.get('execution_inputs_sha256') == file_sha256(DEMO_BASE / 'demo_execution_inputs.json')
                and saved.get('exit_code') == 0, 'cached demo recording identity differs')
        child = (ROOT / saved['child_directory']).resolve()
        require(child.is_relative_to(directory.resolve()), 'cached demo escaped its own output directory')
        checked = validate_replay(spec, child, formal, formal_inputs)
        for name, digest in saved['output_sha256'].items():
            require(file_sha256(ROOT / name) == digest, 'cached demo output changed: ' + name)
    else:
        if cancelled.is_set():
            raise InterruptedError('another demo infrastructure failure cancelled this pending launch')
        directory.mkdir(parents=True, exist_ok=True)
        child = directory / ('attempt_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        require(not child.exists(), 'demo attempt directory collision')
        child.mkdir()
        command = command_for(spec, child)
        # The baseline observer harness requires its --out directory empty.
        log_path = directory / (child.name + '.recording.log')
        environment = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                           MKL_NUM_THREADS='2', PYTHONHASHSEED='0')
        started = now()
        with log_path.open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            code = process.wait()
        if code:
            write(child / 'recording_infrastructure_error.json', dict(exit_code=code, log=relative(log_path),
                demo_only=True, counted_as_policy_failure=False, original_formal_statistics_replaced=False))
            raise RuntimeError('demo child failed; preserved partial attempt: ' + relative(child))
        checked = validate_replay(spec, child, formal, formal_inputs)
        path, record, inputs_path, video, validation, parity = checked
        outputs = [path, inputs_path, video, log_path]
        if spec['method'] == 'physmani': outputs.append(child / 'demo_receipt.json')
        write(receipt_path, dict(version='r09_fixed_first_demo_replay_v1', created_utc=now(),
            demo_only=True, formal_statistics_modified=False, method=spec['method'], task=spec['task'],
            raw_episode_id=spec['raw_episode_id'], formal_record_sha256=file_sha256(formal_path),
            execution_inputs_sha256=file_sha256(DEMO_BASE / 'demo_execution_inputs.json'),
            child_directory=relative(child), launch_command=command, launcher_process_pid=process.pid,
            evaluation_process_pid=record['evaluation_process_pid'], started_utc=started, finished_utc=now(),
            exit_code=code, output_sha256={relative(path): file_sha256(path) for path in outputs},
            formal_result=result(formal), replay_result=result(record), initial_conditions=parity,
            video_validation=validation, selected_by_success=False,
            interpretation='independent replay; failure/success retained without sample replacement'))
    path, record, _, video, validation, _ = checked
    return entry(spec, video=video, validation=validation, formal_path=formal_path,
                 formal=formal, replay_path=path, replay=record, recording_receipt=receipt_path)


@contextmanager
def exclusive_run():
    DEMO_BASE.mkdir(parents=True, exist_ok=True)
    path = DEMO_BASE / 'RUNNING.json'
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError('demo supplement already has a RUNNING marker; inspect it before resuming')
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(dict(pid=os.getpid(), created_utc=now(), source_sha256=file_sha256(Path(__file__))), stream)
    try:
        yield
    finally:
        path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-workers', type=int, default=4)
    parser.add_argument('--gpu-workers', type=int, default=2)
    parser.add_argument('--index-only', action='store_true', help='verify/index already completed demo-only recordings')
    args = parser.parse_args()
    require(1 <= args.cpu_workers <= 4 and 1 <= args.gpu_workers <= 2, 'maximum parallelism is four CPU and two GPU cases')
    context = prepare()
    with exclusive_run():
        execution = dict(version='r09_demo_only_fixed_first_inputs_v1', formal400_completed_first=True,
            snapshot=context['snapshot'], frozen_files=context['locked'], formal_locks=context['lock_paths'],
            cpu_workers=args.cpu_workers, gpu_workers=args.gpu_workers, fixed_cases=FIRST_RAW,
            methods=list(METHODS), demos_per_method_per_task=1, expected=10,
            original_formal_statistics_replaced=False, selection_uses_success_labels=False)
        execution_path = DEMO_BASE / 'demo_execution_inputs.json'
        if execution_path.exists():
            require(read(execution_path) == execution, 'demo execution source/schedule identity changed; preserve the existing run')
        else:
            write(execution_path, execution)
        entries = [reused_formal_entry(spec) for spec in context['specs'] if spec['reuse_formal_video']]
        pending = [spec for spec in context['specs'] if not spec['reuse_formal_video']]
        if args.index_only:
            require(all((DEMO_BASE / spec['method'] / spec['name'] / 'recording_receipt.json').is_file()
                        for spec in pending), '--index-only requires all eight replay receipts')
        cancelled = threading.Event()
        with ThreadPoolExecutor(max_workers=args.cpu_workers) as cpu, ThreadPoolExecutor(max_workers=args.gpu_workers) as gpu:
            futures = [(gpu if spec['queue'] == 'gpu' else cpu).submit(run_replay, spec, context, cancelled)
                       for spec in pending]
            try:
                for future in as_completed(futures):
                    entries.append(future.result())
                    with PROGRESS_LOCK:
                        write(DEMO_BASE / 'progress.json', dict(created_utc=now(), demo_only=True,
                            completed=len(entries), expected=10, complete=False, entries=entries))
            except Exception:
                cancelled.set()
                for future in futures: future.cancel()
                raise
        verify_snapshot(context, full=True)
        keys = {(entry['method'], entry['task'], entry['raw_episode_id']) for entry in entries}
        require(len(entries) == len(keys) == 10, 'canonical demo inventory is incomplete or duplicated')
        entries.sort(key=lambda row: (METHODS.index(row['method']), TASKS.index(row['task'])))
        index = dict(version='r09_canonical_ten_demo_index_v1', created_utc=now(), complete=True,
            expected=10, completed=10, formal_rollout_videos_reused=2, independent_demo_only_replays=8,
            formal_final_report='artifacts/framework/aligned_r09/final_report.json',
            formal_final_report_sha256=context['report_sha256'], execution_inputs=relative(execution_path),
            execution_inputs_sha256=file_sha256(execution_path), entries=entries,
            original_formal_statistics_replaced=False, replay_results_in_formal_statistics=False,
            selection='one fixed first TEST case per method/task; outcomes never used for sample selection',
            disclosure=delivery.DISCLOSURE,
            video_timing='four-view synchronous 20Hz simulator frames; not wall-clock real-time control',
            frame_convention='ours records one pre-actuation sensor frame per tick; baseline includes reset plus post-tick frames')
        write(BASE / 'demo_index.json', index)
        text = ['R09 固定首例演示索引：5方法 × 2任务 = 10段',
            '固定普通 episode2 / 高速 episode0；不按成功率选样。2段来自正式rollout，8段为独立demo-only补录。',
            '正式结果与补录结果分开；补录不覆盖、不计入正式400例统计。', delivery.DISCLOSURE, '']
        for row in entries:
            replay = '无独立重录' if row['replay_result'] is None else str(row['replay_result']['official_task_success'])
            text.append(f"{row['method']} | {row['task']} | formal={row['formal_result']['official_task_success']} | replay={replay}")
            text.append(f"  {row['video']} | decoded={row['video_validation']['decoded_frame_count']} frames | SHA256={row['video_sha256']}")
        (BASE / 'demo_index.txt').write_text('\n'.join(text) + '\n')
        write(DEMO_BASE / 'progress.json', dict(created_utc=now(), demo_only=True, complete=True,
            completed=10, expected=10, index=relative(BASE / 'demo_index.json'), index_sha256=file_sha256(BASE / 'demo_index.json')))
        print(json.dumps(dict(complete=True, demos=10, index=relative(BASE / 'demo_index.json')), ensure_ascii=False))


if __name__ == '__main__':
    main()
