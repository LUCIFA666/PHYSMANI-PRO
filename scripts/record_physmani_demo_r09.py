"""Record one fixed-first-case PhysMani demo after the formal R09 comparison.

Pass --video-out plus the frozen baseline evaluator's single-case arguments.
Both --out and --video-out must be inside aligned_r09/demo_only. The original
formal records are read-only; this independent replay never enters their SR.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from our_mani.benchmark_protocol import file_sha256, verify_lock

BASE = ROOT / 'artifacts/framework/aligned_r09'
DEMO_BASE = BASE / 'demo_only'
CONFIG = ROOT / 'config/framework/aligned_protocol_r09_baseline_v6.json'
CAMERAS = ('front', '8', '16', '36')
EVALUATOR = ROOT / 'scripts/evaluate_physmani_aligned_r09_v6.py'
FFMPEG = ROOT / '.conda-env/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'


class DemoRecorderInvariantError(Exception):
    """Recording infrastructure failure; never a policy or task outcome."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def argument(values, name, default=None):
    if name not in values:
        return default
    require(values.count(name) == 1, 'duplicate argument: ' + name)
    index = values.index(name) + 1
    require(index < len(values) and not values[index].startswith('--'), 'argument needs a value: ' + name)
    return values[index]


def prepare(video_out, forwarded):
    require(not any(flag in forwarded for flag in ('--one-decision', '--init-only')),
            'partial/init diagnostics cannot be delivered as full demos')
    task = argument(forwarded, '--task')
    raw = argument(forwarded, '--raw-episode-id')
    require(task and raw and argument(forwarded, '--split', 'train') == 'test',
            'demo requires explicit --split test --task and --raw-episode-id')
    require((ROOT / argument(forwarded, '--config', str(CONFIG))).resolve() == CONFIG.resolve(),
            'use the unchanged frozen R09 V6 baseline configuration')
    out_arg = argument(forwarded, '--out')
    require(out_arg is not None, 'the evaluator requires --out')
    out = (ROOT / out_arg).resolve()
    video = (ROOT / video_out).resolve()
    require(out.is_relative_to(DEMO_BASE.resolve()) and out != DEMO_BASE.resolve(),
            '--out must be a new case directory inside artifacts/framework/aligned_r09/demo_only')
    require(video.is_relative_to(out) and video.suffix.lower() == '.mp4', '--video-out must be an MP4 inside --out')
    require(not out.exists() or not any(out.iterdir()), 'choose a fresh empty demo output directory; evaluator cache cannot record new frames')
    report_path = BASE / 'final_report.json'
    report = read(report_path)
    require(report.get('complete') is True and report.get('protocol_acceptance_passed') is True and
            report.get('total_test_runs') == report.get('expected_test_runs') == 400 and
            not report.get('acceptance_errors') and report.get('fixed_candidate_evaluation_release'),
            'wait for the full formal400 continuation report before demo-only replays')
    rows = read(ROOT / 'artifacts/framework/pipeline_v1/official_test_manifest.json')['episodes']
    selected = [row for row in rows if row['task'] == task]
    require(selected and raw == selected[0]['raw_episode_id'], 'record only the fixed first official manifest case for this task')
    require(sum(row['task'] == task and row['raw_episode_id'] == raw for row in rows) == 1, 'ambiguous official source')
    config = read(CONFIG)
    lock_path = ROOT / report.get('parallel_evaluation_inputs',
                                  'artifacts/framework/aligned_r09/fixed_evaluation_extra_inputs.json')
    lock = read(lock_path)
    verify_lock(ROOT, lock)
    for source in (EVALUATOR, CONFIG, ROOT / 'our_mani/sensor_alignment/physmani_executor_v6.py'):
        require(lock['files'].get(str(source.relative_to(ROOT))) == file_sha256(source), 'demo evaluator/config is not frozen')
    formal_path = BASE / 'test_physmani_v6' / (task + '__' + raw + '.json')
    formal = read(formal_path)
    require(formal['task'] == task and formal['raw_episode_id'] == raw and formal['split'] == 'test', 'formal source differs')
    model_sha = file_sha256(Path(config['baseline_checkpoint']))
    require(model_sha == formal['checkpoint_sha256'] == report['baseline_checkpoint_sha256'], 'released weights differ from formal comparison')
    require(FFMPEG.is_file() and os.access(FFMPEG, os.X_OK), 'project-local FFmpeg7.0.2 binary unavailable')
    return dict(task=task, raw=raw, row=selected[0], out=out, video=video, config=config,
                lock_path=lock_path, formal_path=formal_path, model_sha=model_sha,
                report_path=report_path, fps=1. / config['dt'])


class FrameRecorder:
    def __init__(self, task, raw, physics_clock):
        self.task, self.raw, self.physics_clock = task, raw, physics_clock
        self.frames, self.timestamps, self.ticks = [], [], []

    def state(self, env):
        return (self.physics_clock(), float(env.task.t), env.ledger.ticks, float(env.time),
                tuple(env.arm.get_joint_positions()), tuple(env.arm.get_joint_target_positions()))

    def capture(self, env, observation, robot, *, initial=False):
        before = self.state(env)
        if self.ticks and env.ledger.ticks <= self.ticks[-1]:
            raise DemoRecorderInvariantError('observer repeated or reversed a counted tick')
        if initial and env.ledger.ticks != 0:
            raise DemoRecorderInvariantError('initial video frame is not the reset frame')
        images = []
        for name in CAMERAS:
            values = np.asarray(observation['observation/images'][name])
            if values.ndim != 3 or values.shape[-1] != 3 or values.dtype != np.uint8:
                raise DemoRecorderInvariantError('video requires existing uint8 RGB channel')
            images.append(values.copy())
        frame = np.concatenate([np.concatenate(images[:2], axis=1), np.concatenate(images[2:], axis=1)], axis=0)
        image = Image.fromarray(frame)
        label = ('PhysMani V6 | DEMO-ONLY | ' + ('initial state' if initial else 'counted tick') +
                 f'\nt={robot.timestamp:.2f}s | tick={env.ledger.ticks}\n{self.task} | {self.raw}')
        ImageDraw.Draw(image).text((8, 8), label, fill='white', stroke_width=1, stroke_fill='black')
        frame = np.asarray(image).copy()
        if self.state(env) != before:
            raise DemoRecorderInvariantError('recording changed native clock/robot state')
        self.frames.append(frame)
        self.timestamps.append(float(robot.timestamp))
        self.ticks.append(int(env.ledger.ticks))


def encode_video(path, frames, fps):
    require(frames and math.isfinite(fps) and fps > 0, 'no recorded frames or invalid FPS')
    height, width = frames[0].shape[:2]
    temporary = path.with_name(path.stem + '.tmp.mp4')
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(FFMPEG), '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps),
               '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-threads', '2', '-crf', '20',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    failure = None
    try:
        for frame in frames:
            require(frame.shape == (height, width, 3) and frame.dtype == np.uint8, 'inconsistent video frame')
            process.stdin.write(frame.tobytes())
    except Exception as exc:
        failure = exc
    finally:
        try:
            process.stdin.close()
        except BrokenPipeError as exc:
            failure = failure or exc
    stderr = process.stderr.read().decode(errors='replace')
    code = process.wait()
    process.stderr.close()
    require(not failure and code == 0, 'video encoding failed: ' + repr(failure) + ' ' + stderr[-1500:])
    decoded = subprocess.run([str(FFMPEG), '-hide_banner', '-v', 'error', '-nostats', '-i', str(temporary),
                              '-map', '0:v:0', '-progress', 'pipe:1', '-f', 'null', '-'],
                             capture_output=True, text=True, check=True)
    frame_counts = [int(line.split('=', 1)[1]) for line in decoded.stdout.splitlines() if line.startswith('frame=')]
    require(frame_counts and frame_counts[-1] == len(frames), 'encoded/decoded frame count differs')
    temporary.replace(path)
    return dict(frame_count=len(frames), decoded_frame_count=frame_counts[-1], size=[width, height], fps=fps,
                nominal_video_duration_seconds=len(frames) / fps, validation='all encoded frames decoded with local FFmpeg',
                ffmpeg_executable=str(FFMPEG), ffmpeg_sha256=file_sha256(FFMPEG))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--video-out', required=True)
    args, forwarded = parser.parse_known_args()
    details = prepare(args.video_out, forwarded)
    import evaluate_physmani_aligned_r09_v6 as evaluator
    from our_mani.aligned.simulation import physics_time
    recorder = FrameRecorder(details['task'], details['raw'], physics_time)
    original_adapter, original_argv = evaluator.PhysManiAdapter, sys.argv

    class RecordingPhysManiAdapter(original_adapter):
        def __init__(self, root, config, task, model):
            # The disabled author logger still mkdirs; route only its output.
            super().__init__(root, dict(config, baseline_output=str(details['out'] / 'disabled_wm_logger')), task, model)

        def reset(self, env, task, variation, observation, robot):
            recorder.capture(env, observation, robot, initial=True)
            return super().reset(env, task, variation, observation, robot)

        def execute(self, env, action, *, observer=None):
            def capture(observation, robot):
                recorder.capture(env, observation, robot)
                if observer is not None:
                    observer(observation, robot)
            return super().execute(env, action, observer=capture)

    evaluator.PhysManiAdapter = RecordingPhysManiAdapter
    sys.argv = [str(EVALUATOR), *forwarded]
    try:
        evaluator.main()
    finally:
        evaluator.PhysManiAdapter = original_adapter
        sys.argv = original_argv
    record_path = details['out'] / (details['task'] + '__' + details['raw'] + '.json')
    record = read(record_path)
    require(record['task'] == details['task'] and record['raw_episode_id'] == details['raw'] and
            record['checkpoint_sha256'] == details['model_sha'], 'demo returned a different source/model')
    ticks = record['alignment_receipt']['clock']['ticks']
    require(recorder.ticks == list(range(ticks + 1)), 'video must contain reset frame plus one frame per counted tick')
    encoded = encode_video(details['video'], recorder.frames, details['fps'])
    inputs_path = details['out'] / (details['task'] + '_inputs.json')
    sources = [Path(__file__), EVALUATOR, CONFIG, ROOT / 'our_mani/aligned/physmani.py',
               ROOT / 'our_mani/ablations/physmani_executor_v4.py',
               *[ROOT / 'our_mani/sensor_alignment' / name for name in
                 ('author_rgbd.py', 'physmani_executor_v5.py', 'physmani_executor_v6.py')]]
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), demo_only=True, formal_metrics_included=False,
        original_formal_statistics_replaced=False, method='physmani', task=details['task'], raw_episode_id=details['raw'],
        selection='fixed first official TEST manifest case, regardless of outcome',
        interpretation='independent demo replay; its native outcome may differ from the original formal rollout',
        official_TEST_source_used=True, native_demo_success=record['official_task_success'], policy_failure=record.get('policy_failure'),
        output_record=str(record_path.relative_to(ROOT)), output_record_sha256=file_sha256(record_path),
        output_inputs_sha256=file_sha256(inputs_path), video=str(details['video'].relative_to(ROOT)), video_sha256=file_sha256(details['video']),
        original_formal_record=str(details['formal_path'].relative_to(ROOT)), original_formal_record_sha256=file_sha256(details['formal_path']),
        formal_final_report_sha256=file_sha256(details['report_path']), frozen_lock=str(details['lock_path'].relative_to(ROOT)),
        frozen_lock_sha256=file_sha256(details['lock_path']), checkpoint_sha256=details['model_sha'],
        source_sha256={str(p.relative_to(ROOT)): file_sha256(p) for p in sources},
        observer=dict(views=CAMERAS, frame_timestamps=recorder.timestamps, counted_ticks=recorder.ticks,
            initial_frame_included=True, extra_observe_or_render_or_physics_calls=False,
            source_images_copied=True, frames_fed_to_policy=False, native_clock_q_target_before_after_checks_passed=True),
        runtime_routing_only_difference=dict(disabled_wm_logger_directory=str(details['out'] / 'disabled_wm_logger')),
        video_validation=encoded)
    write(details['out'] / 'demo_receipt.json', receipt)
    print(json.dumps(dict(demo_only=True, video=receipt['video'], receipt=str((details['out'] / 'demo_receipt.json').relative_to(ROOT))), ensure_ascii=False))


if __name__ == '__main__':
    main()
