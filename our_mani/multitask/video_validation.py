"""Decode fixed task demos using the local FFmpeg executable, without ffprobe."""
import re
import subprocess
from pathlib import Path

from .evaluation import file_sha


def validate_video(root, video):
    root = Path(root).resolve()
    if not isinstance(video, dict) or not {'path', 'sha256', 'decoded_frame_count'} <= set(video):
        raise ValueError('a fixed first-case demo receipt is required')
    path = (root / video['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file() or file_sha(path) != video['sha256']:
        raise ValueError('recorded demo video is missing or changed')
    ffmpeg = root / '.conda-env/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
    result = subprocess.run([str(ffmpeg), '-hide_banner', '-xerror', '-err_detect', 'explode',
        '-i', str(path), '-map', '0:v:0', '-an', '-sn', '-progress', 'pipe:1', '-nostats',
        '-f', 'null', '-'], capture_output=True, text=True, check=False)
    frames = re.findall(r'^frame=(\d+)$', result.stdout, re.M)
    size = re.search(r'Video:.*? (\d{2,5})x(\d{2,5})[ ,\[]', result.stderr)
    fps = re.search(r'Video:.*? ([0-9.]+) fps', result.stderr)
    if result.returncode != 0 or not frames or not size or not fps:
        raise ValueError('recorded demo failed full FFmpeg decoding')
    count, resolution, rate = int(frames[-1]), list(map(int, size.groups())), float(fps[1])
    if (type(video['decoded_frame_count']) is not int or count != video['decoded_frame_count'] or count < 1
            or resolution != [512, 512] or abs(rate - 20.) > 1e-8):
        raise ValueError('recorded demo frame count/resolution/cadence differs')
    return dict(passed=True, decoded_frame_count=count, fps=rate, resolution=resolution,
                decoder=str(ffmpeg.relative_to(root)), decoder_sha256=file_sha(ffmpeg),
                path=str(path.relative_to(root)), sha256=video['sha256'])
