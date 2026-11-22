"""Write RGB demo videos using the project's existing FFmpeg binary."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def write_video(path, frames, fps=20):
    binaries = list((ROOT/'.conda-env/lib').glob('python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-*'))
    if not binaries:
        raise RuntimeError('project-local FFmpeg binary unavailable')
    height, width = frames[0].shape[:2]
    command = [str(binaries[0]), '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
               '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    if process.wait():
        raise RuntimeError('demo video encoding failed')
