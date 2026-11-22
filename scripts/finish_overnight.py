"""Stop only registered our-mani jobs at the user-specified deadline."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT/'config/framework/overnight.json').read_text())
    deadline = datetime.fromisoformat(config['deadline_utc'])
    if datetime.now(timezone.utc) < deadline:
        raise RuntimeError('refusing to finalize before the authorized deadline')
    stopped = []
    for directory in ['runtime/framework/state', 'runtime/framework/domain_state',
                      'runtime/framework/local_state', 'runtime/framework/faster_state']:
        path = ROOT/directory
        path.mkdir(parents=True, exist_ok=True)
        (path/'STOP').write_text('user deadline reached\n')
    # Give training loops an opportunity to save at the next optimizer boundary.
    time.sleep(20)
    for service in config['owned_services']:
        if not service.startswith('our-mani-'):
            raise RuntimeError('unexpected service outside our-mani')
        subprocess.run(['systemctl', '--user', 'stop', service], check=False, timeout=90)
        stopped.append(service)
    pid = config['installer_pid']
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\x00', b' ').decode()
        if str(ROOT/'.conda-env/bin/uv') in command and ' sync ' in command:
            os.kill(pid, signal.SIGTERM)
            stopped.append(f'owned uv installer {pid}')
    except (FileNotFoundError, ProcessLookupError):
        pass
    subprocess.run(['systemctl', '--user', 'stop', 'our-mani-overnight-report.timer'], check=False)
    record = {'finished_utc': datetime.now(timezone.utc).isoformat(), 'deadline_utc': config['deadline_utc'],
              'stopped_owned_jobs': stopped, 'ngff_original_training_resumed': False,
              'other_project_jobs_touched': False}
    (ROOT/'artifacts/framework/overnight').mkdir(parents=True, exist_ok=True)
    (ROOT/'artifacts/framework/overnight/finished.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    subprocess.run([sys.executable, str(ROOT/'scripts/overnight_report.py'), '--final'], check=True)


if __name__ == '__main__':
    main()
