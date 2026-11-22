"""Finish already-running collectors, then resume with the corrected evaluator."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/framework/aligned_r05'

def main():
    revision=json.loads((BASE/'runner_revision.json').read_text())
    pid=revision['original_runner_pid']
    while True:
        reports=list(BASE.glob('collection_cv_train_*/report.json'))
        complete=len(reports)==6 and all((d:=json.loads(p.read_text()))['completed']==d['expected'] for p in reports)
        processes=subprocess.check_output(['ps','-eo','args'],text=True).splitlines()
        running=any('python scripts/collect_aligned_r05.py --predictor cv --split train ' in line for line in processes)
        if complete and not running:break
        time.sleep(10)
    cmdline=Path(f'/proc/{pid}/cmdline')
    if cmdline.exists():
        if b'run_aligned_r05.py' not in cmdline.read_bytes():raise RuntimeError('original PID was reused')
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
    revision.update(handoff_utc=datetime.now(timezone.utc).isoformat(),original_runner_state='terminated after all six original collectors completed',next_runner_started=True)
    (BASE/'runner_revision.json').write_text(json.dumps(revision,ensure_ascii=False,indent=2)+'\n')
    subprocess.run([sys.executable,'scripts/run_aligned_r05_v2.py'],cwd=ROOT,check=True)

if __name__=='__main__':main()
