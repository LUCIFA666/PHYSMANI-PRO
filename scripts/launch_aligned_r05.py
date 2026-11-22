"""Start the corrected full run after native and full-VAL acceptance gates."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/framework/aligned_r05'

def main():
    while True:
        try:
            native=json.loads((BASE/'smoke_protocol_v2/report.json').read_text())
            dynamics=json.loads((BASE/'dynamics_validation/report.json').read_text())
            ready=native['passed'] and dynamics['complete'] and dynamics['completed']==40
        except (OSError,ValueError,KeyError):ready=False
        if ready:break
        if (ROOT/'runtime/framework/aligned_r05_STOP').exists():raise InterruptedError('r05 stop requested')
        time.sleep(5)
    subprocess.run([sys.executable,'scripts/run_aligned_r05_v2.py'],cwd=ROOT,check=True)

if __name__=='__main__':main()
