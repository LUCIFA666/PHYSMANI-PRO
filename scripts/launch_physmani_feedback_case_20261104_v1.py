"""Replace this process with the isolated Native worker shell."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.execv('/bin/bash', ['bash', str(ROOT / 'scripts/run_physmani_feedback_case_20261104_v1.sh'), *sys.argv[1:]])
