"""New current-runtime child entry; Root owner records actual wait/descendants."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if '--execute' in sys.argv[1:]:
    os.execv('/bin/bash',['bash',str(ROOT/'scripts/run_cube_v72_live_20261104_v1.sh'),*sys.argv[1:]])
from our_mani.cube_v72_live_20261104_v1.worker import main
if __name__=='__main__':raise SystemExit(main())
