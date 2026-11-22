"""Native execution goes through current runtime/Xvfb and the two global locks."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if '--execute'in sys.argv[1:]:
    os.execv('/bin/bash',['bash',str(ROOT/'scripts/run_cube_corrected_native_20261104_v1.sh'),*sys.argv[1:]])
from our_mani.cube_corrected_native_20261104_v1.worker import main
if __name__=='__main__':raise SystemExit(main())

