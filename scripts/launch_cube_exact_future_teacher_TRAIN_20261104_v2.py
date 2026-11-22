"""Root-owned serial lifecycle calls this child; no launch without --collect."""
from pathlib import Path
import os,sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
if '--collect' in sys.argv[1:]:
    os.execv('/bin/bash',['bash',str(R/'scripts/run_cube_exact_future_teacher_TRAIN_20261104_v2.sh'),*sys.argv[1:]])
from our_mani.cube_exact_future_teacher_20261104_v2.worker import main
if __name__=='__main__':raise SystemExit(main())
