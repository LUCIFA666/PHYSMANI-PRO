from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from our_mani.cube_motion_capture_recovery_20261104_v2.worker import main
if __name__=='__main__':raise SystemExit(main())
