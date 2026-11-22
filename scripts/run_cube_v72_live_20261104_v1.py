"""Current-runtime Cube V72 worker entry; default is source-only check."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from our_mani.cube_v72_live_20261104_v1.worker import main
if __name__=='__main__':raise SystemExit(main())
