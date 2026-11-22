"""Corrected Cube current-runtime worker; default check starts nothing."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from our_mani.cube_native_contact_heldout_worker_20261104_v1.worker import main
if __name__=='__main__':raise SystemExit(main())

