"""Versioned V63 worker entry; check only unless --execute is explicit."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.local_mpc_20261103.position_v63_worker import main

if __name__ == '__main__': raise SystemExit(main())
