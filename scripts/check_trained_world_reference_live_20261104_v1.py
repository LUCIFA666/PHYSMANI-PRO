"""Default source/artifact-presence check only; no implicit NN/Native/GPU."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.trained_world_reference_live_20261104_v1.runtime import check

if __name__=='__main__':
    print(json.dumps(check(),indent=2,allow_nan=False))
