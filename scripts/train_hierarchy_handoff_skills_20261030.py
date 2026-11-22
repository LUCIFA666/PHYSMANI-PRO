"""Run isolated episode-handoff goal v2 skill pretraining from frozen v1 encoder."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from our_mani.hierarchy_20261030.handoff_training_v2 import parser, run


if __name__ == "__main__":
    run(parser().parse_args())
