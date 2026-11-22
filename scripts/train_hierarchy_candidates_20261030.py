"""Train a new handoff candidate selector from completed blind public forecasts."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from our_mani.hierarchy_20261030.candidate_training import parser, run


if __name__ == "__main__":
    run(parser().parse_args())
