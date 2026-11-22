"""Check dependencies or explicitly start a fresh controlled hierarchy owner."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from our_mani.hierarchy_20261030.workflow import main


if __name__ == "__main__":
    raise SystemExit(main())
