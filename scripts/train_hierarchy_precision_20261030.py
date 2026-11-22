#!/usr/bin/env python3
"""Entry point for the independent 3cm target / 5cm hard-gate experiment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from our_mani.hierarchy_precision_20261030.training import main

if __name__ == "__main__":
    main()
