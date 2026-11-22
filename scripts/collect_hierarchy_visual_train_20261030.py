"""Prepare/check a fixed TRAIN capture plan or explicitly run one native case.

CPU only (default paths stay in the new namespace)::
  .conda-env/bin/python -B scripts/collect_hierarchy_visual_train_20261030.py --check

Each --run-case requires a fresh process, the existing frozen-system wrapper,
and an independent DISPLAY (the existing run_local_sim.sh provides xvfb-run).
This entry point never starts a queue, restores old STOPs, or selects VAL/TEST.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from our_mani.hierarchy_20261030.native_capture import ARTIFACTS, capture_case, check_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="CPU-only plan/source/hash/flock check; no native imports")
    mode.add_argument("--run-case", metavar="CASE_KEY", help="explicitly run one fixed TRAIN hold capture in a fresh native process")
    parser.add_argument("--plan", type=Path, default=ROOT / ARTIFACTS / "plan_v1.json")
    args = parser.parse_args()
    result = check_plan(ROOT, args.plan) if args.check else capture_case(ROOT, args.plan, args.run_case)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    if args.run_case and not result["complete"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
