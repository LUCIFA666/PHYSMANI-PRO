"""Prepare exact four-task causal RGBD/proprio TRAIN400 and validation77 caches."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from our_mani.hierarchy_20261030.data import prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/framework/multitask_r10/hierarchy_20261030/data_v1")
    parser.add_argument("--smoke", action="store_true", help="Prepare only the first exact TRAIN source in a separate namespace")
    args = parser.parse_args()
    prepare_dataset(ROOT, ROOT / args.out, smoke=args.smoke)


if __name__ == "__main__":
    main()
