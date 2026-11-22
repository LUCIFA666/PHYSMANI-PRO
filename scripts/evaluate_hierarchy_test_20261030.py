"""Post-VAL-gated TEST160 entry. No TEST file is opened before a new VAL gate."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from our_mani.hierarchy_20261030.evaluation import EvaluationBlocked
from our_mani.hierarchy_20261030.evaluation_test import (
    ARTIFACTS, prepare_test_plan, cpu_preflight_test, launch_test_case, worker_test_case, audit_test,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--launch-case", metavar="CASE_KEY")
    mode.add_argument("--worker-case", metavar="CASE_KEY", help=argparse.SUPPRESS)
    mode.add_argument("--audit", action="store_true")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--native-val-release", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / ARTIFACTS / "plan_v1.json")
    parser.add_argument("--original-manifest", type=Path)
    parser.add_argument("--verified-manifest", type=Path)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--case-timeout", type=int, default=900)
    args = parser.parse_args()
    try:
        if args.prepare:
            plan, path = prepare_test_plan(ROOT, args.freeze, args.native_val_release, args.plan,
                original_manifest_path=args.original_manifest, verified_manifest_path=args.verified_manifest)
            result = {"plan_path": str(path), "expected_cases": plan["expected_cases"],
                      "complete_Native_VAL_gate_rechecked": True, "native_started": False}
        elif args.check:
            result = cpu_preflight_test(ROOT, args.plan, args.freeze, args.native_val_release)
        elif args.launch_case:
            result = launch_test_case(ROOT, args.plan, args.launch_case, args.freeze, args.native_val_release,
                                      device=args.device, timeout=args.case_timeout)
        elif args.worker_case:
            result = worker_test_case(ROOT, args.plan, args.worker_case, args.freeze, args.native_val_release, device=args.device)
        else:
            result = audit_test(ROOT, args.plan, args.freeze, args.native_val_release)
    except (EvaluationBlocked, ValueError, OSError) as error:
        print(json.dumps({"blocked": True, "reason": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    if args.worker_case and not result.get("complete"):
        return 2 if result.get("administrative_stop") else 1
    return 0 if not args.launch_case or result.get("case_complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
