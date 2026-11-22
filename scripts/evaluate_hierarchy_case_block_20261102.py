"""Prepare provenance/check dependencies or explicitly launch one new-model case.

No action is selected by default. TEST has no reader or execution mode here.
Native execution is refused until every trained hierarchy/tracking dependency
is present in an explicit completed runtime freeze.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from our_mani.hierarchy_block_20261102.evaluation import (
    ARTIFACTS, EvaluationBlocked, audit_training, audit_validation, cpu_preflight,
    launch_case, prepare_plan, worker_case,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--launch-case", metavar="CASE_KEY")
    mode.add_argument("--worker-case", metavar="CASE_KEY", help=argparse.SUPPRESS)
    mode.add_argument("--audit-native-val", action="store_true")
    mode.add_argument("--audit-native-train", action="store_true")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--train-release", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--case-timeout", type=int, default=900)
    parser.add_argument("--resource-slot", type=int, choices=range(4), default=0)
    parser.add_argument("--parallel-native", action="store_true")
    args = parser.parse_args()
    if args.case_timeout < 1:
        parser.error("positive case timeout required")
    plan = args.plan or ROOT / ARTIFACTS / args.split / "plan_v1.json"
    try:
        if args.prepare:
            result, path = prepare_plan(ROOT, args.split, freeze_path=args.freeze, plan_path=plan,
                                        train_release_path=args.train_release)
            result = {"plan_path": str(path), "expected_cases": result["expected_cases"],
                      "native_execution_ready": result["native_execution_ready"],
                      "blockers": result["blockers"], "native_started": False}
        elif args.check:
            result = cpu_preflight(ROOT, plan)
        elif args.worker_case:
            result = worker_case(ROOT, plan, args.worker_case, device=args.device, resource_slot=args.resource_slot, parallel=args.parallel_native)
        elif args.launch_case:
            result = launch_case(ROOT, plan, args.launch_case, timeout=args.case_timeout, device=args.device,
                                 resource_slot=args.resource_slot, parallel=args.parallel_native)
        elif args.audit_native_train:
            result = audit_training(ROOT, plan)
        else:
            result = audit_validation(ROOT, plan)
    except (EvaluationBlocked, OSError, ValueError) as error:
        print(json.dumps({"blocked": True, "reason": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    if args.worker_case and not result.get("complete"):
        return 2 if result.get("administrative_stop") else 1
    if args.launch_case and not result.get("case_complete"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
