"""CPU-only comparison of saved R10 initial parent-metric conditions.

New tolerance fields cannot pass the formal gate before frozen TRAIN repeated
reset fitting/validation evidence is supplied. This script never launches a
simulator, reads TEST success outcomes for tuning, or changes any threshold.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.reset_parity import (pending_tolerance_profile,
    compare_initial_conditions, compare_five_way)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def destination(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to((ROOT / 'artifacts/framework/multitask_r10').resolve()):
        raise ValueError('new audit outputs must stay inside artifacts/framework/multitask_r10')
    if path.exists():
        raise FileExistsError('preserve existing audit outputs; choose a new path')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--create-pending-profile', metavar='NEW_JSON')
    parser.add_argument('--a', help='first saved record containing recorded_initial_conditions')
    parser.add_argument('--b', help='second saved record')
    parser.add_argument('--five-way', help='JSON mapping all five method keys to saved record paths')
    parser.add_argument('--profile')
    parser.add_argument('--out')
    parser.add_argument('--diagnostic-only', action='store_true', help='show provisional exact-only differences; not formal acceptance')
    args = parser.parse_args()
    if args.create_pending_profile:
        if any((args.a, args.b, args.five_way, args.profile, args.out)):
            parser.error('pending profile creation cannot be combined with comparison')
        path = destination(args.create_pending_profile)
        legacy = ROOT / 'artifacts/framework/aligned_r09/acceptance_tolerances.json'
        result = pending_tolerance_profile(read(legacy))
        result['inherited_profile_source'] = dict(path=str(legacy.relative_to(ROOT)), sha256=sha(legacy))
    else:
        if not args.profile or not args.out or (bool(args.five_way) == bool(args.a or args.b)):
            parser.error('supply --profile --out and either --a/--b or --five-way')
        if not args.five_way and not (args.a and args.b):
            parser.error('paired comparison needs both --a and --b')
        profile_path = ROOT / args.profile
        profile = read(profile_path);inputs = [profile_path]
        if args.five_way:
            mapping_path = ROOT / args.five_way;mapping = read(mapping_path);inputs.append(mapping_path)
            paths = {method: ROOT / name for method, name in mapping.items()};inputs.extend(paths.values())
            result = compare_five_way({method: read(path) for method, path in paths.items()}, profile,
                evidence_root=ROOT, require_frozen=not args.diagnostic_only)
        else:
            paths = [ROOT / args.a, ROOT / args.b];inputs.extend(paths)
            result = compare_initial_conditions(*(read(path) for path in paths), profile,
                evidence_root=ROOT, require_frozen=not args.diagnostic_only)
        path = destination(args.out)
        result.update(created_utc=datetime.now(timezone.utc).isoformat(), diagnostic_only=args.diagnostic_only,
            metric_only=True, policy_use_allowed=False, simulation_or_GPU_launched=False,
            inputs_sha256={str(item.resolve().relative_to(ROOT.resolve())): sha(item) for item in inputs},
            comparer_source_sha256=sha(ROOT / 'our_mani/multitask/reset_parity.py'),
            entrypoint_source_sha256=sha(Path(__file__)))
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(path.relative_to(ROOT)), accepted=result.get('accepted'),
                         status=result.get('status')), ensure_ascii=False))
    if not args.create_pending_profile and not result['accepted']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
