"""CPU-only ring TRAIN grasp/orbit proof; no skill is invented or fitted."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.ring_affordance import build_train_permission,VerifiedRingAffordance,EXACT_ORBIT_SCHEMA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symmetry-report',required=True)
    parser.add_argument('--manifest',default='artifacts/framework/multitask_r10/TRAIN_smoke_collection_v7/manifest.json')
    parser.add_argument('--physmani-root',default='/home/ubuntu/projects/PhysMani')
    parser.add_argument('--library-pointers',help='optional frozen JSON task -> {path,sha256}; no library means pending')
    parser.add_argument('--development-only',action='store_true')
    parser.add_argument('--out',required=True)
    args=parser.parse_args();out=(ROOT/args.out).resolve()
    if not out.is_relative_to(ROOT/'artifacts/framework/multitask_r10') or out.exists():
        raise ValueError('fresh desktop R10 proof directory required')
    def pointer(path):
        path=Path(path).resolve()
        if not path.is_relative_to(ROOT):raise ValueError('proof pointers must remain in desktop project')
        return dict(path=str(path.relative_to(ROOT)),sha256=file_sha(path))
    report_path=(ROOT/args.symmetry_report).resolve();source=json.loads(report_path.read_text())
    manifest=pointer(ROOT/args.manifest);base=Path(args.physmani_root).resolve()
    libraries=json.loads((ROOT/args.library_pointers).read_text()) if args.library_pointers else {}
    out.mkdir(parents=True);rows=[]
    for row in source['entries']:
        task=row['task'];value=build_train_permission(ROOT,row['certificate'],manifest,
            native_gripper_source=base/'third_party/PyRep/pyrep/robots/end_effectors/gripper.py',
            native_task_source=base/'third_party/RLBench/rlbench/tasks'/(task+'.py'),
            library_pointer=libraries.get(task),development_only=args.development_only)
        path=out/(task+'__ring_phase_permission.json');atomic_json(path,value)
        loaded=VerifiedRingAffordance.load(ROOT,pointer(path),allow_development=args.development_only)
        rows.append(dict(task=task,permission=pointer(path),status=value['status'],
            exact_action_orbit_schema=EXACT_ORBIT_SCHEMA,all_permission_fields_replayed_exactly=True,
            actual_TRAIN_grasp_parents=len(value['actual_TRAIN_grasp_rows']),
            phase_support=value['phase_support'],policy_skill_authorized=value['policy_skill_authorized'],
            reproducible_loaded_permission=loaded.permission==value,full_orientation_observable=False))
    report=dict(version='r10_TRAIN_ring_grasp_orbit_affordance_diagnostic_v2',complete=True,
        source_symmetry_report=pointer(report_path),source_manifest=manifest,source_script=pointer(__file__),
        entries=rows,TRAIN_only=True,VAL_read=False,TEST_read=False,native_or_GPU_launched=False,
        model_fitted=False,actor_success_rate_evaluated=False,full_orientation_promoted=False,
        only_ring_approach_acquire_can_be_authorized=True,formal_comparison_ready=False)
    atomic_json(out/'report.json',report)
    print(json.dumps(dict(report=pointer(out/'report.json'),entries=rows),ensure_ascii=False))


if __name__=='__main__':main()
