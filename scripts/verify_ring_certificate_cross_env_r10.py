"""CPU-only actual-environment replay of freshly frozen ring evidence."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import scipy
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.rotation_equivalence import VerifiedFiniteSymmetry
from our_mani.multitask.ring_affordance import VerifiedRingAffordance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--certificates',default='artifacts/framework/multitask_r10/rotation_equivalence_runtime_v3/report.json')
    parser.add_argument('--permissions',default='artifacts/framework/multitask_r10/ring_affordance_runtime_v2/report.json')
    args=parser.parse_args();out=(ROOT/args.out).resolve()
    if not out.is_relative_to(ROOT/'artifacts/framework/multitask_r10') or out.exists():raise ValueError('fresh R10 output required')
    certificates=json.loads((ROOT/args.certificates).read_text());permissions=json.loads((ROOT/args.permissions).read_text())
    rows=[]
    for row in certificates['entries']:
        value=VerifiedFiniteSymmetry.load(ROOT,row['certificate'])
        rows.append(dict(task=row['task'],certificate_loaded=True,
            certificate_file_SHA_matches=file_sha(ROOT/row['certificate']['path'])==row['certificate']['sha256'],
            replay_receipt=value.replay_receipt))
    permit_rows=[]
    for row in permissions['entries']:
        try:
            value=VerifiedRingAffordance.load(ROOT,row['permission'])
            permit_rows.append(dict(task=row['task'],permission_loaded=True,policy_skill_authorized=value.permission['policy_skill_authorized']))
        except Exception as error:
            permit_rows.append(dict(task=row['task'],permission_loaded=False,error=type(error).__name__+': '+str(error)))
    output=dict(version='r10_actual_environment_ring_evidence_replay_v1',environment=args.environment,
        python=sys.version,numpy=np.__version__,scipy=scipy.__version__,native_simulator_started=False,
        old_evidence_files_modified=False,entries=rows,permission_entries=permit_rows,
        complete=all(r['certificate_loaded'] for r in rows))
    atomic_json(out,output)
    print(json.dumps(dict(environment=args.environment,complete=output['complete'],report=str(out.relative_to(ROOT)),
        certificate_difference_counts=[r['replay_receipt']['different_derived_float_fields'] for r in rows],
        permission_entries=permit_rows),ensure_ascii=False))


if __name__=='__main__':main()
