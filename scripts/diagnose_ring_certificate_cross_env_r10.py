"""Read-only code/CAD diagnosis of ring certificate cross-environment replay."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import scipy
from our_mani.multitask.evaluation import file_sha,atomic_json
from our_mani.multitask.rotation_equivalence import build_train_certificate


def differences(a,b,path='$'):
    if type(a)!=type(b):return [dict(path=path,kind='type',stored=type(a).__name__,replayed=type(b).__name__)]
    if isinstance(a,dict):
        result=[]
        for key in sorted(set(a)|set(b)):
            if key not in a or key not in b:result.append(dict(path=path+'.'+key,kind='missing_key'))
            else:result.extend(differences(a[key],b[key],path+'.'+key))
        return result
    if isinstance(a,list):
        if len(a)!=len(b):return [dict(path=path,kind='length',stored=len(a),replayed=len(b))]
        return [r for i,(x,y) in enumerate(zip(a,b)) for r in differences(x,y,path+'['+str(i)+']')]
    if a==b:return []
    if isinstance(a,float):
        return [dict(path=path,kind='float',stored=a,replayed=b,absolute_delta=abs(a-b),
            relative_delta=abs(a-b)/max(abs(a),abs(b)) if max(abs(a),abs(b)) else 0.,
            maximum_ULP_units=abs(a-b)/max(math.ulp(a),math.ulp(b)))]
    return [dict(path=path,kind='value',stored=a,replayed=b)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True);parser.add_argument('--environment',required=True)
    parser.add_argument('--report',default='artifacts/framework/multitask_r10/rotation_equivalence_runtime_v2/report.json')
    args=parser.parse_args();out=(ROOT/args.out).resolve()
    if not out.is_relative_to(ROOT/'artifacts/framework/multitask_r10') or out.exists():raise ValueError('fresh R10 output required')
    out.mkdir(parents=True);report=json.loads((ROOT/args.report).read_text());rows=[]
    for item in report['entries']:
        pointer=item['certificate'];path=ROOT/pointer['path'];value=json.loads(path.read_text())
        checks={name:file_sha(ROOT/name)==checksum for name,checksum in value['inputs_sha256'].items()}
        expected=build_train_certificate(ROOT,value['CAD_file'],value['body_id'],value['group_rotations_body'],
            value['center_body'],TRAIN_manifest_pointer=value['TRAIN_acquisition_manifest'],
            TRAIN_receipt_pointer=value['TRAIN_acquisition_receipt'],surface_tolerance_m=value['surface_tolerance_m'])
        destination=out/(item['task']+'__replayed.json');atomic_json(destination,expected)
        diff=differences(value,expected)
        rows.append(dict(task=item['task'],stored_file_SHA_matches=file_sha(path)==pointer['sha256'],
            every_input_SHA_matches=all(checks.values()),failed_inputs=[k for k,v in checks.items() if not v],
            build_predicates_passed=True,exact_dict_equal=value==expected,difference_count=len(diff),differences=diff,
            replayed_file=dict(path=str(destination.relative_to(ROOT)),sha256=file_sha(destination))))
    output=dict(version='r10_read_only_finite_certificate_cross_environment_diagnosis_v1',
        environment=args.environment,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
        native_simulator_started=False,locked_code_modified=False,old_certificate_modified=False,entries=rows)
    atomic_json(out/'report.json',output)
    print(json.dumps(output,ensure_ascii=False))


if __name__=='__main__':main()
