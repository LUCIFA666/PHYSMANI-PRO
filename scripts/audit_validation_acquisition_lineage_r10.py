"""Read-only first-execution VAL275 profile-witness audit; no native imports."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask import validation_lineage
from our_mani.multitask.acquisition_lineage import local,pointer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,help='ordinary TRACK manifest with shared_profile_execution_registration')
    parser.add_argument('--allow-partial',action='store_true',help='honest pending report; never a complete consumer release')
    parser.add_argument('--out',default='artifacts/framework/multitask_r10/VAL_lineage_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args=parser.parse_args();output=local(ROOT,args.out)
    if output.exists():raise ValueError('fresh audit output directory required; original artifacts are immutable')
    report=validation_lineage.audit_validation_acquisition_lineage(ROOT,args.manifest,require_complete=not args.allow_partial)
    report['audit_module']=pointer(ROOT,Path(validation_lineage.__file__))
    report['audit_script']=pointer(ROOT,Path(__file__))
    output.mkdir(parents=True)
    path=output/'report.json';path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],verified=report['verified_completed_acquisitions'],expected=275,report=pointer(ROOT,path))))


if __name__=='__main__':main()
