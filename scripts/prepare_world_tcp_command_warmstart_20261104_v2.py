"""Fixed max16/stage-preserving pilot; default reads only existing metadata."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.world_tcp_command_warmstart_20261104_v2.data import metadata_check,prepare_subset
from our_mani.world_tcp_command_warmstart_20261104_v2.model import parameter_count


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-index',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--prepare-subset',action='store_true')
    args=parser.parse_args(argv)
    if args.prepare_subset:
        if args.output is None:parser.error('--prepare-subset requires fresh --output')
        result=prepare_subset(args.full_index,args.output)
        result={key:result[key]for key in ('complete','role_counts','parent_counts','maximum_stage_count','RGBD_reads','NN_calls','GPU','Native','TEST_reads')}
    else:result=metadata_check(args.full_index)
    result['model_exact_parameters']=parameter_count()
    print(json.dumps(result,ensure_ascii=False,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
