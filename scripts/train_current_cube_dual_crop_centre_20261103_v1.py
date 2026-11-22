"""Default metadata/source check; optimizer requires explicit --train."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.current_cube_rgbd_centre_20261103_v1.training import check,fit

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest');p.add_argument('--encoder-checkpoint');p.add_argument('--expected-encoder-sha256');p.add_argument('--output');p.add_argument('--device',default='cuda')
 mode=p.add_mutually_exclusive_group();mode.add_argument('--check',action='store_true');mode.add_argument('--train',action='store_true');a=p.parse_args()
 if a.train:
  if not all((a.manifest,a.encoder_checkpoint,a.expected_encoder_sha256,a.output)):p.error('--train requires manifest/encoder/SHA/fresh output')
  r=fit(a.manifest,a.encoder_checkpoint,a.expected_encoder_sha256,a.output,device=a.device)
 else:r=check(a.manifest)
 print(json.dumps(r,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
