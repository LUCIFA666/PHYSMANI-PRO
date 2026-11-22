"""Reserved-head evaluation only after selection: --predict freeze then --score."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.current_cube_rgbd_centre_20261103_v1.training import check
from our_mani.current_cube_rgbd_centre_20261103_v1.evaluation import predict,score

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest');p.add_argument('--selected');p.add_argument('--expected-sha256');p.add_argument('--encoder-checkpoint');p.add_argument('--output')
 mode=p.add_mutually_exclusive_group();mode.add_argument('--check',action='store_true');mode.add_argument('--predict',action='store_true');mode.add_argument('--score',action='store_true');a=p.parse_args()
 if a.predict:
  if not all((a.manifest,a.selected,a.expected_sha256,a.encoder_checkpoint,a.output)):p.error('--predict requires manifest/selected/SHA/encoder/fresh output')
  r=predict(a.selected,a.expected_sha256,a.manifest,a.output,a.encoder_checkpoint)
 elif a.score:
  if not a.output:p.error('--score requires frozen output')
  r=score(a.output)
 else:r=check(a.manifest)
 print(json.dumps(r,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
