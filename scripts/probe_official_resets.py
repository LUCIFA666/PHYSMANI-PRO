"""Audit official TEST reset plumbing only, without policy inference or tuning."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from our_mani.official_simulation import OfficialTestSimulation


def main():
    manifest=json.loads((ROOT/'artifacts/framework/pipeline_v1/official_test_manifest.json').read_text())
    out=ROOT/'artifacts/framework/pipeline_v1/reset_audit'
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for task in sorted({r['task'] for r in manifest['episodes']}):
        entries=[r for r in manifest['episodes'] if r['task']==task]
        environment=OfficialTestSimulation(task,entries[0]['path'])
        try:
            for variation in (0,1):
                row=next(r for r in entries if r['variation']==variation)
                _,observation=environment.reset_official(row)
                images=observation['observation/images']
                mosaic=np.concatenate([np.concatenate([images[n] for n in ('front','8')],1),
                                       np.concatenate([images[n] for n in ('16','36')],1)],0)
                Image.fromarray(mosaic).save(out/f'{task}_variation{variation}.png')
                rows.append(environment.reset_receipt)
        finally:
            environment.close()
    (out/'report.json').write_text(json.dumps({'kind':'official_reset_infrastructure_audit_only',
        'policy_inference_run':False,'models_selected_using_test':False,'resets':rows},indent=2)+'\n')
    print(json.dumps({'resets':len(rows),'policy_inference_run':False,'output':str(out)}))


if __name__=='__main__':main()
