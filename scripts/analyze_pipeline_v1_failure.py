"""Frozen-v1 failure evidence and VAL-only history/context interventions."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
from our_mani.context_dynamics import ContextLatentDynamics
from train_context_dynamics import examples,tensors


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    out=ROOT/'artifacts/framework/pipeline_v1/failure_analysis'
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'artifacts/framework/pipeline_v1/context_tracks/tracks_manifest.json').read_text())
    # Require at least ten historical observations in every comparison so the
    # five-frame intervention actually removes earlier history.
    samples=[s for s in examples(manifest) if s['split']=='validation' and s['length']>=10]
    data=tensors(samples,'cpu')
    dynamics=ContextLatentDynamics(ROOT/'artifacts/framework/pipeline_v1/dynamics_simnorm_v1/dynamics.pt')
    model=dynamics.model.eval()
    context=model.encode_context(data['history'],data['length'])
    short=data['history'].clone().zero_()
    for i,s in enumerate(samples):short[i,:5]=data['history'][i,s['length']-5:s['length']]
    shuffle=np.zeros(len(samples),dtype=int)
    for task in sorted({s['task'] for s in samples}):
        ids=[i for i,s in enumerate(samples) if s['task']==task]
        # Match task, but use another raw episode's context. No future labels
        # select pairs, and no retraining is performed with these interventions.
        for i in ids:
            peers=[j for j in ids if samples[j]['episode']!=samples[i]['episode']]
            shuffle[i]=peers[(i*103+916)%len(peers)]
    predictions={}
    for mode in ('full_history','last_five_frames','uniform_context_intervention','other_episode_context_same_task'):
        chunks=[]
        for begin in range(0,len(samples),128):
            sl=slice(begin,begin+128)
            history,lengths=data['history'][sl],data['length'][sl]
            override=None
            if mode=='last_five_frames':history,lengths=short[sl],torch.full_like(lengths,5)
            elif mode=='uniform_context_intervention':override=torch.full_like(context[sl],1/8)
            elif mode=='other_episode_context_same_task':override=context[shuffle[sl]]
            chunks.append(model(history,lengths,data['state'][sl],context_override=override)['positions'].numpy())
        predictions[mode]=np.concatenate(chunks)
    results={}
    for mode,prediction in predictions.items():
        results[mode]={}
        for seconds in (.1,.5,1.1):
            k=round(seconds/.05);groups={}
            for i,s in enumerate(samples):
                if not s['mask'][k]:continue
                error=float(np.linalg.norm(prediction[i,k]-s['target'][k]))
                groups.setdefault((s['task'],s['episode']),[]).append(error)
            results[mode][str(seconds)]={'macro_endpoint_error_m':float(np.mean([np.mean(v) for v in groups.values()])),
                'episodes':len(groups),'windows':sum(map(len,groups.values())),
                'per_task_macro_endpoint_error_m':{task:(float(np.mean([np.mean(v) for key,v in groups.items() if key[0]==task]))
                    if any(key[0]==task for key in groups) else None)
                    for task in sorted({s['task'] for s in samples})}}
    audit=json.loads((ROOT/'artifacts/framework/pipeline_v1/TRAIN_handover_audit/report.json').read_text())
    nearest=[]
    for row in audit['summaries']:
        detail=json.loads((ROOT/'artifacts/framework/pipeline_v1/TRAIN_handover_audit'/f'{row["task"]}.json').read_text())
        eligible=[g for g in detail['gates'] if g['stage']=='approach_faster' and g['position_metric'] is not None]
        g=min(eligible,key=lambda g:g['position_metric'])
        nearest.append({'task':row['task'],'raw_episode_id':row['raw_episode_id'],
            'nearest_in_window_distance_m':float(np.linalg.norm(np.array(g['robot_pose'][:3])-g['current_target_region_pose'][:3])),
            'orientation_error_rad':g['orientation_error'],'relative_speed_mps':g['relative_speed'],
            'position_passes':sum(g['position_metric']<=1 for g in eligible),'in_window_observations':len(eligible)})
    report={'kind':'frozen_v1_TRAIN_audit_and_VAL_interventions','formal_test_used_for_diagnosis':False,
        'dynamics_sha256':dynamics.checkpoint_sha256,'encoder_sha256':dynamics.encoder_sha256,
        'context_interventions':results,'nearest_handover_errors':nearest,
        'limitations':['context interventions are diagnostics without retraining, not a matched-training ablation',
            'history cohorts require >=10 observed frames and differ from the main VAL report',
            'servo and FASTER causality require controlled TRAIN counterfactuals',
            'two audited TRAIN episodes do not estimate generalization success']}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
