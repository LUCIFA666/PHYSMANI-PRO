"""Train pipeline v1 causal context/SimNorm dynamics on PhysMani TRAIN only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from our_mani.context_dynamics import ContextDynamicsNetwork, history_features, normalized_state


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def examples(manifest, steps=22):
    samples = []
    for group, entry in enumerate(manifest['entries']):
        if entry['split'] not in ('train', 'validation'):
            raise ValueError('latent dynamics training cannot read TEST trajectories')
        path = ROOT/entry['cache']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['cache_sha256']:
            raise ValueError('causal track cache checksum differs')
        with np.load(path) as data:
            times, p, v = data['times'], data['positions'], data['velocities']
            valid, truth = data['valid'], data['truth_metric_only']
            for anchor in range(4, len(times)-1, 2):
                begin = max(0, anchor-19)
                while begin < anchor and not valid[begin:anchor+1].all():
                    begin += 1
                if anchor-begin+1 < 5:
                    continue
                features, length, state = history_features(times[begin:anchor+1], p[begin:anchor+1],
                    v[begin:anchor+1], data['confidence'][begin:anchor+1])
                future = truth[anchor:min(len(truth), anchor+steps+1)]
                # Future labels supervise forecasts; they never enter history.
                count = len(future)
                target = np.zeros((steps+1, 3), dtype=np.float32)
                target[:count] = future
                mask = np.zeros(steps+1, dtype=bool)
                mask[1:count] = np.isfinite(future[1:]).all(1)
                if not mask.any():
                    continue
                samples.append({'history': features, 'length': length, 'state': state, 'target': target,
                    'mask': mask, 'group': group, 'task': entry['task'], 'episode': entry['raw_episode_id'],
                    'split': entry['split'], 'anchor': anchor})
    return samples


def tensors(samples, device):
    return {key: torch.tensor(np.asarray([r[key] for r in samples]), device=device)
            for key in ('history', 'length', 'state', 'target', 'mask', 'group')}


def loss_for(model, data, indices, steps):
    prediction = model(data['history'][indices], data['length'][indices], data['state'][indices], steps)
    mask = data['mask'][indices, :steps+1]
    error = (prediction['positions']-data['target'][indices, :steps+1])/.1
    loss = torch.square(error)[mask].mean()
    loss = loss+.01*torch.square(prediction['reconstructed_state']-data['state'][indices]).mean()
    # Weak within-episode context consistency: state/phase should live in z.
    context, groups = prediction['context'], data['group'][indices]
    unique, inverse = groups.unique(return_inverse=True)
    sums = context.new_zeros((len(unique), context.shape[-1])).index_add_(0, inverse, context)
    counts = torch.bincount(inverse, minlength=len(unique)).to(context.dtype)[:, None]
    loss = loss+.001*torch.square(context-sums[inverse]/counts[inverse]).mean()
    return loss


@torch.inference_mode()
def evaluate(model, data, samples, device):
    model.eval()
    chunks = []
    for begin in range(0, len(samples), 128):
        sl = slice(begin, begin+128)
        chunks.append(model(data['history'][sl], data['length'][sl], data['state'][sl])['positions'].cpu().numpy())
    prediction = np.concatenate(chunks)
    result = {}
    for name in ('context_model', 'constant_velocity'):
        forecast = prediction if name == 'context_model' else (
            np.array([s['state'][:3]*np.array([.25,.6,.1])+np.array([.2,0,.775]) for s in samples])[:, None]
            +np.arange(23)[None, :, None]*.05*np.array([s['state'][3:]*.5 for s in samples])[:, None])
        horizons = {}
        for seconds in (.05, .1, .25, .3, .5, .6, 1.1):
            k = int(round(seconds/.05))
            by_episode = {}
            for i, sample in enumerate(samples):
                if not sample['mask'][k]:
                    continue
                error = np.linalg.norm(forecast[i, 1:k+1]-sample['target'][1:k+1], axis=-1)
                by_episode.setdefault((sample['task'], sample['episode']), []).append((error.mean(), error[-1]))
            values = np.array([np.mean(v, axis=0) for v in by_episode.values()])
            horizons[str(seconds)] = {'episodes': len(values), 'windows': sum(map(len, by_episode.values())),
                'macro_mean_position_error_m': float(values[:, 0].mean()) if len(values) else None,
                'macro_endpoint_error_m': float(values[:, 1].mean()) if len(values) else None}
        result[name] = horizons
    model.train()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='artifacts/framework/pipeline_v1/context_tracks/tracks_manifest.json')
    parser.add_argument('--out', default='artifacts/framework/pipeline_v1/dynamics_simnorm')
    parser.add_argument('--normalization', choices=('simnorm', 'layernorm'), default='simnorm')
    parser.add_argument('--updates', type=int, default=3000)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    out = ROOT/args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out/'dynamics.pt').exists():
        raise FileExistsError('choose a new frozen context model version')
    torch.set_num_threads(2)
    torch.manual_seed(916)
    manifest_path = ROOT/args.manifest
    manifest = json.loads(manifest_path.read_text())
    samples = examples(manifest)
    groups = sorted({s['group'] for s in samples if s['split'] == 'train'})
    rng = np.random.default_rng(916)
    held = set()
    for task in sorted({s['task'] for s in samples if s['split'] == 'train'}):
        task_groups = sorted({s['group'] for s in samples if s['split'] == 'train' and s['task'] == task})
        held.update(task_groups[i] for i in rng.permutation(len(task_groups))[:max(1,len(task_groups)//5)])
    fit = [s for s in samples if s['split'] == 'train' and s['group'] not in held]
    select = [s for s in samples if s['split'] == 'train' and s['group'] in held]
    val = [s for s in samples if s['split'] == 'validation']
    if not fit or not select or not val:
        raise ValueError('fit, grouped selection and independent VAL are required')
    device = torch.device(args.device)
    fit_data, select_data, val_data = [tensors(s, device) for s in (fit, select, val)]
    model = ContextDynamicsNetwork(normalization=args.normalization).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best, best_state, best_update, history = float('inf'), None, None, []
    started = time.perf_counter()
    for update in range(args.updates):
        indices = torch.randint(len(fit), (min(64, len(fit)),), device=device)
        steps = min(22, 6+update//100)
        loss = loss_for(model, fit_data, indices, steps)
        if not torch.isfinite(loss):
            raise ValueError('context dynamics loss became non-finite')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if update % 100 == 0 or update == args.updates-1:
            model.eval()
            with torch.inference_mode():
                total, count = 0., 0
                for begin in range(0,len(select),128):
                    ii = torch.arange(begin,min(begin+128,len(select)),device=device)
                    total += float(loss_for(model, select_data, ii, 22))*len(ii)
                    count += len(ii)
                score = total/count
            model.train()
            row = {'update':update+1,'training_loss':float(loss.detach()),'TRAIN_group_selection_loss':score,
                   'elapsed_seconds':time.perf_counter()-started,'curriculum_steps':steps}
            history.append(row)
            if score < best:
                best, best_update = score, update+1
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            write(out/'status.json', {'stage':'training_context_latent_dynamics',**row,'best_update':best_update,'formal_test_used':False})
            print(json.dumps(row),flush=True)
    model.load_state_dict(best_state)
    model.eval()
    metadata = {'created_utc':datetime.now(timezone.utc).isoformat(),'version':'pipeline_v1_context_simnorm',
        'encoder_sha256':model.encoder_sha256(),'maximum_training_horizon_seconds':1.1,
        'selected_update':best_update,'completed_updates':args.updates,'formal_test_used':False,
        'selection':'TRAIN episode-group-heldout, not independent VAL or TEST',
        'track_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'fit_episodes':len(groups)-len(held),'selection_episodes':len(held),
        'fit_windows':len(fit),'selection_windows':len(select),'validation_windows':len(val),
        'context_dim':16,'latent_dim':64,'history_frames':[5,20],
        'context_physical_interpretability_verified':False,'ngff_used':False,
        'parameters':sum(p.numel() for p in model.parameters())}
    torch.save({'spec':model.spec,'state_dict':best_state,'metadata':metadata},out/'dynamics.pt')
    report = {'metadata':metadata,'validation':evaluate(model,val_data,val,device),'history':history,
              'checkpoint_sha256':hashlib.sha256((out/'dynamics.pt').read_bytes()).hexdigest(),
              'paper_reference_comparability':'same forecast horizons; this two-task VAL is not paper all16 TEST'}
    write(out/'report.json',report)
    write(out/'status.json',{'stage':'context_dynamics_training_and_VAL_complete','metadata':metadata,'validation':report['validation']})
    print(json.dumps({k:v for k,v in report.items() if k!='history'},indent=2),flush=True)


if __name__ == '__main__':
    main()
