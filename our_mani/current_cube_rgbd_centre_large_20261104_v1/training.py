"""Large centre model: fixed40 epochs, weighted micro2x8, explicit GPU preflight.

Default checks construct no model/optimizer and read no NPZ arrays. Only Root
invokes the explicit preflight/train entrypoints after the complete40 closure.
"""
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import numpy as np

from . import VERSION
from .config import fixed_config, PROTOCOL_PATH, PROTOCOL_SHA256, ENCODER_SHA256
from ..current_cube_rgbd_centre_20261103_v1.data import ROOT, DualCropCubeDataset, sha256, checked_pointer
from ..current_cube_rgbd_centre_20261103_v1 import training as legacy

pointer, write_new = legacy.pointer, legacy.write_new
cache_role, tensor_batch, array_digest = legacy.cache_role, legacy.tensor_batch, legacy.array_digest
INPUT_KEYS = ('features', 'wide_features', 'K', 'camera_rotations', 'initial_sphere')


def training_source_map():
    # Preserve the exact old data/feature/encoder transfer and loss definitions.
    sources = legacy.training_source_map()
    for name in ('__init__.py', 'model.py', 'config.py', 'training.py', 'evaluation.py'):
        path = Path(__file__).with_name(name)
        sources[str(path)] = sha256(path)
    for name in ('train_current_cube_dual_crop_centre_large_20261104_v1.py',
                 'evaluate_current_cube_dual_crop_centre_large_20261104_v1.py'):
        path = ROOT / 'scripts' / name
        sources[str(path)] = sha256(path)
    sources[str(PROTOCOL_PATH)] = PROTOCOL_SHA256
    return sources


def check(manifest=None):
    config = fixed_config()
    out = dict(version=VERSION, config=config, source_sha256=training_source_map(),
        protocol=pointer(PROTOCOL_PATH), dataset_ready=False,
        model_constructed=False, optimizer_constructed=False, model_forward_calls=0,
        GPU_used=False, Native_started=False, training_started=False,
        model_qualified=False, runtime_or_control_admitted=False)
    if manifest is not None:
        fit = DualCropCubeDataset(manifest, 'fit')
        selection = DualCropCubeDataset(manifest, 'selection')
        if len(fit) != 208 or len(selection) != 208:
            raise ValueError('Complete original16fit/16selection/8reserved40parents required')
        fit.revalidate(); selection.revalidate()
        out.update(dataset_ready=True, manifest=pointer(manifest), fit_samples=208,
            selection_samples=208, reserved_getters=0, arrays_decoded=0)
    return out


def loss_from_outputs(output, batch):
    import torch
    from ..current_cube_rgbd_centre_20261103_v1.model import centre_presence_loss
    # Loss math stays FP32 even while the large network uses BF16 autocast.
    with torch.autocast(device_type=output['centre_world'].device.type, enabled=False):
        converted = {k: v.float() if getattr(v, 'is_floating_point', lambda: False)() else v
                     for k, v in output.items()}
        return centre_presence_loss(converted, batch['centre_target_world'].float(),
            batch['centre_valid'], batch['presence_target'])


def weighted_micro_loss(loss, logical_centre_count, logical_known_count):
    """Exact logical-batch means; never an unweighted mean of eight means."""
    if min(logical_centre_count, logical_known_count) < 0:
        raise ValueError('Nonnegative whole logical-batch counts required')
    return (loss['centre'] * loss['centre_samples'] / max(logical_centre_count, 1)
            + loss['presence'] * loss['known_presence_samples'] / max(logical_known_count, 1))


def logical_batch_backward(model, arrays, indices, device, config):
    import torch
    presence = arrays['presence_target'][indices]
    nc = int(np.count_nonzero(arrays['centre_valid'][indices] & (presence == 1)))
    nk = int(np.count_nonzero(presence >= 0))
    values, centre_sum, presence_sum, calls = [], 0., 0., 0
    for start in range(0, len(indices), config['micro_batch_size']):
        ids = indices[start:start + config['micro_batch_size']]
        batch = tensor_batch(arrays, ids, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=str(device).startswith('cuda')):
            output = model(**{k: batch[k] for k in INPUT_KEYS})
        loss = loss_from_outputs(output, batch)
        value = weighted_micro_loss(loss, nc, nk)
        if not bool(torch.isfinite(value)):
            raise ValueError('Nonfinite logical-batch supervised loss')
        value.backward()
        values.append(float(value.detach())); calls += 1
        centre_sum += float(loss['centre'].detach()) * loss['centre_samples']
        presence_sum += float(loss['presence'].detach()) * loss['known_presence_samples']
    return dict(total=sum(values), centre=centre_sum / max(nc, 1),
        presence=presence_sum / max(nk, 1), centre_samples=nc,
        known_presence_samples=nk, micro_forward_calls=calls)


def selection_loss(model, arrays, config, device):
    import torch
    model.eval(); sums = dict(centre=0., presence=0.); nc = nk = calls = 0
    with torch.no_grad():
        for start in range(0, len(arrays['presence_target']), config['micro_batch_size']):
            ids = np.arange(start, min(start + config['micro_batch_size'], len(arrays['presence_target'])))
            batch = tensor_batch(arrays, ids, device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=str(device).startswith('cuda')):
                output = model(**{k: batch[k] for k in INPUT_KEYS})
            loss = loss_from_outputs(output, batch); calls += 1
            nc += loss['centre_samples']; nk += loss['known_presence_samples']
            sums['centre'] += float(loss['centre']) * loss['centre_samples']
            sums['presence'] += float(loss['presence']) * loss['known_presence_samples']
    centre, presence = sums['centre'] / max(nc, 1), sums['presence'] / max(nk, 1)
    return dict(total=centre + presence, centre=centre, presence=presence,
        centre_samples=nc, known_presence_samples=nk, forward_batch_calls=calls,
        samples=len(arrays['presence_target']))


def _process_inventory():
    known = []
    for item in Path('/proc').iterdir():
        if not item.name.isdigit() or int(item.name) == os.getpid(): continue
        try:
            args = [a.decode(errors='replace') for a in (item / 'cmdline').read_bytes().split(b'\0') if a]
        except (OSError, PermissionError): continue
        if not args or '--check' in args: continue
        scripts = [Path(a).name for a in args if a.endswith('.py')]
        native = any((s.startswith('collect_current_cube_dual_crop') and '--capture' in args)
            or s.startswith(('run_position_', 'run_local_mpc_position_')) for s in scripts)
        native |= any(Path(a).name.lower() in ('coppeliasim', 'vrep') for a in args[:1])
        if native: known.append(dict(pid=int(item.name), argv=args))
    return known


def _other_GPU_compute_processes():
    completed = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    rows = []
    for line in completed.stdout.splitlines():
        if not line.strip(): continue
        values = [v.strip() for v in line.split(',', 2)]
        if len(values) != 3 or not values[0].isdigit():
            raise ValueError('Cannot verify actual GPU compute process inventory')
        if int(values[0]) != os.getpid():
            rows.append(dict(pid=int(values[0]), process_name=values[1], used_memory_MiB=values[2]))
    return rows


@contextmanager
def exclusive_resources():
    files = []
    try:
        for name in ('fixed_TRAIN104_geometry_native.lock', 'TRAIN_postcomplete_diagnostics_v1106.lock',
                     'current_cube_centre_large_training_20261104_v1.lock'):
            path = ROOT / 'runtime/framework' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open('a'); files.append(stream)
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        processes = _process_inventory()
        if processes: raise ValueError('Known Native process is still active: ' + str(processes))
        compute = _other_GPU_compute_processes()
        if compute: raise ValueError('Another actual GPU compute process is active: ' + str(compute))
        yield dict(dual_Native_locks_held=True, training_lock_held=True,
            known_Native_processes=processes, other_GPU_compute_processes=compute)
    finally:
        for stream in reversed(files): stream.close()


def _seed_and_model(encoder_checkpoint, device, config):
    import torch
    from .model import LargeDualCropCubeCentreV1, CentreConfig, ENCODER_PREFIXES
    if str(device) != 'cuda:0' or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('Actual CUDA0 BF16 required; never silently use another device/precision')
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.set_num_threads(config['training_num_threads'])
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    model = LargeDualCropCubeCentreV1(CentreConfig(**config['architecture']), encoder_trainable=True)
    initialization = model.load_selected_v5_encoder(encoder_checkpoint, ENCODER_SHA256)
    if sum(p.numel() for p in model.parameters()) != config['expected_parameters']:
        raise ValueError('Fixed large model parameter count changed')
    encoder = [p for n, p in model.named_parameters() if n.startswith(ENCODER_PREFIXES)]
    heads = [p for n, p in model.named_parameters() if not n.startswith(ENCODER_PREFIXES)]
    if sum(p.numel() for p in encoder) != config['encoder_parameters']:
        raise ValueError('Original exact74520 encoder prefix required')
    model.to(device)
    optimizer = torch.optim.AdamW([dict(params=encoder, lr=config['encoder_learning_rate'], group_name='encoder'),
        dict(params=heads, lr=config['head_learning_rate'], group_name='heads')], weight_decay=config['weight_decay'])
    return model, optimizer, encoder, initialization


def _datasets(manifest, encoder_checkpoint, expected_encoder_sha256):
    if expected_encoder_sha256 != ENCODER_SHA256 or sha256(encoder_checkpoint) != ENCODER_SHA256:
        raise ValueError('Exact dbf original encoder initializer required')
    fit, selection = DualCropCubeDataset(manifest, 'fit'), DualCropCubeDataset(manifest, 'selection')
    if len(fit) != 208 or len(selection) != 208:
        raise ValueError('Whole original40parents must be closed before centre GPU use')
    fit.revalidate(); selection.revalidate()
    return fit, selection


def gpu_preflight(manifest, encoder_checkpoint, expected_encoder_sha256, output, *, device='cuda:0'):
    config = fixed_config(); sources = training_source_map()
    with exclusive_resources() as resources:
        fit_data, selection_data = _datasets(manifest, encoder_checkpoint, expected_encoder_sha256)
        out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
        write_new(out / 'request.json', dict(version=VERSION, config=config, source_sha256=sources,
            manifest=pointer(manifest), initializer=pointer(encoder_checkpoint), resources=resources,
            fixed_fit_indices=list(range(16)), full_encoder_trainable=True,
            optimizer_steps_requested=1, discard_updated_model=True, selected_or_runtime_artifact_created=False))
        report = dict(version=VERSION, passed=False, Native_started=False, reserved_getters=0,
            source_sha256=sources, manifest=pointer(manifest), initializer=pointer(encoder_checkpoint),
            config=config, actual_optimizer_steps=0, model_saved=False)
        try:
            import torch
            arrays, cache = cache_role(fit_data)
            model, optimizer, encoder, initialization = _seed_and_model(encoder_checkpoint, device, config)
            free_before, total = torch.cuda.mem_get_info(); torch.cuda.reset_peak_memory_stats()
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss = logical_batch_backward(model, arrays, np.arange(16), device, config)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
            if not bool(torch.isfinite(norm)): raise ValueError('Nonfinite GPU preflight gradients')
            optimizer.step(); torch.cuda.synchronize(); report['actual_optimizer_steps'] = 1
            peak = torch.cuda.max_memory_reserved(); free_after, _ = torch.cuda.mem_get_info()
            if peak > total * config['preflight_peak_reserved_fraction_max'] or free_after < config['preflight_minimum_free_after_bytes']:
                raise ValueError('Fixed GPU reserved-memory/free-memory gate failed')
            fit_data.revalidate(); selection_data.revalidate()
            if sources != training_source_map(): raise ValueError('Sources changed during preflight')
            report.update(passed=True, loss=loss, gradient_norm=float(norm), initialization=initialization,
                resources=resources, fit_cache=cache, encoder_trainable=True,
                gpu_name=torch.cuda.get_device_name(0), total_memory_bytes=total,
                free_before_bytes=free_before, free_after_bytes=free_after,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=peak,
                environment=dict(torch=str(torch.__version__), numpy=np.__version__, device=device,
                    autocast='bfloat16', loss='float32'), updated_weights_discarded=True,
                no_optimizer_or_model_reused_by_fit=True)
            return report
        except BaseException as error:
            report['error'] = type(error).__name__ + ': ' + str(error)
            raise
        finally:
            write_new(out / 'gpu_preflight.json', report)


def _check_preflight(path, expected_sha256, manifest, encoder_checkpoint, config, sources):
    if sha256(path) != expected_sha256: raise ValueError('External exact GPU-preflight SHA required')
    value = json.loads(Path(path).read_text())
    if (value.get('version') != VERSION or value.get('passed') is not True
            or value.get('actual_optimizer_steps') != 1 or value.get('model_saved') is not False
            or value.get('updated_weights_discarded') is not True
            or value.get('encoder_trainable') is not True or value.get('config') != config
            or value.get('manifest') != pointer(manifest) or value.get('initializer') != pointer(encoder_checkpoint)
            or value.get('source_sha256') != sources):
        raise ValueError('Actual same-config/source/whole40 successful discarded-step preflight required')
    peak, total, free = (value.get(k) for k in ('peak_reserved_bytes', 'total_memory_bytes', 'free_after_bytes'))
    if (any(type(v) is not int or v <= 0 for v in (peak, total, free))
            or peak > total * config['preflight_peak_reserved_fraction_max']
            or free < config['preflight_minimum_free_after_bytes']
            or value.get('resources', {}).get('dual_Native_locks_held') is not True
            or value.get('resources', {}).get('training_lock_held') is not True
            or value.get('environment', {}).get('device') != 'cuda:0'
            or value['environment'].get('autocast') != 'bfloat16'
            or value['environment'].get('loss') != 'float32'):
        raise ValueError('Actual same-GPU resource and memory gates required')
    return value


def fit(manifest, encoder_checkpoint, expected_encoder_sha256, output, preflight, expected_preflight_sha256, *, device='cuda:0'):
    config = fixed_config(); sources = training_source_map()
    admitted_preflight = _check_preflight(preflight, expected_preflight_sha256, manifest, encoder_checkpoint, config, sources)
    with exclusive_resources() as resources:
        fit_data, selection_data = _datasets(manifest, encoder_checkpoint, expected_encoder_sha256)
        out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=False)
        identity = dict(version=VERSION, config=config, source_sha256=sources,
            protocol=pointer(PROTOCOL_PATH), manifest=pointer(manifest), selected_encoder=pointer(encoder_checkpoint),
            actual_gpu_preflight=pointer(preflight), resources=resources, reserved_getters=0,
            data_before=fit_data.revalidate(), role_scope='fit208 optimize; selection208 select epoch; reserved104 not read')
        write_new(out / 'identity.json', identity)
        status = dict(version=VERSION, training_complete=False, actual_optimizer_steps=0,
            model_qualified=False, runtime_or_control_admitted=False)
        steps = train_calls = selection_calls = 0; history = []
        try:
            import torch
            if (torch.cuda.get_device_name(0) != admitted_preflight['gpu_name']
                    or torch.cuda.get_device_properties(0).total_memory != admitted_preflight['total_memory_bytes']
                    or str(torch.__version__) != admitted_preflight['environment']['torch']
                    or np.__version__ != admitted_preflight['environment']['numpy']):
                raise ValueError('Actual training GPU/environment differs from its preflight')
            fit_cache, fit_receipt = cache_role(fit_data)
            selection_cache, sel_receipt = cache_role(selection_data)
            before = dict(fit=fit_receipt, selection=sel_receipt); write_new(out / 'cache_before.json', before)
            model, optimizer, encoder, initialization = _seed_and_model(encoder_checkpoint, device, config)
            rng = np.random.default_rng(config['seed']); best = best_state = best_epoch = None
            for epoch in range(1, 41):
                enabled = epoch > config['head_only_epochs']
                for p in encoder: p.requires_grad_(enabled)
                model.train(); order = rng.permutation(208); logical = []
                for start in range(0, 208, 16):
                    optimizer.zero_grad(set_to_none=True)
                    value = logical_batch_backward(model, fit_cache, order[start:start + 16], device, config)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
                    if not bool(torch.isfinite(norm)): raise ValueError('Nonfinite accumulated gradients')
                    optimizer.step(); steps += 1; train_calls += value['micro_forward_calls']
                    logical.append(dict(**value, gradient_norm=float(norm)))
                selected = selection_loss(model, selection_cache, config, device)
                selection_calls += selected['forward_batch_calls']
                if not np.isfinite(selected['total']): raise ValueError('Nonfinite full208 selection loss')
                if best is None or selected['total'] < best:
                    best, best_epoch = selected['total'], epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                row = dict(epoch=epoch, encoder_trainable=enabled, actual_optimizer_steps=steps,
                    logical_batches=logical, selection=selected, selected_epoch=best_epoch,
                    optimizer_groups=[dict(name=g['group_name'], learning_rate=g['lr']) for g in optimizer.param_groups])
                history.append(row)
                print(json.dumps(dict(epoch=epoch, optimizer_steps=steps, selection_loss=selected['total'])), flush=True)
            if steps != 520 or len(history) != 40 or train_calls != 4160 or selection_calls != 4160:
                raise ValueError('Fixed40epochs/520updates/4160microforwards per role required')
            after = dict(fit=array_digest(fit_cache), selection=array_digest(selection_cache))
            if any(after[k] != before[k]['ordered_content_sha256'] for k in after): raise ValueError('Cache mutated')
            fit_data.revalidate(); selection_data.revalidate()
            if sources != training_source_map(): raise ValueError('Sources changed during fit')
            write_new(out / 'cache_after.json', after); write_new(out / 'history.json', history)
            write_new(out / 'sample_access.json', dict(fit=fit_data.access, selection=selection_data.access, reserved_getters=0))
            write_new(out / 'source_verification.json', dict(source_sha256=sources,
                data_after=fit_data.revalidate(), all_source_and_data_unchanged=True))
            metadata = dict(version=VERSION, training_complete=True, epochs_completed=40,
                actual_optimizer_steps=520, selected_epoch=best_epoch, selected_selection_loss=best,
                config=config, encoder_initialization=initialization, manifest=pointer(manifest),
                identity=pointer(out / 'identity.json'), source_sha256=sources, protocol=pointer(PROTOCOL_PATH),
                actual_gpu_preflight=pointer(preflight), model_spec=model.spec(),
                evidence={n: pointer(out / (n + '.json')) for n in ('identity', 'cache_before', 'cache_after', 'history', 'sample_access', 'source_verification')},
                model_qualified=False, position_measurement_qualified=False, uncertainty_qualified=False,
                runtime_or_control_admitted=False, reserved_forward_calls=0, selected_threshold=None,
                environment=dict(torch=str(torch.__version__), numpy=np.__version__, training_device=device,
                    autocast='bfloat16', loss='float32', torch_num_threads=torch.get_num_threads()),
                forward_counts=dict(fit_batch_calls=train_calls, selection_batch_calls=selection_calls,
                    fit_rows=8320, selection_rows=8320))
            torch.save(dict(model_state_dict=best_state, metadata=metadata), out / 'selected.pt')
            status.update(training_complete=True, actual_optimizer_steps=520, epochs_completed=40,
                selected_epoch=best_epoch, selected_checkpoint=pointer(out / 'selected.pt'))
            return status
        except BaseException as error:
            status.update(actual_optimizer_steps=steps, error=type(error).__name__ + ': ' + str(error))
            raise
        finally:
            write_new(out / 'fit_status.json', status)
