"""Small TRAIN-only head fine-tune, reusing frozen public RGBD feature caches.

Pretraining exposed these TRAIN parents already; the new Native validation/test
are independent generated resets. No old validation cache/labels are decoded.
"""
from dataclasses import asdict
import time
import numpy as np
import torch
from torch import nn

from .common import ROOT, OUT, PLAN, read, write, append, pointer, sha, sources, verify, stopped
from ..trained_world_reference_live_20261104_v1.runtime import WORLD, WORLD_SHA256, CACHE, E2_SHA256
from ..world_tcp_command_warmstart_20261104_v3.evaluation import load_selected
from ..world_tcp_command_warmstart_20261104_v3.cache import selected_targets, targets_tensor, INPUT_KEYS
from ..world_tcp_command_warmstart_20261104_v3.model import command_imitation_loss
from ..world_tcp_command_warmstart_20261104_v2.model import quaternion_rotation
from ..local_mpc_p4_20261103.model import rotation_6d_to_matrix

HEADS = ('position_head', 'rotation_head', 'opening_head', 'demo_likelihood_head')


class HeadDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.heads = nn.ModuleDict({k: getattr(model, k) for k in HEADS})

    def forward(self, hidden, robot):
        p = robot[:, None, 7:10] + self.heads['position_head'](hidden)
        R = quaternion_rotation(robot[:, 10:14])[:, None] @ rotation_6d_to_matrix(self.heads['rotation_head'](hidden))
        op = self.heads['opening_head'](hidden).squeeze(-1)
        return dict(world_tcp_positions=p, world_tcp_rotations=R, opening_logits=op,
                    opening_intent=op.sigmoid(), proposal_logits=self.heads['demo_likelihood_head'](hidden).squeeze(-1))


def fit():
    from ..current_cube_rgbd_centre_large_20261104_v2.training import exclusive_resources
    plan = read(PLAN)
    cfg = plan['training']
    out = OUT / 'head_fit'
    out.mkdir(parents=True, exist_ok=False)
    frozen = sources()
    write(out / 'sources.json', frozen)
    inputs = {str(PLAN): sha(PLAN), str(WORLD): WORLD_SHA256}
    public_path = plan['cached_public_manifest']['path']
    inputs[public_path] = plan['cached_public_manifest']['sha256']
    public = read(public_path)
    subset_path = public['subset']['path']
    subset = read(subset_path)
    index_path = subset['full_index']['path']
    index = read(index_path)
    inputs.update({subset_path: public['subset']['sha256'], index_path: subset['full_index']['sha256'],
                   index['targets']['path']: index['targets']['sha256']})
    for row in plan['cached_TRAIN']:
        inputs[row['public_chunk']['path']] = row['public_chunk']['sha256']
    write(out / 'request.json', dict(plan=pointer(PLAN), inputs_sha256=inputs,
                                     configuration=cfg, no_new_VAL_or_TEST_payload_access=True,
                                     cached_TRAIN_parents_pretrained_exposed=True,
                                     labels_are_expert_intent_not_actual_contact=True))
    steps = 0
    started = time.perf_counter()
    with exclusive_resources() as resource_receipt:
        stopped()
        verify(inputs)
        verify(frozen)
        torch.set_num_threads(2)
        torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed_all(cfg['seed'])
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        model, _ = load_selected(WORLD, WORLD_SHA256, CACHE, device='cuda')
        role_data = {'fit': [], 'selection': []}
        target_indices = []
        identities = []
        captured = []
        handle = model.output_norm.register_forward_hook(lambda module, args, value: captured.append(value.detach()))
        try:
            for parent_index, parent in enumerate(plan['cached_TRAIN']):
                stopped()
                if parent['source_split'] != 'train':
                    raise ValueError('No old validation parent may enter new fit/selection')
                chunk = torch.load(parent['public_chunk']['path'], map_location='cpu', weights_only=True)
                parent_hidden = []
                for first in range(0, 16, 2):
                    batch = {k: v[first:first+2].to(device='cuda', dtype=torch.float32 if v.is_floating_point() else v.dtype)
                             for k, v in chunk['inputs'].items() if k in INPUT_KEYS}
                    captured.clear()
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        prediction = model(**batch)
                    if len(captured) != 1:
                        raise RuntimeError('Exactly one real frozen backbone feature pass required')
                    parent_hidden.append(captured[0].float().cpu().numpy().copy())
                    del prediction, batch
                hidden = np.concatenate(parent_hidden)
                offset = len(identities)
                for identity in chunk['identities']:
                    if identity['source_parent_group'] != parent['source_parent_group']:
                        raise ValueError('Cached identity differs from fixed TRAIN parent')
                    identities.append(identity)
                    target_indices.append(index['records'][identity['global_record_index']]['target_index'])
                role_data[parent['role']].append(dict(hidden=hidden, robot=chunk['inputs']['current_robot'].float().numpy().copy(),
                                                      rows=list(range(offset, offset+16)), parent_index=parent_index))
                print(f'public backbone freeze {parent_index+1}/56 {parent["task"]}', flush=True)
        finally:
            handle.remove()
        # Freeze the complete selected public feature set before decoding labels.
        feature_arrays = {}
        for role, parents in role_data.items():
            for key in ('hidden', 'robot'):
                feature_arrays[role + '_' + key] = np.concatenate([p[key] for p in parents])
        np.savez_compressed(out / 'public_backbone_features.npz', **feature_arrays)
        write(out / 'public_feature_freeze.json', dict(complete=True, features=pointer(out / 'public_backbone_features.npz'),
                                                       identities=identities, command_labels_decoded=0,
                                                       TRAIN_only=True, actual_backbone_forward_calls=56*8))
        selected = selected_targets(index['targets']['path'], target_indices)
        targets = targets_tensor([selected[i] for i in target_indices])
        data = {}
        for role, parents in role_data.items():
            rows = np.concatenate([p['rows'] for p in parents])
            data[role] = dict(hidden=torch.from_numpy(feature_arrays[role + '_hidden']).cuda(),
                              robot=torch.from_numpy(feature_arrays[role + '_robot']).cuda(),
                              target={k: v[rows].cuda() for k, v in targets.items()})
        write(out / 'dataset_report.json', dict(fit_parents=42, selection_parents=14, fit_windows=672,
                                               selection_windows=224, target_labels_decoded=len(target_indices),
                                               old_VAL_target_labels_decoded=0, old_VAL_pixels_decoded=0,
                                               new_VAL_TEST_payloads_read=0, Native_used=False,
                                               parent_roles=[{k:p[k] for k in ('role', 'task', 'source_parent_group')}
                                                             for p in plan['cached_TRAIN']]))
        model.requires_grad_(False)
        decoder = HeadDecoder(model).cuda()
        decoder.requires_grad_(True)
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        def loss_for(role, rows=None):
            values = data[role]
            if rows is None:
                rows = torch.arange(len(values['hidden']), device='cuda')
            return command_imitation_loss(decoder(values['hidden'][rows], values['robot'][rows]),
                                          **{k: v[rows] for k, v in values['target'].items()})['loss']
        with torch.no_grad():
            initial_loss = float(loss_for('selection'))
        write(out / 'initial_selection_report.json', dict(loss=initial_loss, TRAIN_selection_only=True))
        rng = np.random.default_rng(cfg['seed'])
        groups = np.arange(672).reshape(42, 16)
        best_loss, best_state, best_epoch = float('inf'), None, None
        for epoch in range(1, cfg['epochs'] + 1):
            stopped()
            losses = []
            for _ in range(cfg['steps_per_epoch']):
                selected_rows = np.concatenate([rng.choice(g, cfg['windows_per_parent_per_step'], replace=True) for g in groups])
                rows = torch.as_tensor(selected_rows, device='cuda')
                optimizer.zero_grad(set_to_none=True)
                loss = loss_for('fit', rows)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite head-training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                steps += 1
            if epoch == 1 or epoch % 5 == 0:
                with torch.no_grad():
                    selected_loss = float(loss_for('selection'))
                if selected_loss < best_loss:
                    best_loss, best_epoch = selected_loss, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
                row = dict(epoch=epoch, optimizer_steps=steps, fit_loss=float(np.mean(losses)), selection_loss=selected_loss)
                append(out / 'history.jsonl', row)
                print(row, flush=True)
        decoder.load_state_dict(best_state, strict=True)
        verify(inputs)
        verify(frozen)
        metadata = dict(schema='multitask_world_command_head_finetune_20261112_v1',
                        encoder_sha256=E2_SHA256, baseline_world_sha256=WORLD_SHA256,
                        training_updates=steps, epochs_completed=cfg['epochs'], selected_epoch=best_epoch,
                        selected_epoch_indexing='one_based', selected_selection_loss=best_loss,
                        source_sha256=frozen, model_spec=model.spec(), model_config=asdict(model.config),
                        TRAIN_only=True, VAL_read=False, TEST_read=False, new_generated_evaluation=True,
                        targets_are_future_expert_intent=True, actual_contact_time_supervised=False,
                        initial_selection_loss=initial_loss, trained_parameters=sum(p.numel() for p in decoder.parameters()),
                        fit_parents=42, selection_parents=14, parent_pretraining_exposure_disclosed=True,
                        feature_extraction_device='cuda_BF16', head_training_precision='FP32',
                        resources=resource_receipt)
        torch.save(dict(metadata=metadata, model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}), out/'selected.pt')
        write(out / 'fit_status.json', dict(complete=True, actual_optimizer_steps=steps, selected_epoch=best_epoch,
                                             initial_selection_loss=initial_loss, selected_selection_loss=best_loss,
                                             checkpoint=pointer(out/'selected.pt'), trained_parameters=metadata['trained_parameters'],
                                             wall_seconds=time.perf_counter()-started, GPU=torch.cuda.get_device_name(0),
                                             Native_GPU_fitting_overlap=False, VAL_read=False, TEST_read=False,
                                             source_unchanged=True, input_unchanged=True))


if __name__ == '__main__':
    fit()
