"""Sharedview UNet8; two current-depth logits, no pose/completion head."""
import torch
from torch import nn

from .data import FEATURE_NAMES, CLASS_NAMES


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.SiLU())


class TinyDepthInfluenceUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.e0, self.e1, self.e2 = _block(9, 8), _block(8, 16), _block(16, 32)
        self.bottom = _block(32, 64)
        self.d2, self.d1, self.d0 = _block(96, 32), _block(48, 16), _block(24, 8)
        self.head = nn.Conv2d(8, 2, 1)

    def forward(self, features):
        if (not isinstance(features, torch.Tensor) or features.ndim != 4
                or features.shape[1:] != (9, 64, 64) or not features.is_floating_point()
                or not torch.isfinite(features).all()):
            raise ValueError('Only finite publicfeatures[B,9,64,64]; no labels/state/robot dictionary')
        a = self.e0(features)
        b = self.e1(nn.functional.max_pool2d(a, 2))
        c = self.e2(nn.functional.max_pool2d(b, 2))
        z = self.bottom(nn.functional.max_pool2d(c, 2))
        def up(x):
            return nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        z = self.d2(torch.cat((up(z), c), 1))
        z = self.d1(torch.cat((up(z), b), 1))
        return self.head(self.d0(torch.cat((up(z), a), 1)))


def binary_segmentation_loss(logits, labels):
    """[B] CE+CubeDice; allUNKNOWN views contribute zero, remain in ledger."""
    if (not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor)
            or logits.ndim != 4 or logits.shape[1] != 2
            or labels.shape != logits.shape[:1]+logits.shape[2:]
            or labels.is_floating_point() or labels.dtype == torch.bool
            or not torch.isfinite(logits).all()):
        raise ValueError('Aligned finite2class logits and integer0/1/255 labels required')
    valid = labels != 255
    target = labels.long().masked_fill(~valid, 0)
    if torch.any(valid & ((target < 0) | (target > 1))):
        raise ValueError('Only known0/rawCube1/ignore255')
    ce = nn.functional.cross_entropy(logits, target, reduction='none')
    ce = (ce*valid).sum((1, 2))/valid.sum((1, 2)).clamp_min(1)
    p, t = logits.softmax(1)[:, 1]*valid, (target == 1)*valid
    dice = 1-(2*(p*t).sum((1, 2))+1)/(p.sum((1, 2))+t.sum((1, 2))+1)
    return ce+dice


def predict_probabilities(model, features):
    model.eval()
    with torch.inference_mode():
        x = torch.as_tensor(features, dtype=torch.float32, device=next(model.parameters()).device)
        return model(x).softmax(1).cpu().numpy()


class CurrentCubeDepthRefiner:
    """SHA-bound completed binary model only; pilot never supplies measurements."""
    def __init__(self, checkpoint, expected_sha256, *, allow_unqualified_diagnostic=False):
        from pathlib import Path
        import json
        from .data import sha256, LABEL_PROTOCOL
        from .evaluation import QUALITY_PROTOCOL, THRESHOLD_GRID, choose_calibrated_threshold
        from .training import VERSION, fixed_config
        if type(allow_unqualified_diagnostic) is not bool:
            raise ValueError('Explicit diagnostic-only bool required')
        path = Path(checkpoint).resolve(strict=True)
        if sha256(path) != expected_sha256:
            raise ValueError('Exact trained checkpoint SHA required')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        meta = payload.get('metadata', {})
        if (payload.get('version') != VERSION or meta.get('dataset_split') != 'TRAIN'
                or meta.get('TEST_read') is not False or meta.get('training_complete') is not True
                or type(meta.get('epochs_completed')) is not int or meta['epochs_completed'] != 40
                or meta.get('label_protocol') != LABEL_PROTOCOL or meta.get('quality_protocol') != QUALITY_PROTOCOL
                or meta.get('completion') is not False or meta.get('current_depth_influence_only') is not True
                or meta.get('complete_visible_instance_claimed') is not False
                or meta.get('TCP_GRU_GT_input') is not False or meta.get('absolute_worldXYZ_input') is not False
                or meta.get('geometry_erosions') != 1
                or tuple(meta.get('feature_names', ())) != FEATURE_NAMES
                or tuple(meta.get('class_names', ())) != CLASS_NAMES
                or meta.get('model_config') != dict(input_channels=9, classes=2, base_channels=8)):
            raise ValueError('Completed exact TRAIN binary-depth-influence model contract required')
        status = json.loads(path.with_name('status.json').read_text())
        if (status.get('complete') is not True or status.get('training_complete') is not True
                or type(status.get('completed_epochs')) is not int or status['completed_epochs'] != 40
                or status.get('best_checkpoint_sha256') != expected_sha256
                or status.get('identity') != meta.get('identity')
                or status.get('model_qualified') is not meta.get('model_qualified')
                or status.get('scope') != meta.get('scope')):
            raise ValueError('Exact actual completed checkpoint/status identity required')
        identity = meta.get('identity')
        if not isinstance(identity, dict) or identity.get('config') != json.loads(json.dumps(fixed_config())):
            raise ValueError('Exact fixed binary training/config identity required')
        evidence = {}
        for name in ('calibration_evidence', 'oracle_evidence'):
            pointer = meta.get(name, {})
            source = (path.parent/pointer.get('path', '')).resolve(strict=True)
            if source.parent != path.parent or sha256(source) != pointer.get('sha256'):
                raise ValueError('Immutable accompanying calibration/oracle evidence required')
            evidence[name] = json.loads(source.read_text())
        selected = evidence['calibration_evidence'].get('selection', {})
        recalculated = choose_calibrated_threshold(evidence['calibration_evidence']['reports'])
        if selected != recalculated:
            raise ValueError('Saved threshold selection differs from exact conservative reports')
        self.qualified = (meta.get('model_qualified') is True and meta.get('scope') == 'full32'
                          and meta.get('pilot_is_smoke_only') is False)
        if self.qualified and (selected.get('quality_protocol') != QUALITY_PROTOCOL
                or selected.get('calibration_passed') is not True
                or selected.get('selected_threshold') != meta.get('selected_threshold')
                or meta.get('selected_threshold') not in THRESHOLD_GRID
                or evidence['oracle_evidence'].get('ready_for_segmentation_fit') is not True):
            raise ValueError('Qualified model must retain exact passing oracle and new conservative calibration')
        if not self.qualified and not allow_unqualified_diagnostic:
            raise ValueError('Pilot/unqualified model cannot provide a runtime current-depth refiner')
        self.model = TinyDepthInfluenceUNet()
        self.model.load_state_dict(payload['model_state_dict'], strict=True)
        self.model.eval()
        self.threshold, self.metadata = meta.get('selected_threshold'), meta
        self.checkpoint_sha256 = expected_sha256

    def predict(self, inputs):
        from .data import make_near_goal_crops
        from .evaluation import probability_to_observation
        crops = make_near_goal_crops(inputs)
        p = predict_probabilities(self.model, crops['features'])
        if not self.qualified:
            return dict(refiner_available=False, centre=None, probabilities=p,
                reason='unqualified_diagnostic_only_no_threshold_fallback', completion=False,
                checkpoint_sha256=self.checkpoint_sha256, Native_control_qualified=False)
        result = probability_to_observation(inputs, crops, p, self.threshold)
        result.update(refiner_available=result['measurement']['refiner_available'],
                      centre=result['measurement']['centre'], checkpoint_sha256=self.checkpoint_sha256)
        return result
