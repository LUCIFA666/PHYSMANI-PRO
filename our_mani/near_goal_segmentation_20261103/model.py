"""Small shared-view UNet8; only current9-channel public crop tensors."""
import torch
from torch import nn
from .data import FEATURE_NAMES,CLASS_NAMES


def block(cin,cout):
    return nn.Sequential(nn.Conv2d(cin,cout,3,padding=1),nn.GroupNorm(4,cout),nn.SiLU(),
                         nn.Conv2d(cout,cout,3,padding=1),nn.GroupNorm(4,cout),nn.SiLU())


class TinyNearGoalUNet(nn.Module):
    def __init__(self):
        super().__init__();self.e0=block(9,8);self.e1=block(8,16);self.e2=block(16,32);self.bottom=block(32,64)
        self.d2=block(96,32);self.d1=block(48,16);self.d0=block(24,8);self.head=nn.Conv2d(8,3,1)
    def forward(self,features):
        if not isinstance(features,torch.Tensor) or features.ndim!=4 or features.shape[1:]!=(9,64,64) or not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError('finite public features[B,9,64,64] only; no labels/robot/state dictionaries')
        a=self.e0(features);b=self.e1(nn.functional.max_pool2d(a,2));c=self.e2(nn.functional.max_pool2d(b,2));z=self.bottom(nn.functional.max_pool2d(c,2))
        up=lambda v:nn.functional.interpolate(v,scale_factor=2,mode='bilinear',align_corners=False)
        z=self.d2(torch.cat((up(z),c),1));z=self.d1(torch.cat((up(z),b),1));z=self.d0(torch.cat((up(z),a),1))
        return self.head(z)


def segmentation_loss(logits,labels):
    """CE+foregroundDice per view, then mean; parent/shot weighting is outside."""
    if logits.ndim!=4 or logits.shape[1]!=3 or labels.shape!=logits.shape[:1]+logits.shape[2:]:raise ValueError('aligned3class logits/labels required')
    valid=labels!=255;target=labels.long().masked_fill(~valid,0)
    if torch.any(valid&((target<0)|(target>2))):raise ValueError('only3visible roles or ignore255')
    ce=nn.functional.cross_entropy(logits,target,reduction='none')
    ce=(ce*valid).sum((1,2))/valid.sum((1,2)).clamp_min(1)
    probs=logits.softmax(1);dice=[]
    for k in (1,2):
        p=probs[:,k]*valid;t=(target==k)*valid
        dice.append(1-(2*(p*t).sum((1,2))+1)/(p.sum((1,2))+t.sum((1,2))+1))
    return ce+torch.stack(dice).mean(0)


def predict_probabilities(model,features):
    """Shared weights applied to four current views; no cross-shot history."""
    model.eval()
    with torch.inference_mode():
        x=torch.as_tensor(features,dtype=torch.float32,device=next(model.parameters()).device)
        return model(x).softmax(1).cpu().numpy()


class CurrentNearGoalSegmenter:
    """Explicit trained-checkpoint inference; no control or attachment assertion."""
    def __init__(self,checkpoint,expected_sha256,*,allow_unqualified_diagnostic=False):
        from .data import sha256
        from pathlib import Path
        import json
        if type(allow_unqualified_diagnostic) is not bool:raise ValueError('explicit diagnostic flag required')
        if sha256(checkpoint)!=expected_sha256:raise ValueError('segmentation checkpoint SHA mismatch')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True);meta=payload.get('metadata',{})
        if (payload.get('version')!='near_goal_current_visible_segmentation_20261103_v1'
            or meta.get('training_complete') is not True or meta.get('epochs_completed')!=40
            or meta.get('quality_protocol')!='near_goal_calibration_with_oracle_eligible_coverage_v2'
            or meta.get('dataset_split')!='TRAIN' or meta.get('TEST_read') is not False
            or meta.get('current_visible_pixels_only') is not True or meta.get('completion') is not False
            or meta.get('TCP_GRU_GT_input') is not False or meta.get('absolute_worldXYZ_input') is not False
            or tuple(meta.get('feature_names',()))!=FEATURE_NAMES or tuple(meta.get('class_names',()))!=CLASS_NAMES):
            raise ValueError('completed exact TRAIN-only currentvisible model contract required')
        self.qualified=meta.get('model_qualified') is True and meta.get('scope')=='full32'
        if not self.qualified and not allow_unqualified_diagnostic:raise ValueError('unqualified model cannot provide a runtime refiner')
        status_path=Path(checkpoint).with_name('status.json')
        if not status_path.is_file():raise ValueError('completed training status must accompany checkpoint')
        status=json.loads(status_path.read_text())
        if (status.get('complete') is not True or status.get('training_complete') is not True or status.get('completed_epochs')!=40
            or status.get('best_checkpoint_sha256')!=expected_sha256 or status.get('identity')!=meta.get('identity')
            or status.get('model_qualified') is not meta.get('model_qualified')):
            raise ValueError('checkpoint/status exact completed identity required')
        self.model=TinyNearGoalUNet();self.model.load_state_dict(payload['model_state_dict'],strict=True);self.model.eval()
        self.threshold=meta.get('selected_threshold');self.metadata=meta
        if self.qualified and self.threshold not in (.5,.7,.9,.95,.99):raise ValueError('qualified model requires the selected fixed-grid threshold')
        self.checkpoint_sha256=expected_sha256
    def predict(self,inputs):
        from .data import make_near_goal_crops
        from .evaluation import probability_to_observation
        crops=make_near_goal_crops(inputs);probabilities=predict_probabilities(self.model,crops['features'])
        if not self.qualified:
            return dict(refiner_available=False,centre=None,semantic_roles=None,probabilities=probabilities,
                reason='unqualified_diagnostic_only_no_threshold_fallback',current_visible_pixels_only=True,
                checkpoint_sha256=self.checkpoint_sha256,Native_control_qualified=False)
        observation=probability_to_observation(inputs,crops,probabilities,self.threshold)
        observation.update(checkpoint_sha256=self.checkpoint_sha256,Native_control_qualified=False,TEST_used=False)
        return observation
