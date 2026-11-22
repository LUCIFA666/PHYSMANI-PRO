"""Scene-only task e16, separate from the frozen GRU motion context c16.

This classifier distinguishes two benchmark scene families. Scene appearance
cannot recover an unobserved instruction among multiple goals in one scene.
No task ID, robot state, camera ID, calibration or action phase enters forward.
"""
import torch
from torch import nn


TASKS = ('pick_moving_target_on_the_table', 'insert_onto_rotating_peg')


class SceneTaskEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Exactly the old visual submodule's parameter shapes; optional camera
        # warmstart never imports its robot GRU, eleven-way head or E table.
        self.camera_encoder = nn.Sequential(
            nn.Conv2d(4, 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.scene_projection = nn.Sequential(nn.Linear(128, 128), nn.SiLU(),
                                               nn.LayerNorm(128), nn.Linear(128, 16))
        self.classifier = nn.Linear(16, 2)

    def forward(self, rgbd):
        """rgbd[B,V,4,64,64], 1<=V<=4; full scene normally uses four views."""
        if (not isinstance(rgbd, torch.Tensor) or not rgbd.is_floating_point()
                or rgbd.ndim != 5 or not 1 <= rgbd.shape[1] <= 4
                or tuple(rgbd.shape[2:]) != (4, 64, 64) or rgbd.shape[0] < 1):
            raise ValueError('finite floating scene-only RGBD[B,V,4,64,64],1<=V<=4 required')
        if not bool(torch.isfinite(rgbd).all()):
            raise ValueError('nonfinite scene pixels are forbidden')
        batch, views = rgbd.shape[:2]
        features = self.camera_encoder(rgbd.reshape(batch * views, 4, 64, 64)).reshape(batch, views, 64)
        # Permutation invariant; the camera index has no learned embedding or
        # ordered flattening. View-specific appearance can still be a shortcut.
        pooled = torch.cat((features.mean(1), features.amax(1)), -1)
        task_e = self.scene_projection(pooled).reshape(batch, 2, 8).softmax(-1).reshape(batch, 16)
        logits = self.classifier(task_e)
        return dict(task_e=task_e, task_logits=logits, task_probabilities=logits.softmax(-1))

    def warmstart_camera_encoder(self, state_dict, prefix='encoder.camera_encoder.'):
        """Explicit state-dict-only initialization of the shared CNN, strictly."""
        own = self.camera_encoder.state_dict()
        selected = {key: state_dict[prefix + key] for key in own if prefix + key in state_dict}
        if set(selected) != set(own):
            raise ValueError('complete matching visual CNN weights required; no whole-encoder replacement')
        if any(selected[key].shape != own[key].shape or not torch.isfinite(selected[key]).all()
               for key in own):
            raise ValueError('matching finite visual parameter shapes required')
        self.camera_encoder.load_state_dict(selected, strict=True)
        return dict(prefix=prefix, parameter_keys=sorted(selected), robot_or_classifier_weights_loaded=False)

    def spec(self):
        return dict(schema='scene_only_semantic_e16_SimNorm_20261103_v1',
                    tasks=list(TASKS), input='rgbd[B,V,4,64,64],1<=V<=4',
                    default_views=4, camera_pool='shared CNN then mean+max set pooling',
                    task_e_shape=[16], simnorm_groups=2, simnorm_group_size=8,
                    output_logits=2, task_ID_input=False, robot_input=False,
                    camera_ID_or_calibration_input=False, action_phase_supervision=False,
                    motion_context_c_separate=True, scene_intention_ambiguity_resolved=False,
                    frozen_uniform_e_P4_direct_substitution_allowed=False,
                    caveat='benchmark scenes use different camera calibration; set pooling does not prove semantic invariance')
