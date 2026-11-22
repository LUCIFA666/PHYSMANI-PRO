"""A real FASTER LoRA configuration; import in its isolated Python >=3.11 env."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]


class OurManiTransforms:
    def __call__(self, model_config):
        from openpi import transforms
        from openpi.policies.our_mani_policy import OurManiInputs, OurManiOutputs
        return transforms.Group(inputs=[OurManiInputs()], outputs=[OurManiOutputs()])


def make_config(repo_id, initial_params, *, exp_name="physmani_grasp2", batch_size=2,
                updates=30000, resume=False):
    from openpi.models.pi0_config import Pi0FasterConfig
    from openpi.training.config import TrainConfig, SimpleDataConfig, DataConfig, AssetsConfig
    from openpi.training.weight_loaders import CheckpointWeightLoader
    from openpi.training.optimizer import CosineDecaySchedule, AdamW
    from flax import nnx
    from openpi.shared import nnx_utils
    model = Pi0FasterConfig(pi05=True, action_dim=32, action_horizon=16,
                           max_token_len=384, discrete_state_input=True,
                           paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
                           max_delay=10, mix_prob=.5, alpha=.6, u0=.9,
                           active_image_keys=('base_0_rgb',))
    asset_id = "physmani_panda_"+hashlib.sha256(repo_id.encode()).hexdigest()[:8]
    return TrainConfig(name="pi05_faster_our_mani_lora", exp_name=exp_name,
                       project_name="our-mani", model=model,
                       data=SimpleDataConfig(repo_id=repo_id, assets=AssetsConfig(asset_id=asset_id),
                                             base_config=DataConfig(prompt_from_task=True),
                                             data_transforms=OurManiTransforms()),
                       weight_loader=CheckpointWeightLoader(initial_params),
                       freeze_filter=nnx.Any(model.get_freeze_filter(), nnx_utils.PathRegex('.*img.*')), ema_decay=None,
                       batch_size=batch_size, num_train_steps=updates, num_workers=2,
                       lr_schedule=CosineDecaySchedule(warmup_steps=min(500, max(1, updates//10)), peak_lr=5e-5,
                                                       decay_steps=updates, decay_lr=5e-6),
                       optimizer=AdamW(clip_gradient_norm=1.),
                       assets_base_dir=str(ROOT/'artifacts/framework/faster_assets'),
                       checkpoint_base_dir=str(ROOT/'artifacts/framework/faster_checkpoints'),
                       save_interval=min(1000, max(1, updates//5)), log_interval=min(100, max(1, updates//100)),
                       keep_period=5000, wandb_enabled=False,
                       resume=resume, seed=0,
                       policy_metadata={"our_mani_conditioning": "our_mani_region_prompt_v1",
                                        "action_space": "world_ee_xyz_xyzw_gripper_open",
                                        "camera_encoding": "four_source_views_2x2_mosaic",
                                        "vision_encoder_frozen": True,
                                        "active_model_image_keys": ["base_0_rgb"],
                                        "max_delay": 10, "alpha": .6, "u0": .9})
