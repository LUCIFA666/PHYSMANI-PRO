"""Released PhysMani networks with r05 sensors, shared map and counted actuation.

The policy and online world-model algorithms are unchanged. Camera calibration
comes from the common sensors; dataset expert/future frames are never loaded.
"""
from collections import deque
from pathlib import Path
import time
import numpy as np
import torch

from ..contracts import Pose
from .simulation import physics_time


CAMERAS = ('front','8','16','36')


def tensors(observation,robot,device='cuda'):
    rgb,pcd,depth=[],[],[]
    for name in CAMERAS:
        image=np.asarray(observation['observation/images'][name])
        d=np.asarray(observation['observation/depths'][name])
        calibration=observation['observation/calibrations'][name]
        points=calibration.backproject(d,np.ones(d.shape,dtype=bool)).reshape(*d.shape,3)
        rgb.append(image.transpose(2,0,1).astype(np.float32)/127.5-1.)
        pcd.append(points.transpose(2,0,1).astype(np.float32))
        depth.append(d[None].astype(np.float32))
    gripper=np.r_[robot.pose.xyzw(),float(robot.gripper_open)].astype(np.float32)
    return tuple(torch.as_tensor(np.asarray(value)[None],device=device)
                 for value in (rgb,pcd,depth,gripper))


def build_policy(root,config,task):
    from physmani_policy.models.foresight_diffuser_actor_v6 import ForesightDiffuserActorV6
    from physmani_bench.evaluation_utils import get_gripper_loc_bounds
    # Keep any newly downloaded backbone asset inside our-mani.
    import clip
    clip_cache=Path(root)/'runtime/framework/clip';clip_cache.mkdir(parents=True,exist_ok=True)
    cached=Path.home()/'.cache/clip/RN50.pt'
    destination=clip_cache/'RN50.pt'
    if cached.is_file() and not destination.exists():
        destination.symlink_to(cached)
    original_load=clip.load
    def cached_load(*args,**kwargs):
        kwargs.setdefault('download_root',str(clip_cache))
        return original_load(*args,**kwargs)
    clip.load=cached_load
    bounds_file=Path(config['baseline_repo'])/'tasks/18_peract_tasks_location_bounds.json'
    bounds=get_gripper_loc_bounds(str(bounds_file),task=task,buffer=.08)
    try:
        model=ForesightDiffuserActorV6(backbone='clip',image_size=(256,256),embedding_dim=120,
            num_vis_ins_attn_layers=2,use_instruction=True,fps_subsampling_factor=5,
            gripper_loc_bounds=bounds,rotation_parametrization='6D',quaternion_format='xyzw',
            diffusion_timesteps=100,denoise_model='rectified_flow',num_inference_steps=10,
            nhist=3,relative=False,lang_enhanced=False,prob_dropout_vel_features=0.,
            bool_use_gating_and_adapter=False).cuda()
    finally:
        clip.load=original_load
    checkpoint=torch.load(config['baseline_checkpoint'],map_location='cpu',weights_only=False)
    weights={key.removeprefix('module.'):value for key,value in checkpoint['weight'].items()}
    model.load_state_dict(weights,strict=True)
    return model.eval()


class PhysManiAdapter:
    def __init__(self,root,config,task,model):
        from physmani_bench.evaluation_utils import load_instructions
        from physmani_policy.actioner import Actioner
        repo=Path(config['baseline_repo'])
        instructions=load_instructions(repo/'instructions/rmt/rmt_instructions_v5_rldyna19task_genvel_withstr.pkl',tasks=[task])
        self.actioner=Actioner(model,instructions,apply_cameras=CAMERAS,action_dim=8)
        self.config=config
        self.history=deque(maxlen=3)
        self.world=None
        self.decisions=0
        self.compute_seconds=0.

    def reset(self,env,task,variation,observation,robot):
        from physmani_wm.world_model import WorldModel
        from physmani_bench.env import convert_cam_param_to_gs_format
        from physmani_bench.logger import BenchmarkLogger
        class SensorWorldModel(WorldModel):
            def get_cam_views(self,episode_dir,cam_names):
                views=[]
                for name in cam_names:
                    camera=env.cameras[name]
                    matrix=camera.get_matrix()
                    if matrix.shape==(3,4):
                        matrix=np.vstack([matrix,[0.,0.,0.,1.]])
                    params=dict(extrinsics=matrix,intrinsics=camera.get_intrinsic_matrix(),
                                near=camera.get_near_clipping_plane(),far=camera.get_far_clipping_plane())
                    view=convert_cam_param_to_gs_format(params,device=self.device)
                    view['cam_name']=name;views.append(view)
                return dict(views=views,viewmats=torch.stack([v['world_view_transform'].T for v in views]),
                            Ks=torch.stack([v['K'] for v in views]),image_height=256,image_width=256)
        self.actioner.load_episode(task,variation)
        self.history.clear();self.decisions=0
        self.world=SensorWorldModel(logger=BenchmarkLogger(log_dir=Path(self.config['baseline_output']),bool_enable=False),
            device=torch.device('cuda'),episode_dir=Path('.'),cam_names=list(CAMERAS),
            num_init_gaussian=25000,max_iter_cano_gaussian=3500,max_iter_gaussian=7,
            max_iter_network=50,num_future_frames=10,bool_mask=False,bool_wm_predict_velmappc=True)
        rgb,pcd,depth,gripper=tensors(observation,robot)
        clock=physics_time();start=time.perf_counter()
        masks=torch.as_tensor(np.stack([observation['observation/instance_masks'][name].transpose(2,0,1)
                                       for name in CAMERAS])[None],device='cuda')
        self.world.init(rgb,pcd,depth,gripper=gripper,mask=masks)
        torch.cuda.synchronize();self.compute_seconds+=time.perf_counter()-start
        self.world.last_t=self.world.cur_t=0.
        if abs(physics_time()-clock)>1e-6:
            raise RuntimeError('world-model initialization advanced physics')

    def observe(self,observation,robot):
        rgb,pcd,depth,gripper=tensors(observation,robot)
        start=time.perf_counter()
        self.world.update(rgb=rgb,pcd=pcd,depth=depth,mask=None,gripper=gripper[0].clone(),
                          last_t=self.world.last_t,cur_t=robot.timestamp)
        torch.cuda.synchronize();self.compute_seconds+=time.perf_counter()-start
        self.world.last_t=self.world.cur_t=robot.timestamp

    def predict(self,observation,robot):
        from physmani_bench.runner import update_actioner_kwargs_init_frames,update_actioner_kwargs_subsequent_frames
        rgb,pcd,depth,gripper=tensors(observation,robot)
        self.history.append(gripper)
        history=list(self.history)
        while len(history)<3:history.insert(0,history[0])
        grippers=torch.stack(history,dim=1)
        start=time.perf_counter()
        if self.decisions==0:
            kwargs=update_actioner_kwargs_init_frames(rgb.cpu(),pcd.cpu(),gripper.cpu(),1,10,True)
        else:
            prediction=self.world.predict(num_future_frames=10,dt=.05)
            kwargs=update_actioner_kwargs_subsequent_frames(prediction,self.world,10,.05,1,True)
        kwargs={key:value.cuda() if isinstance(value,torch.Tensor) else value for key,value in kwargs.items()}
        pcd=pcd.clone();pcd[:,:,0:2].clamp_(-2.5,2.5);pcd[:,:,2].clamp_(0.,2.)
        result=self.actioner.predict(rgb[:,None],pcd[:,None],grippers,interpolation_length=2,
                                    step_id=self.decisions,**kwargs)
        torch.cuda.synchronize();self.compute_seconds+=time.perf_counter()-start
        self.decisions+=1
        action=result['trajectory'][-1].detach().cpu().numpy().reshape(-1,8)[0]
        if not np.isfinite(action).all():raise ValueError('nonfinite PhysMani action')
        return action

    def execute(self,env,action,*,observer=None):
        """RRTConnect retained; shared map and actuator replace hidden physics steps."""
        from pyrep.const import ConfigurationPathAlgorithms as Algos
        from pyrep.errors import ConfigurationPathError, IKError
        clock=physics_time();path=None;error=None
        try:
            path=env.map_service.plan_path(action[:3].tolist(),quaternion=action[3:7].tolist(),
                ignore_collisions=False,trials=100,max_configs=10,max_time_ms=10,
                trials_per_goal=5,algorithm=Algos.RRTConnect)
        except (ConfigurationPathError,IKError) as exc:
            error=type(exc).__name__
        if abs(physics_time()-clock)>1e-6:raise RuntimeError('RRT advanced physics')
        # No physics-only trajectory extraction. RML .step() only produces a joint target.
        done=False
        for tick in range(self.config['baseline_decision_ticks']):
            if env.ledger.remaining<env.ledger.dt-1e-8 or env._native_success_latched:break
            joints=env.arm.get_joint_positions()
            if path is not None and not done:
                done=path.step()
                joints=path._joint_position_action
                if joints is None:joints=env.arm.get_joint_positions()
            opening=float(action[7]>=.5) if tick==self.config['baseline_decision_ticks']-1 else env._requested_opening
            env.command_joints(joints,opening=opening,phase='physmani_path')
            observation,robot=env.observe()
            if observer:observer(observation,robot)
            if env._native_success_latched:break
            self.observe(observation,robot)
        return dict(path_found=path is not None,error=error)
