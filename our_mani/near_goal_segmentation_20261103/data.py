"""Physical public-input/privileged-label separation and one runtime crop.

Every zero-tick shot is an independent scene. Shot indices are never converted
into a trajectory clock. Only the seven declared public NPZ keys enter crops.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from torch.utils.data import Dataset

SCHEMA='near_goal_current_rgbd_capture_train_v1'
CAMERAS=('front','8','16','36')
PUBLIC_KEYS=frozenset(('rgb','depth_m','K','T','public_sphere_center','camera_names','timestamp'))
FEATURE_NAMES=('red_centered','green_centered','blue_centered','log1p_depth_m',
 'sphere_relative_camera_x_div_0p1m','sphere_relative_camera_y_div_0p1m','sphere_relative_camera_z_div_0p1m',
 'valid_metric_depth','inside_original_image')
CLASS_NAMES=('background','cube','goal');CROP=64;IGNORE=255
ROLE_MAP={'fit':'fit','selection':'selection','calibration':'calib','calib':'calib'}
PARENT_PREFIX='near_goal_perception_train_seed'
PARTITION_IDS=tuple(PARENT_PREFIX+str(seed) for seed in range(261103600,261103632))
PILOT_IDS=PARTITION_IDS[:6]+(PARTITION_IDS[24],PARTITION_IDS[28])
SHOT_IDS=tuple(f'shot{i:02d}' for i in range(8))
ROLE_COUNTS={'pilot_smoke':{'fit':6,'selection':1,'calib':1},'full32':{'fit':24,'selection':4,'calib':4}}


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def calibration_digest(inputs):
    value={k:np.asarray(inputs[k]).tolist() for k in ('K','T','camera_names')}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def load_npz(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}

def validate_public(inputs):
    if not isinstance(inputs,dict) or set(inputs)!=PUBLIC_KEYS:raise ValueError('exact public RGBD/calibration/sphere keys only; no labels/robot/prediction')
    rgb,depth=np.asarray(inputs['rgb']),np.asarray(inputs['depth_m'])
    if rgb.dtype!=np.uint8 or rgb.shape!=(4,256,256,3) or depth.shape!=(4,256,256) or depth.dtype!=np.float32:
        raise ValueError('four original256 RGB uint8 and metricdepth float32 views required')
    K=np.asarray(inputs['K'],float);T=np.asarray(inputs['T'],float);sphere=np.asarray(inputs['public_sphere_center'],float)
    if K.shape!=(4,3,3) or T.shape!=(4,4,4) or sphere.shape!=(3,) or not all(np.isfinite(v).all() for v in (K,T,sphere)):
        raise ValueError('finite signedK/camera-to-world/publicsphere required')
    if tuple(np.asarray(inputs['camera_names']).tolist())!=CAMERAS:raise ValueError('camera ordering differs')
    now=np.asarray(inputs['timestamp'])
    if now.shape!=() or now.dtype.kind not in 'fi' or not np.isfinite(now) or float(now)<0:raise ValueError('finite scalar actual scene clock required')
    for i in range(4):
        if abs(np.linalg.det(K[i]))<1e-12 or not np.allclose(K[i,2],[0,0,1],rtol=0,atol=1e-10):raise ValueError('invertible signed pinhole K required')
        R=T[i,:3,:3]
        if not np.allclose(T[i,3],[0,0,0,1],atol=1e-10) or not np.allclose(R.T@R,np.eye(3),atol=1e-6) or abs(np.linalg.det(R)-1)>1e-6:
            raise ValueError('proper camera-to-world pose required')
    return inputs


def make_near_goal_crops(inputs):
    """[4,9,64,64]; no resize, absoluteXYZ, TCP, GRU, labels or adaptive ROI.

    Integer centre=round-half-up(project(publicsphere)); origin=centre-32.
    XYZ features are camera-point minus camera-sphere, in0.1m units.
    """
    validate_public(inputs);features=np.zeros((4,9,CROP,CROP),np.float32)
    rgb=np.zeros((4,CROP,CROP,3),np.uint8);depth=np.zeros((4,CROP,CROP),np.float32)
    cropvalid=np.zeros((4,CROP,CROP),bool);depthvalid=cropvalid.copy();origins=np.zeros((4,2),np.int64)
    projection=np.zeros(4,bool);sphere=np.asarray(inputs['public_sphere_center'],float)
    yy,xx=np.indices((CROP,CROP))
    for cam in range(4):
        K=np.asarray(inputs['K'][cam],float);T=np.asarray(inputs['T'][cam],float)
        sc=(sphere-T[:3,3])@T[:3,:3]
        if sc[2]<=0:continue
        uv=(K@(sc/sc[2]))[:2];origin=np.floor(uv+.5).astype(np.int64)-CROP//2
        origins[cam]=origin;projection[cam]=True;x=xx+origin[0];y=yy+origin[1]
        valid=(x>=0)&(x<256)&(y>=0)&(y<256);cropvalid[cam]=valid
        rgb[cam,valid]=inputs['rgb'][cam,y[valid],x[valid]]
        depth[cam,valid]=inputs['depth_m'][cam,y[valid],x[valid]]
        dv=valid&np.isfinite(depth[cam])&(depth[cam]>0);depthvalid[cam]=dv
        pix=np.stack((x,y,np.ones_like(x)),axis=-1);rays=pix@np.linalg.inv(K).T
        points=rays*np.where(dv,depth[cam],0)[...,None]
        features[cam,:3]=np.moveaxis(np.where(valid[...,None],rgb[cam]/255.-.5,0),-1,0)
        features[cam,3]=np.where(dv,np.log1p(np.where(dv,depth[cam],0)),0)
        features[cam,4:7]=np.moveaxis(np.where(dv[...,None],(points-sc)/.1,0),-1,0)
        features[cam,7]=dv;features[cam,8]=valid
    return dict(features=features,rgb=rgb,depth_m=depth,valid_depth=depthvalid,valid_crop=cropvalid,
                origins_xy=origins,projection_available=projection)


def crop_semantics(semantic,crops):
    semantic=np.asarray(semantic)
    if semantic.dtype!=np.uint8 or semantic.shape!=(4,256,256) or not set(np.unique(semantic))<={0,1,2,255}:
        raise ValueError('visible Native instance roles0bg1cube2goal255ignore required; farCube staysCube')
    labels=np.full((4,CROP,CROP),IGNORE,np.uint8);yy,xx=np.indices((CROP,CROP))
    for cam in range(4):
        x,y=xx+crops['origins_xy'][cam,0],yy+crops['origins_xy'][cam,1];valid=crops['valid_crop'][cam]
        labels[cam,valid]=semantic[cam,y[valid],x[valid]]
    return labels


def current_cube_measurement(inputs,crops,cube_mask):
    """Only measured current depth; exact old quantile/geometry confidence algebra.

    Does not call the legacy table-initialization/history/velocity machinery.
    Identity rotation is an explicit unobserved prior. No completed box cloud.
    """
    validate_public(inputs);mask=np.asarray(cube_mask)
    if mask.shape!=(4,CROP,CROP) or mask.dtype!=np.bool_:raise ValueError('explicit current visible Cube pixel mask required')
    reasons=[];cloud=[];views=[];yy,xx=np.indices((CROP,CROP))
    if np.any(mask&~crops['valid_crop']):reasons.append('cube_outside_observed_crop')
    if mask[:,0,:].any() or mask[:,-1,:].any() or mask[:,:,0].any() or mask[:,:,-1].any():reasons.append('cube_touches_crop_boundary')
    for cam in range(4):
        selected=mask[cam]&crops['valid_depth'][cam]
        if not selected.any():continue
        x=xx[selected]+crops['origins_xy'][cam,0];y=yy[selected]+crops['origins_xy'][cam,1]
        rays=np.c_[x,y,np.ones(len(x))]@np.linalg.inv(inputs['K'][cam]).T
        pc=rays*crops['depth_m'][cam,selected,None];T=inputs['T'][cam]
        cloud.append(pc@T[:3,:3].T+T[:3,3]);views.extend([cam]*len(x))
    points=np.concatenate(cloud) if cloud else np.empty((0,3));n=len(points);visible_views=len(set(views))
    confidence=min(1.,visible_views/3)*min(1.,n/60)
    if n<12:reasons.append('insufficient_current_depth_points')
    if visible_views<2:reasons.append('insufficient_current_camera_views')
    if confidence<.5:reasons.append('original_geometry_confidence_below_half')
    centre=None
    if n>=12:
        centre=(np.quantile(points,.01,axis=0)+np.quantile(points,.99,axis=0))/2
        centre[2]=np.quantile(points[:,2],.98)-.025
        if np.linalg.norm(centre-inputs['public_sphere_center'])>.08:reasons.append('outside_predeclared_near_scope_0p08m')
    available=not reasons and centre is not None
    return dict(refiner_available=available,centre=None if not available else centre.tolist(),
        diagnostic_centre_only=None if centre is None else centre.tolist(),measured_points=n,visible_views=visible_views,
        geometry_confidence=confidence,reasons=reasons,near_scope_m=.08,orientation_observed=False,
        rotation_identity_prior=True,completion_used=False,mask_is_current_visible_pixels_only=True)


class CaptureDataset(Dataset):
    """TRAIN manifest owns grouping. Labels are opened only by offline dataset."""
    def __init__(self,manifest_path,role=None):
        self.path=Path(manifest_path).resolve()
        if any('test' in part.lower() for part in self.path.parts):raise ValueError('TEST manifests are outside TRAIN loader')
        self.manifest=json.loads(self.path.read_text());m=self.manifest
        if m.get('schema')!=SCHEMA or m.get('dataset_split')!='TRAIN' or m.get('complete') is not True:
            raise ValueError('completed explicitly TRAIN capture manifest required')
        if any(m.get(k) is not True for k in ('all_native_closed','all_process_exits_observed','source_unchanged')):
            raise ValueError('actual closed/sourceunchanged capture evidence required')
        self.scope=m.get('scope');expected=ROLE_COUNTS.get(self.scope)
        if expected is None:raise ValueError('pilot_smoke/full32 scope required')
        pointer=m.get('partition',{});partition_path=(self.path.parent/pointer.get('path','')).resolve()
        if (not isinstance(pointer.get('sha256'),str) or any('test' in part.lower() for part in partition_path.parts)
            or not partition_path.is_file() or sha256(partition_path)!=pointer['sha256']):
            raise ValueError('frozen pre-capture TRAIN partition pointer/SHA required')
        partition=json.loads(partition_path.read_text());declared=partition.get('parents',[])
        if (len(declared)!=32 or {p.get('parent_id') for p in declared}!=set(PARTITION_IDS)):
            raise ValueError('entire preassigned32-parent partition required')
        partition_rows={p['parent_id']:p for p in declared}
        for i,parent_id in enumerate(PARTITION_IDS):
            row=partition_rows[parent_id];expected_role='fit' if i<24 else ('selection' if i<28 else 'calib')
            if (ROLE_MAP.get(row.get('role'))!=expected_role or tuple(row.get('shot_ids',()))!=SHOT_IDS
                or row.get('pilot') is not (parent_id in PILOT_IDS)):
                raise ValueError('preassigned parent role/pilot/shot partition changed')
        self.partition_path=partition_path;self.partition_sha256=pointer['sha256']
        parents=m.get('parents',[]);ids=[p['parent_id'] for p in parents]
        if set(ids)!=set(PILOT_IDS if self.scope=='pilot_smoke' else PARTITION_IDS):
            raise ValueError('capture must match exact preassigned parent cohort')
        for parent in parents:
            if ROLE_MAP.get(parent.get('role'))!=ROLE_MAP[partition_rows[parent['parent_id']]['role']]:
                raise ValueError('parent role differs from frozen partition')
        if any(p.get('role') not in ROLE_MAP for p in parents):raise ValueError('unknown declared parent role')
        if len(set(ids))!=len(ids) or {k:sum(ROLE_MAP[p['role']]==k for p in parents) for k in expected}!=expected or len(parents)!=sum(expected.values()):
            raise ValueError('exact preassigned parent roles required')
        self.records=[];scenes=set();paths=set()
        for parent in parents:
            anchor=parent.get('public_sphere_fit',{})
            if (anchor.get('source_kind')!='initial_public_RGBD_RedGoalMarker' or anchor.get('GT_pose_used') is not False
                or not isinstance(anchor.get('source_frame_id'),str) or not anchor['source_frame_id']
                or any(not isinstance(anchor.get(k),str) or len(anchor[k])!=64 for k in ('source_rgbd_sha256','calibration_sha256','fit_metadata_sha256'))
                or np.asarray(anchor.get('center_world'),float).shape!=(3,) or not np.isfinite(anchor['center_world']).all()):
                raise ValueError('one immutable public first-frame sphere source identity required')
            shots=parent.get('shots',[])
            if len(shots)!=8 or tuple(shot.get('shot_id') for shot in shots)!=SHOT_IDS:
                raise ValueError('all8 preassigned independent shot IDs/order retained')
            for shot in shots:
                scene=shot['scene_id']
                if scene in scenes:raise ValueError('unique independent scene identity required')
                scenes.add(scene);record=dict(shot,parent_id=parent['parent_id'],declared_role=parent['role'],role=ROLE_MAP[parent['role']],public_sphere_fit=anchor)
                for key,folder in (('inputs','inputs'),('labels','labels')):
                    pointer=shot[key];path=(self.path.parent/pointer['path']).resolve()
                    if (any('test' in part.lower() for part in path.parts) or self.path.parent not in path.parents
                        or path.parent.name!=folder or path in paths or sha256(path)!=pointer['sha256']):raise ValueError('separated immutable '+key+' path/SHA required')
                    paths.add(path);record[key+'_path']=path
                if record['inputs_path'].parent.parent!=record['labels_path'].parent.parent:
                    raise ValueError('input/label directories must be physical siblings')
                if shot.get('independent_scene') is not True or (type(shot.get('physics_ticks')) is not int or shot['physics_ticks']!=0):
                    raise ValueError('zero-tick independent scenes; no fabricated trajectory')
                self.records.append(record)
        self.all_records=self.records
        role=ROLE_MAP.get(role,role)
        self.records=[r for r in self.records if role is None or r['role']==role]
        if role is not None and role not in expected:raise ValueError('unknown TRAIN role')

    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        r=self.records[index]
        for key in ('inputs','labels'):
            if sha256(r[key+'_path'])!=r[key]['sha256']:raise ValueError('capture changed after manifest load')
        inputs=load_npz(r['inputs_path']);validate_public(inputs)
        anchor=r['public_sphere_fit']
        if (not np.array_equal(np.asarray(inputs['public_sphere_center']),np.asarray(anchor['center_world']))
            or calibration_digest(inputs)!=anchor['calibration_sha256']):
            raise ValueError('initial public sphere/calibration identity changed; no refit or GT fallback')
        if float(inputs['timestamp'])!=0.:raise ValueError('independent zero-step shot has actual clock0, not index*.05')
        labels=load_npz(r['labels_path']);semantic=labels.get('semantic_role_ids')
        crops=make_near_goal_crops(inputs);semantic=crop_semantics(semantic,crops)
        centre=np.asarray(labels.get('cube_center_world'),float);present=np.asarray(labels.get('cube_present'))
        if centre.shape!=(3,) or present.shape!=() or present.dtype!=np.bool_ or (bool(present) and not np.isfinite(centre).all()):
            raise ValueError('offline Cube centre/presence labels required; never model features')
        return dict(features=crops['features'],semantic=semantic,inputs=inputs,crops=crops,
            cube_center_world=centre,cube_present=bool(present),parent_id=r['parent_id'],scene_id=r['scene_id'],
            role=r['role'],condition=r.get('condition','unspecified'),frame_weight=1/8.)
