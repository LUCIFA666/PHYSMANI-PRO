"""Closed CF pilot40 data. Public inputs and private supervision stay separate.

GT pose/projection never creates a visibility positive. Strong negatives use
explicit label-side proofs. Constructor checks metadata (including proofGT)
and bytes only; role-specific getters cannot expose the eight reserved parents
unless a separate post-selection evaluation explicitly permits them.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='current_cube_dual_crop_CF_TRAIN_dataset_20261103_v1'
SCOPE='current_cube_dual_crop_CF_pilot40'
PLAN_SCHEMA='CURRENT_CUBE_dual_crop_CF_pilot40_plan_20261103_v1'
LABEL_PROTOCOL='current_cube_crop_depth_influence_availability_20261103_v1'
SHOT_IDS=tuple(f'shot{i:02d}'for i in range(13))
PREPARATION_PATH=ROOT/'artifacts/framework/local_mpc_20261102/current_RGBD_centre_TRAIN_data_audit_v1/dual_crop_capture_preparation_manifest_v2.json'
PREPARATION_SHA='befefa20fd2c0d26635729f38138ba4ba8b459c6c31ab3083f753f01cbf5e32c'
SHARED_REGISTRY_SHA='76e836b6a872dce703598b18a694953d4ae48be69458bfde7bbde0c214c3cb9c'
REGISTRY_SHA='c482d57a9145924471ddf4bb0b7f4887805d09e89a42d2574a85c8d22684c272'
ROLE_COUNTS={'fit':16,'selection':16,'reserved_fresh_centre_VAL':8}
FEATURES_SHA='705417d630ddb8935305dcbe0d4cadea3178b05054fb4a04a1e99efee0e72b1a'
MODEL_SHA='331f6cea2e1a5089da2dbc109a2a68651766ef83e7b33eb2dd07aaaceefcbec2'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def checked_pointer(base,ref):
    if not isinstance(ref,dict)or set(ref)!={'path','sha256'}:raise ValueError('Exact path/SHA pointer required')
    path=(Path(base)/ref['path']).resolve(strict=True)
    if sha256(path)!=ref['sha256']:raise ValueError('Source changed: '+str(path))
    return path


def pilot_metadata():
    if sha256(PREPARATION_PATH)!=PREPARATION_SHA:raise ValueError('Frozen pilot metadata changed')
    meta=read_json(PREPARATION_PATH);ids=tuple(meta['minimal_pilot_capture']['parent_ids'])
    jobs={j['parent_id']:j for j in meta['jobs']};rows=[jobs[i]for i in ids]
    if len(ids)!=40 or len(set(ids))!=40 or {k:sum(r['role']==k for r in rows)for k in ROLE_COUNTS}!=ROLE_COUNTS:
        raise ValueError('Exact predeclared40 parents/16fit/16selection/8reserved required')
    return rows


def check_frozen_definitions():
    for name,digest in (('features.py',FEATURES_SHA),('model.py',MODEL_SHA)):
        if sha256(Path(__file__).with_name(name))!=digest:raise ValueError('STOP model/features changed')


PARTITION_IDS=tuple(p['parent_id']for p in pilot_metadata())


def load_npz(path):
    with np.load(path,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}


def construction_proofs(metadata,shot_id):
    """Capture writes explicit nulls for inapplicable proof channels."""
    required='scope_negative_proof'if shot_id=='shot01'else None
    if required is not None and metadata.get(required)is None:raise ValueError('Required construction proof cannot be null')
    return {key:metadata[key]for key in ('absence_proof','scope_negative_proof')if metadata.get(key)is not None}


def validated_target_metadata(case,metadata,shot_id):
    """Labels-only helper also usable for Root's first-fit-parent13shot QA."""
    case=Path(case).resolve();absence_verified=False;bound={}
    for key,ref in construction_proofs(metadata,shot_id).items():
        path=checked_pointer(case,ref);proof=read_json(path);bound[str(path)]=ref['sha256']
        if path.parent!=case/'labels'or type(proof.get('verified'))is not bool:raise ValueError('Original explicit label-side proof status required')
        if key=='absence_proof':absence_verified=proof['verified']and proof.get('all_four_camera_frusta_outside')is True
        if key=='scope_negative_proof'and (proof.get('verified')is not True or proof.get('construction')!='outside_scope'or proof.get('intended_scope_m')!=.20 or not proof.get('distance_from_initial_public_sphere_m',0)>.20):raise ValueError('Actual outside-scope readback proof required')
    return dict(metadata,_verified_absence_for_supervision=absence_verified),bound


def derive_targets(inputs,labels,metadata,shot_id):
    """Labels-only classification; no model prediction enters target choice."""
    from .features import make_dual_goal_crops
    from ..near_goal_depth_refiner_20261103.data import binary_depth_labels
    crops=make_dual_goal_crops(inputs)
    owner=binary_depth_labels(inputs,labels)==1
    without_rgb=np.asarray(labels['without_cube_rgb']);without_depth=np.asarray(labels['without_cube_depth_m'])
    if without_rgb.shape!=(4,256,256,3)or without_rgb.dtype!=np.uint8 or without_depth.shape!=(4,256,256)or without_depth.dtype!=np.float32:
        raise ValueError('Actual numeric withoutCube normalRGBD labels required')
    support=[];response=[]
    for scale in ('near','wide'):
        c=crops[scale];yy,xx=np.indices((64,64));count=0;changed=0
        for camera in range(4):
            if scale=='near':u=c['origins_xy'][camera,0]+xx;v=c['origins_xy'][camera,1]+yy
            else:u=c['source_pixels_uv'][camera,:,:,0];v=c['source_pixels_uv'][camera,:,:,1]
            valid=c['valid_crop'][camera];u=u[valid];v=v[valid]
            count+=int(owner[camera,v,u].sum())
            changed+=int(np.any(inputs['rgb'][camera,v,u]!=without_rgb[camera,v,u],axis=-1).sum())
            changed+=int(np.count_nonzero(inputs['depth_m'][camera,v,u]!=without_depth[camera,v,u]))
        support.append(count);response.append(changed)
    centre=np.asarray(labels['cube_pose_world'],float)
    if centre.shape!=(7,)or not np.isfinite(centre).all():raise ValueError('Label-only actual Cube pose required')
    if metadata.get('cube_scene_present')is not True:raise ValueError('Physical Cube exists; availability is separate')
    if shot_id in ('shot00','shot02'):
        if metadata.get('_verified_absence_for_supervision')is True:
            if sum(support):raise ValueError('Current Cube depth influence contradicts constructed outside-frusta negative')
            target,reason=0,'constructed_outside_all_camera_frusta'
        else:target,reason=-1,'outside_position_requested_but_four_frusta_absence_unproven_unknown'
    elif shot_id=='shot01':
        # This is scope-negative, never a claim of semantic object absence.
        if metadata.get('construction')!='outside_scope' or np.linalg.norm(centre[:3]-inputs['public_sphere_center'])<=.20:
            raise ValueError('Predeclared actual outside20cm scope label required')
        target,reason=0,'constructed_outside_measurement_scope'
    elif sum(support)>0:target,reason=1,'current_crop_Cube_depth_influence'
    elif sum(response)==0:target,reason=0,'counterfactually_identical_model_RGBD_image_unobservable'
    else:target,reason=-1,'no_qualified_depth_owner_but_current_RGBD_response_unknown'
    return dict(presence_target=target,centre_valid=target==1,centre_target_world=centre[:3].astype(np.float32),
        reason=reason,near_owner_pixels=support[0],wide_owner_pixels=support[1],crop_RGBD_response_counts=response,
        pure_face_or_full_instance_claim=False,physical_existence_label=False)


class DualCropCubeDataset:
    def __init__(self,manifest_path,role='fit',*,allow_reserved=False):
        check_frozen_definitions();self.path=Path(manifest_path).resolve(strict=True)
        if any('test' in p.lower()for p in self.path.parts):raise ValueError('TRAIN only, no TEST')
        if role not in ROLE_COUNTS or role=='reserved_fresh_centre_VAL'and not allow_reserved:
            raise ValueError('Reserved getters require explicit post-selection evaluation')
        self.role=role;self.manifest=read_json(self.path);m=self.manifest;self.bound_pointers={}
        def pointer(ref,base=None):
            p=checked_pointer(self.path.parent if base is None else base,ref);self.bound_pointers[str(p)]=ref['sha256'];return p
        if m.get('schema')!=SCHEMA or m.get('scope')!=SCOPE or m.get('dataset_split')!='TRAIN' or m.get('label_protocol')!=LABEL_PROTOCOL or any(m.get(k)is not True for k in ('complete','all_native_closed','all_process_exits_observed','source_unchanged')):
            raise ValueError('Complete closed CF pilot40 manifest required')
        prep=pointer(m['preparation_manifest']);shared=pointer(m['shared_registry']);plan_path=pointer(m['capture_plan'])
        if sha256(prep)!=PREPARATION_SHA or sha256(shared)!=SHARED_REGISTRY_SHA:raise ValueError('Fixed shared source partition required')
        plan=read_json(plan_path);self.capture_plan=plan;self.capture_plan_path=plan_path
        registry_path=pointer(plan['TRAIN_q_registry'],ROOT)
        if sha256(registry_path)!=REGISTRY_SHA:raise ValueError('Preassigned40 TRAIN q registry changed')
        registry_rows=read_json(registry_path)['rows'];q_rows={r['parent_id']:r for r in registry_rows}
        expected=pilot_metadata();ids=tuple(p['parent_id']for p in expected)
        if plan.get('schema')!=PLAN_SCHEMA or plan.get('scope')!=SCOPE or plan.get('split')!='train' or tuple(p['parent_id']for p in plan.get('parents',[]))!=ids or tuple(p['parent_id']for p in m.get('parents',[]))!=ids:
            raise ValueError('Whole fixed40 source parents required in original order')
        if tuple(r['parent_id']for r in registry_rows)!=ids:raise ValueError('Original ordered40 q registry rows required')
        if plan.get('render_warmup_rounds')!=12 or plan.get('label_method')!='counterfactual_current_depth_owner' or plan.get('native_instance_id_masks')is not False or plan.get('material_or_alpha_edits')is not False or plan.get('preparation_only')is not False or plan.get('TEST_read')is not False:
            raise ValueError('Original normal-render depth-owner protocol required')
        self.all_records=[];self.capture_source_identity=None;groups=set();data_paths=set()
        # Entire closure is checked without any NPZ decode, even for fit/selection.
        for parent,planned,old in zip(m['parents'],plan['parents'],expected):
            pid=parent['parent_id'];declared=old['role']
            if parent.get('role')!=declared or planned.get('role',planned.get('fold'))!=declared or parent.get('seed')!=old['seed']or planned.get('seed')!=old['seed']:
                raise ValueError('Fixed role/seed changed')
            group=old['source_parent_group']
            if group in groups or parent.get('source_parent_group')!=group:raise ValueError('Source episode cannot repeat/cross folds')
            groups.add(group);q=planned.get('TRAIN_q_source',{})
            if q!=q_rows[pid]or q.get('source_parent_group')!=group:raise ValueError('Whole planned TRAIN_q_source must equal fixed registry row')
            if q.get('file')!=old['source_HDF5']or q.get('dataset')!=old['q_source_dataset']or q.get('index')!=0:
                raise ValueError('Exact metadata-first q0 source changed')
            if set(parent['capture'])!={'result','owner_actual_exit','run_inputs','source_manifest'}:raise ValueError('Four capture pointers required')
            paths={k:pointer(v)for k,v in parent['capture'].items()};result,owner,run,sources=(read_json(paths[k])for k in ('result','owner_actual_exit','run_inputs','source_manifest'))
            case=paths['result'].parent
            if paths['run_inputs']!=case/'run_inputs.json'or paths['source_manifest']!=case/'source_snapshot/manifest.json'or paths['owner_actual_exit']!=case.with_name(pid+'.owner.actual_exit.json'):
                raise ValueError('Exact case and sibling-owner binding required')
            if type(owner.get('actual_process_returncode'))is not int or owner['actual_process_returncode']!=0 or owner.get('actual_wait_completed')is not True or owner.get('source_unchanged')is not True or owner.get('stop_reason')is not None or owner.get('parent_id')!=pid or Path(owner.get('output','')).resolve()!=case:
                raise ValueError('Actual closed0/reaped/samecase owner required')
            if any(result.get(k)is not True for k in ('complete','native_closed','source_unchanged'))or result.get('parent_id')!=pid or type(result.get('controlled_physics_ticks'))is not int or result['controlled_physics_ticks']!=0:
                raise ValueError('Zero-tick closed capture required')
            if run.get('parent_id')!=pid or run.get('seed')!=old['seed']or run.get('fold')!=declared or run.get('TRAIN_q_source')!=q or run.get('plan_sha256')!=sha256(plan_path):raise ValueError('Actual planned parent/source differs')
            identity=dict(repo=run['source_sha256'],external=run['external_source_sha256'],native=run['Native_sources_sha256'])
            if any(not value for value in identity.values()):raise ValueError('Nonempty actual source fingerprints required')
            if identity['repo']!=plan['source_sha256']or identity['external']!=plan['external_source_sha256']or result.get('Native_sources_sha256_after')!=identity['native']:raise ValueError('Capture source graph changed')
            if self.capture_source_identity is not None and identity!=self.capture_source_identity:raise ValueError('One common actual capture source graph required')
            self.capture_source_identity=identity
            archived={}
            for item in sources['files']:
                p=(paths['source_manifest'].parent/item['archive']).resolve(strict=True)
                if paths['source_manifest'].parent not in p.parents or sha256(p)!=item['sha256']or item['original']in archived:raise ValueError('Byte archive missing/duplicate/changed')
                self.bound_pointers[str(p)]=item['sha256'];archived[item['original']]=item['sha256']
            expected_sources={str((ROOT/k).resolve()):v for k,v in identity['repo'].items()};expected_sources.update(identity['external'])
            if any(archived.get(k)!=v or sha256(k)!=v for k,v in expected_sources.items()):raise ValueError('All current/archive sources must match planned SHA')
            archived_plan=case/'source_snapshot/plan.json'
            if sha256(archived_plan)!=sha256(plan_path):raise ValueError('Archived plan changed')
            initial=pointer(parent['initial_public_rgbd']);sphere=pointer(parent['public_sphere_fit'])
            if initial!=case/'initialization/public_rgbd.npz'or sphere!=case/'public_initial_sphere.json':raise ValueError('Original public bootstrap pointers required')
            if tuple(s.get('shot_id')for s in parent['shots'])!=SHOT_IDS or tuple(s.get('shot_id')for s in result['shots'])!=SHOT_IDS or tuple(s.get('shot_id')for s in planned['shots'])!=SHOT_IDS:raise ValueError('All13 ordered unique shots required')
            for shot,frame,spec in zip(parent['shots'],result['shots'],planned['shots']):
                if any(shot.get(k)!=spec.get(k)or frame.get(k)!=spec.get(k)for k in ('shot_id','condition','condition_detail'))or shot.get('scene_id')!=pid+'/'+shot['shot_id']or frame.get('scene_id')!=shot['scene_id']:
                    raise ValueError('Exact per-shot planned scene/condition identity required')
                camera=frame.get('camera',{})
                if frame.get('timestamp')!=0 or type(frame.get('physics_ticks'))is not int or frame['physics_ticks']!=0 or frame.get('independent_scene')is not True or frame.get('augmentation',{}).get('restored')is not True or camera.get('label_method')!='counterfactual_current_depth_owner' or camera.get('warmup_rounds_fixed')!=12 or any(camera.get(k)is not True for k in ('original_clone_equal','all_counterfactual_flags_restored','all_restore_RGBD_equal','clock_unchanged','handling_restored','global_vision_sensors_enabled','created_nonpassive')):
                    raise ValueError('Actual restored normal-render current shot required')
                paths2={}
                for key,fkey,folder in (('inputs','input','inputs'),('labels','labels','labels'),('annotation_metadata','annotation_metadata','labels')):
                    p=pointer(shot[key]);actual=checked_pointer(case,frame[fkey])
                    if p!=actual or p.parent!=case/folder or p in data_paths:raise ValueError('Original unique input/labels sibling SHA required')
                    data_paths.add(p)
                    paths2[key]=p
                metadata=read_json(paths2['annotation_metadata'])
                if any(metadata.get(k)!=shot.get(k)for k in ('scene_id','shot_id','condition','condition_detail')):raise ValueError('Label metadata differs from exact scene')
                if metadata.get('role')!=declared or metadata.get('source_parent_group')!=group or metadata.get('source_task')!=old['source_task']or metadata.get('intended_scope_m')!=.20 or metadata.get('cube_scene_present')is not True:raise ValueError('Actual label-only role/source/scope metadata required')
                if shot['shot_id']=='shot01'and (metadata.get('construction')!='outside_scope'or 'scope_negative_proof'not in metadata):raise ValueError('Explicit predeclared outside-scope construction proof required')
                metadata,proof_bindings=validated_target_metadata(case,metadata,shot['shot_id'])
                self.bound_pointers.update(proof_bindings)
                self.all_records.append(dict(**shot,parent_id=pid,role=declared,source_task=old['source_task'],source_parent_group=group,paths=paths2,metadata=metadata,initial_path=initial,sphere_path=sphere))
        self.bound_pointers[str(self.path)]=sha256(self.path)
        self.records=[r for r in self.all_records if r['role']==role]
        if len(self.all_records)!=520 or len(self.records)!=ROLE_COUNTS[role]*13:raise ValueError('Whole fixed520 ledger required')
        self.access=[]

    def __len__(self):return len(self.records)

    def public_sample(self,index):
        from .features import public_model_arrays
        from ..near_goal_segmentation_20261103.data import validate_public
        record=self.records[index];inputs=load_npz(record['paths']['inputs']);validate_public(inputs)
        if float(inputs['timestamp'])!=0:raise ValueError('Static capture timestamp required')
        initial=load_npz(record['initial_path']);validate_public(initial);fit=read_json(record['sphere_path'])
        if fit.get('Native_pose_used')is not False or fit.get('metadata',{}).get('simulator_pose_used')is not False or not np.array_equal(inputs['public_sphere_center'],initial['public_sphere_center'])or not np.array_equal(initial['public_sphere_center'],fit['sphere']['center']):raise ValueError('Same original publicSphere bootstrap required')
        self.access.append(dict(scene_id=record['scene_id'],role=self.role,kind='public_only'))
        return dict(scene_id=record['scene_id'],parent_id=record['parent_id'],role=self.role,record=record,inputs=inputs,model_inputs=public_model_arrays(inputs))

    def __getitem__(self,index):
        sample=self.public_sample(index);record=sample['record'];labels=load_npz(record['paths']['labels'])
        sample['targets']=derive_targets(sample['inputs'],labels,record['metadata'],record['shot_id'])
        self.access.append(dict(scene_id=record['scene_id'],role=self.role,kind='label_supervision'))
        return sample

    def revalidate(self):
        if any(sha256(p)!=digest for p,digest in self.bound_pointers.items()):raise ValueError('Bound dataset/source bytes changed')
        return dict(files=len(self.bound_pointers),sha256=hashlib.sha256(json.dumps(self.bound_pointers,sort_keys=True).encode()).hexdigest())
