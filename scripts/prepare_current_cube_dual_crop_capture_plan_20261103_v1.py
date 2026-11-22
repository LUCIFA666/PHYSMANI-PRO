"""Metadata-only dual-crop CF capture proposal; no execution capability.

The existing shared140/16/8/36registry is immutable. This script creates no
Native scene/input, reads no HDF arrays, imports no model and cannot launch.
"""
from collections import Counter
from datetime import datetime
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REGISTRY=ROOT/'artifacts/framework/local_mpc_20261102/current_Cube_shared_TRAIN_source_parent_registry_v1/registry.json'
REGISTRY_SHA='76e836b6a872dce703598b18a694953d4ae48be69458bfde7bbde0c214c3cb9c'
TASKS=('pick_moving_target_on_the_table','pick_moving_target_on_the_table_high_speed')
DIRECTIONS=np.asarray([(1,1,1),(-1,-1,-1),(1,1,-1),(-1,-1,1),(1,-1,1),(-1,1,-1),(-1,1,1),(1,-1,-1)],float)/np.sqrt(3.)
ANGLES=(0.,3.,-3.,5.,-5.,10.,-10.,15.,-15.,3.,-3.,5.,-5.,10.,-10.,0.)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def build():
    if sha(REGISTRY)!=REGISTRY_SHA:raise ValueError('Exact shared source-parent role registry required')
    registry=read(REGISTRY)
    if Counter(r['role']for r in registry['rows'])!=Counter(fit=140,selection=16,reserved_fresh_centre_VAL=8,excluded_consumed_calibration=36):
        raise ValueError('Original140/16/8/36whole-source roles required')
    # All13 recipes are fixed before any proposed new scene/visibility result.
    recipes=[dict(shot_id='shot00',condition='sphere_only_parked',cube_layout='outside_all_four_camera_frusta',hand_layout='TRAIN_q0_asset_legal_open',
                  supervision='frusta_absence_proof→availability0; no centre target'),
             dict(shot_id='shot01',condition='cube_outside_scope_nearhand',distance_from_public_sphere_m=.25,
                  hand_layout='nearhand_asset_rule',supervision='outside_scope0,notsemanticabsence'),
             dict(shot_id='shot02',condition='sphere_only_nearhand',cube_layout='outside_all_four_camera_frusta',hand_layout='nearhand_asset_rule',
                  supervision='frusta_absence_proof→availability0; no centre target')]
    for j,d in enumerate((.04,.08,.12,.16,.20)):
        for k,hand in enumerate(('TRAIN_q0_asset_legal_open','asset_derived_nearhand_or_palm_occlusion')):
            recipes.append(dict(shot_id=f'shot{3+2*j+k:02d}',condition='current_cube_near_transition',
                distance_from_public_sphere_m=d,hand_layout=hand,cube_center='initial_publicSphere+d*fixed_direction',
                supervision='currentCFownerinpubliccrops→positive+centreGT; exactnoCubeinputresponse→unobservable0,NOTobjectabsent; otherwiseUNKNOWN/no centreloss'))
    jobs=[];ordinal=Counter()
    eligible=[r for r in registry['rows']if r['role']!='excluded_consumed_calibration']
    for index,row in enumerate(eligible):
        role=row['role'];i=ordinal[role];ordinal[role]+=1
        jobs.append(dict(parent_id=f"current_cube_dual_crop_{row['task']}_{row['parent_raw_episode_id']}",
            source_parent_group=row['parent_source_group'],role=role,source_task=row['task'],
            source_HDF5=row['source_HDF5'],q_source_dataset='low_dim_obs/joint_positions',q_source_index=0,
            q_values_decoded=False,q_legality_pending=True,source_pose_or_expert_action_used=False,
            seed=271103000+index,direction_ordinal=i%8,direction_unit_world=DIRECTIONS[i%8].tolist(),
            cube_rotation_degrees=ANGLES[i%len(ANGLES)],cube_rotation_axis=('x','y','z')[i%3],
            Cube_orientation_is_labels_and_layout_only=True,
            static_placement_is_NOT_grasp_or_attachment=True,physics_ticks_per_shot=0,
            shot_ids=[s['shot_id']for s in recipes],recipes=recipes,
            teacher_preexposed=True,fresh_for_GRU_teacher=False,
            fresh_for_new_visual_head=row['role']=='reserved_fresh_centre_VAL'))
    # Minimal labelled prototype tranche, fixed by metadata, never by outcome.
    pilot=[]
    for task in TASKS:
        pilot.extend([j['parent_id']for j in jobs if j['role']=='fit'and j['source_task']==task][:8])
    pilot.extend(j['parent_id']for j in jobs if j['role']in('selection','reserved_fresh_centre_VAL'))
    previous=read(ROOT/'artifacts/framework/near_goal_segmentation_20261103/position_calibration8_dataset_v10/manifest.json')
    timings=[];timing_sources={}
    for p in previous['parents']:
        path=Path(p['capture']['owner_actual_exit']['path']);owner=read(path);timing_sources[str(path)]=sha(path)
        timings.append((datetime.fromisoformat(owner['finished_utc'])-datetime.fromisoformat(owner['started_utc'])).total_seconds())
    per_shot=sum(timings)/80.
    feature=ROOT/'our_mani/current_cube_rgbd_centre_20261103_v1/features.py'
    model=ROOT/'our_mani/current_cube_rgbd_centre_20261103_v1/model.py'
    return dict(schema='current_cube_dual_crop_CF_TRAIN_capture_preparation_20261103_v1',
        preparation_only=True,execution_authorized=False,capture_ready=False,Native_started=False,training_started=False,
        source_parent_registry=dict(path=str(REGISTRY),sha256=REGISTRY_SHA),whole_source_role_counts=dict(Counter(r['role']for r in registry['rows'])),
        intended_measurement_kind='learned_current_RGBD_near_and_transition_centre_availability_uncertainty',
        capture_scope='static_Cube_visual_snapshots_not_original_HDF_motion_or_physical_grasp_replay',
        proposed_scope_radius_m=.20,actual_scope_qualified=False,not_full_workspace=True,
        features=dict(near='original4x9x64make_near_goal_crops exact',wide='Sphere-centred128pixelwindow, stride2actualpixels→4x9x64, noresize',
            public_only=['currentRGBD','K','T','initial_publicSphere'],forbidden_runtime=['Native_state','robotq','controlledpose','GRU','future','GTlabels']),
        prepared_code=dict(features=dict(path=str(feature),sha256=sha(feature)),model=dict(path=str(model),sha256=sha(model)),
            source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__))),
        remaining_code_and_admission=['Nativecapture/layout/CFhelper/newCLI/sourcefreeze notyetimplemented or authorized.',
            'Readfixedq0,validate7finite/legal withno clipping; freezeplannedposes/assetfinger rules before capture.',
            'OriginalRGBDnormalrenderer/cloneequality, zero-clock/currentfullpose/wholeflagsrestore and finalRGBDrestore must pass.',
            'FixedinitialSphere fitted from realRGBD once whileCubeoutside; never refit after object/hand layout.',
            'Actualwidecrop visiblecoverage/frustum/occlusion must be reported for every distance/layout; no promised20cmcertificate.',
            'Both trainingloaders refuse8reserved/36excluded; freshheadvalidation once only after train/selection/sourceprofile frozen.',
            'Separate operationalquality/uncertainty protocol stillpending; no oldmask995failure relabelledpassed.'],
        labels_only=['actualsameframeCubeXYZ/quaternion','CubeCFdepth/RGBresponse','visibility/absence/unobservableproof','sourceq/handlayout/roles'],
        visibility_semantics=dict(positive='Current Cube depth influence within actual public input crops; NOTpureface or fullinstance mask.',
            unknown='Owner0 without independentabsence/unobservable proof remainsUNKNOWN; no centre or availability loss.',
            unobservable='Only strictcounterfactual equalmodelinput RGBD in bothcrops can label0 as image-unobservable; notobjectabsent.',
            negative='ConstructedSphereOnly/allfrustaproof or explicitscope-negative are separately tagged.',
            raw_HDF_GT_projection_does_not_establish_current_visibility=True),
        jobs=jobs,full_capture=dict(parents=164,fit=140,selection=16,reserved=8,shots=164*13),
        minimal_pilot_capture=dict(parents=40,fit=16,selection=16,reserved=8,shots=40*13,parent_ids=pilot,
            description='Metadata-first smallerdevelopment prototype only, not140fitparents claim; allvalidationparents retained.'),
        estimated_serial_Native=dict(reference_closed_V10_80shots_seconds=sum(timings),reference_owner_receipt_sha256=timing_sources,
            seconds_per_shot_including_launch_average=per_shot,pilot_520shots_seconds=per_shot*520,full_2132shots_seconds=per_shot*2132,
            expected_pilot_minutes_range=[35,50],expected_full_hours_range=[2.5,3.5],
            estimates_not_a_walltime_or_realtime_guarantee=True,GPU_training_time_not_included=True),
        Native_must_wait_for_current_GPU_and_explicit_Root_authorization=True,
        default_check_imports_Native_or_model=False,TEST_data_read=False,
        old_calibration_RGBD_or_labels_used=False,old_calibration_owner_timing_metadata_read=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true');args=p.parse_args()
    if args.execute:raise ValueError('This metadata preparation tool has no execution capability; Native not authorized')
    result=build()
    if args.output:
        if args.output.exists():raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x')as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(preparation_only=True,ready=False,Native_started=False,training_started=False,
        full=result['full_capture'],pilot=result['minimal_pilot_capture']|{'parent_ids':'fixedinmanifest'},
        estimate=result['estimated_serial_Native']|{'reference_owner_receipt_sha256':'8metadatareceipts'},
        output=None if args.output is None else str(args.output)),ensure_ascii=False))


if __name__=='__main__':main()
