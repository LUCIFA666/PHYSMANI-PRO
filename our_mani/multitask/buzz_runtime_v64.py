"""Source-bound live adapter for the unchanged v63 conditional registration."""
from copy import deepcopy
from .buzz_branch_v63 import PriorConditionedRegistration,SCHEMA


def binding(root,protocol,task):
    from pathlib import Path
    import json
    from .evaluation import file_sha
    from .two_center_pose import load_two_center_pose_prior
    root=Path(root).resolve();base=root/'artifacts/framework/multitask_r10'
    def ptr(path):return dict(path=str(path.relative_to(root)),sha256=file_sha(path))
    model=base/'buzz_two_center_TRAIN_models_and_teacher_scope_v3/models'/task/'model.json'
    source_audit=base/'buzz_initial_branch_source_audit_v61/report.json'
    replay=base/'buzz_initial_rgbd_replay_v63/report.json'
    audit=json.loads(source_audit.read_text());rp=json.loads(replay.read_text())
    if not audit['all_200_original_TRAIN_parents_retained'] or not audit['all_initials_passed']:raise ValueError('full initial TRAIN branch audit required')
    if not rp['first_five_dense_observations_qualified'] or not rp['tasks'][task]['first_five_dense_observations_qualified']:raise ValueError('actual public RGBD integration replay required')
    asset=protocol['learned_artifacts'][task]['geometry']
    load_two_center_pose_prior(root,ptr(model),canonical_asset_pointer=asset)
    files={str(Path(__file__).relative_to(root)):file_sha(__file__),**audit['inputs_sha256'],**rp['inputs_sha256']}
    files[rp['source']['path']]=rp['source']['sha256']
    for name,checksum in files.items():
        if file_sha(root/name)!=checksum:raise ValueError('buzz branch evidence/source bytes changed: '+name)
    files.update({str(source_audit.relative_to(root)):file_sha(source_audit),str(replay.relative_to(root)):file_sha(replay)})
    return dict(schema=SCHEMA,task=task,model=ptr(model),geometry=asset,source_audit=ptr(source_audit),RGBD_replay=ptr(replay),
        inputs_sha256=files,conditional_orientation_protocol=True,
        initialization='original public open-gripper, no-contact observation at t=0 only; 0.15rad prior support',
        thereafter='current RGBD likelihood conditioned on previous qualified public orientation; max angular speed times dt +0.15rad',
        maximum_angular_speed=protocol['candidate_config']['maximum_angular_speed'],maximum_interobservation_gap_seconds=.1,
        absolute_RMS_inlier_gates_changed=False,grasp_collision_joint_task_gates_changed=False,
        physical_CAD_symmetry_claimed=False,post_action_pair_rigidity_claimed=False,native_dynamic_labels_used=False,
        old_Context_orientation_training_qualified=False,formal_release=False)


def install(actor,protocol,root,task):
    from types import MethodType
    from .two_center_pose import load_two_center_pose_prior
    expected=binding(root,protocol,task)
    if protocol.get('buzz_runtime_v64',{}).get(task)!=expected:raise ValueError('freeze actual buzz conditional tracker binding')
    prior=load_two_center_pose_prior(root,expected['model'],canonical_asset_pointer=expected['geometry'])
    adapter=PriorConditionedRegistration(actor.tracker,prior,maximum_angular_speed=expected['maximum_angular_speed'],
        maximum_age=expected['maximum_interobservation_gap_seconds'])
    actor.tracker._registration=adapter.register
    previous_act=actor.act
    def act(instance,observation,robot,remaining_seconds):
        adapter.set_public_context(timestamp=robot.timestamp,gripper_open=robot.gripper_open,contact=robot.contact)
        return previous_act(observation,robot,remaining_seconds)
    actor.act=MethodType(act,actor)
    previous_observe=actor.tracker.observe
    def observe(instance,*args,**kwargs):
        objects,metadata=previous_observe(*args,**kwargs)
        for role,d in metadata['bodies'].items():
            if d.get('orientation_observable'):
                d['pose_source']='current_RGBD_registration_conditioned_on_TRAIN_initial_branch_and_causal_orientation_prior'
                d['rotation_gauge']='conditional_full_CAD_frame; no unique_unconditioned_SO3_measurement_claimed'
            d['conditional_orientation_v63']=[e for e in adapter.events[-4:] if e['role']==role and abs(e['time']-instance._current_timestamp)<1e-9]
        return objects,metadata
    actor.tracker.observe=MethodType(observe,actor.tracker)
    previous_metrics=actor.metrics_report
    def metrics(instance):return dict(previous_metrics(),buzz_branch_v63=dict(binding=expected,events=deepcopy(adapter.events)))
    actor.metrics_report=MethodType(metrics,actor)
