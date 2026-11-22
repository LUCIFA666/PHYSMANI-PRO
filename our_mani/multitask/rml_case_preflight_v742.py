"""Genuine full16 source gates for the explicit author-outcome native worker.

TRAIN qualification uses the original fixed four. VAL/TEST use their genuine
source and component purposes. TEST cannot read its source manifest until the
complete unchanged-policy VAL release is accepted by the common suite gate.
"""
from pathlib import Path
import json
from .evaluation import case_identity,digest,file_sha
from .evaluation_roster16_v283 import TASKS,METHODS,validate_roster
from .input_stat_cache_v79 import verify_inputs
from .formal_suite_admission_v532 import locked_json,locked_pointer
from .common_rml_protocol_v530 import binding as common_binding,fingerprint
from .physmani_geometry_executor_v517 import binding as executor_binding,SOURCES as EXECUTOR_SOURCES
from .rml_component_domain_v640 import for_train,for_evaluation
from .rml_component_extensions_v708 import cpu_preflight
from .rml_policy_components_v702 import binding as policy_binding,SOURCES as POLICY_SOURCES
from .joint16_tracks_v31 import BLOCKS

SOURCES=tuple(dict.fromkeys((*EXECUTOR_SOURCES,*POLICY_SOURCES,
    'our_mani/multitask/native_outcome_sampling_v741.py',
    'our_mani/multitask/rml_case_preflight_v742.py',
    'our_mani/multitask/author_outcome_receipt_v749.py',
    'tests/test_author_outcome_runtime_v743.py',
    'scripts/run_task_executor_smoke_author_outcome_v743_r10.py',
    'scripts/run_task_executor_smoke_author_suite16_v744_r10.py',
    'scripts/run_task_executor_smoke_rml_inputs_v709_r10.py',
    'scripts/run_task_executor_smoke_rml_suite16_v664_r10.py',
    'our_mani/multitask/rml_case_preflight_v710.py','our_mani/multitask/rml_component_extensions_v708.py',
    'our_mani/multitask/rml_cup_support_v707.py','our_mani/multitask/cup_public_support_v148.py','our_mani/multitask/physmani_block_policy_v538.py',
    'our_mani/multitask/physmani_block_receipt_v541.py','our_mani/multitask/physmani_block_record_v545.py',
    'our_mani/multitask/baseline_measured_windows_v345.py','our_mani/multitask/native_window_observer_v191.py',
    'our_mani/multitask/joint16_block_context_v97.py','our_mani/multitask/common_rml_protocol_v530.py',
    'our_mani/multitask/physmani_executor_v7.py','our_mani/sensor_alignment/physmani_executor_v6.py',
    'our_mani/aligned/physmani.py','our_mani/sensor_alignment/author_rgbd.py',
    'our_mani/multitask/formal_suite_admission_v532.py','our_mani/multitask/three_method_release_v531.py',
    'our_mani/multitask/evaluation_roster16_v283.py')))


def pointer(root,path):
    root=Path(root).resolve();path=Path(path).resolve()
    if not path.is_relative_to(root):raise ValueError('suite pointer must remain in project')
    return dict(path=str(path.relative_to(root)),sha256=file_sha(path))


def outcome_protocol_binding(root):
    root=Path(root).resolve()
    return dict(schema='r10_author_action_outcome_protocol_v741',
        implementation=pointer(root,root/'our_mani/multitask/native_outcome_sampling_v741.py'),
        runtime_consumer=pointer(root,root/'scripts/run_task_executor_smoke_author_suite16_v744_r10.py'),
        original_runtime=pointer(root,root/'scripts/run_task_executor_smoke_rml_suite16_v664_r10.py'),
        failure_sampling='current_native_result_at_explicit_action_end',
        success_sampling='original_per_tick_success_latch',pose_action_max_ticks=10,public_hold_max_ticks=1,
        budget_partial_action_consumed=True,native_RML_parameters_unchanged=True,
        block_native_fail_conditions=0,block_extra_task_success_calls=0)


_indices={}

def _checked(root,files,path):
    # Code/models/reset inputs may be explicitly locked external dependencies;
    # suite evidence pointers themselves retain the stricter project-only gate.
    from .frozen_input_edges_v463 import FrozenInputIndex,InputTopologyChanged
    path=Path(path).resolve();key=(str(root),id(files))
    for _ in range(3):
        try:
            index=_indices.get(key)
            if index is None or index.files is not files or not index.valid():
                index=FrozenInputIndex(root,files)
                if len(_indices)>=4:_indices.clear()
                _indices[key]=index
            index.checked(path)
            return path
        except InputTopologyChanged:_indices.pop(key,None)
    raise ValueError('declared used-input edges remain unstable')


def _suite_route(root,args):
    if not args.suite or not args.suite_lock:raise ValueError('formal execution needs the frozen complete suite and evidence lock')
    lock=Path(root)/args.suite_lock;files=json.loads(lock.read_text())['files'];verify_inputs(root,files)
    path=_checked(root,files,Path(root)/args.suite);suite=json.loads(path.read_text())
    routes=locked_json(root,files,suite['routes'])
    from .three_method_release_v531 import policy_fingerprint
    policy=policy_fingerprint(routes)
    selected=[r for r in routes['routes'] if r['task']==args.task and r['method']==args.method]
    if len(selected)!=1:raise ValueError('one explicit immutable source/method route required')
    route=selected[0]
    for key,arg in (('configuration',args.config),('frozen_inputs',args.frozen_inputs)):
        actual=locked_pointer(root,files,route[key])
        if actual!=(Path(root)/arg).resolve():raise ValueError('actual argv differs from the frozen method route: '+key)
    return suite,files,route,policy,pointer(root,lock)


def prepare(root,args,worker):
    root=Path(root).resolve();worker=Path(worker).resolve()
    if args.task not in TASKS or args.method not in METHODS or args.split not in ('train','validation','test'):
        raise ValueError('exactly full16 tasks and the three authorized methods')
    if args.replica!=0 or args.exploration_seed is not None or args.record_demo or args.demo_only:
        raise ValueError('original unaugmented native cases; no exploration or video retry')
    formal=args.split!='train'
    if args.development_demo is formal:raise ValueError('TRAIN requires development scope; VAL/TEST require formal scope')
    config_path=(root/args.config).resolve();lock_path=(root/args.frozen_inputs).resolve()
    lock=json.loads(lock_path.read_text());files=lock['files'];verify_inputs(root,files)
    _checked(root,files,config_path);_checked(root,files,worker)
    config=json.loads(config_path.read_text())
    profile=common_binding(root,config['baseline_repo'])
    if config.get('common_author_RML_profile')!=profile:raise ValueError('same original common RML profile required')
    if (config['dt']!=.05 or config['simulation_seconds']!=10. or config['gripper_velocity']!=.2
        or config['baseline_decision_ticks']!=10):raise ValueError('shared physical clock/gripper protocol required')
    for name in SOURCES:_checked(root,files,root/name)
    for name in profile['native_sources_sha256']:_checked(root,files,Path(config['baseline_repo'])/name)
    suite=suite_files=route=policy=suite_lock_pointer=validation_release=None
    source_files=files;source_lock_path=None;source_lock_sha256=None
    if formal or args.suite or args.suite_lock:
        suite,suite_files,route,policy,suite_lock_pointer=_suite_route(root,args)
        if locked_pointer(root,suite_files,route['worker'])!=worker:raise ValueError('actual native worker changed from route')
    if args.split=='test':
        if not args.validation_release:raise ValueError('complete VAL release required before any TEST source read')
        release_path=(root/args.validation_release).resolve()
        if release_path!=(root/suite['validation_release_path']).resolve():raise ValueError('explicit suite VAL release output required')
        validation_release=json.loads(release_path.read_text())
    if formal:
        domain,row,admission=for_evaluation(root,suite,suite_files,args.split,args.task,args.raw_episode_id,args.method,
            validation_release=validation_release)
        declared=suite['source_manifests'][args.split]
        manifest_path=locked_pointer(root,suite_files,declared)
        if manifest_path!=(root/args.manifest).resolve():raise ValueError('formal source manifest argv differs from admitted source')
        policy=admission['policy_sha256']
    else:
        manifest_path=root/'artifacts/framework/protocol/train_manifest.json'
        if (root/args.manifest).resolve()!=manifest_path:raise ValueError('exact original TRAIN manifest required')
        _checked(root,files,manifest_path)
        rows=validate_roster(json.loads(manifest_path.read_text()),'train')
        domain=for_train(root,files,args.task,args.raw_episode_id,args.method)
        row=next(r for r in rows if r['task']==args.task and r['raw_episode_id']==args.raw_episode_id)
    if formal:
        # Data scopes are split-specific, so reading TEST reset files happens
        # only after complete VAL admission. Their immutable source lock does
        # not change executable/model policy or the VAL/TEST route fingerprint.
        if not args.source_inputs:raise ValueError('formal cases require a post-admission split source input lock')
        source_lock_path=(root/args.source_inputs).resolve()
        if not source_lock_path.is_relative_to(root):raise ValueError('source input lock must remain in project')
        source_lock=json.loads(source_lock_path.read_text());source_lock_sha256=file_sha(source_lock_path)
        if (source_lock.get('schema')!='r10_full16_post_admission_split_source_lock_v1'
            or source_lock.get('split')!=args.split or source_lock.get('source_manifest_sha256')!=file_sha(manifest_path)
            or source_lock.get('source_count')!=(640 if args.split=='test' else 315)
            or source_lock.get('policy_sha256')!=policy or source_lock.get('TEST_used_for_fitting') is not False):
            raise ValueError('actual unchanged-policy full16 split source lock required')
        source_files=source_lock['files'];verify_inputs(root,source_files)
    source=Path(row['path']).resolve()
    if args.split=='test':
        if not source.is_dir() or row.get('mapping_status')!='verified':raise ValueError('original verified official reset alias required')
        for name in ('low_dim_obs.pkl','variation_number.pkl','target_state.pkl'):_checked(root,source_files,source/name)
        for camera in ('8','16','36'):_checked(root,source_files,source/'nerf_data/0/poses'/f'{camera}.pkl')
    else:
        package='train_package_compressed' if args.split=='train' else 'val_package_compressed'
        if not source.is_file() or source.name!='data.h5' or package not in source.parts:raise ValueError('actual original HDF5 camera source required')
        _checked(root,source_files,source)
        if row.get('source_file_stat') is not None:
            st=source.stat();actual=dict(size_bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino,device=st.st_dev)
            if actual!=row['source_file_stat']:raise ValueError('original development HDF5 source stat changed')
    if args.method=='physmani':
        checkpoint=Path(config['baseline_checkpoint']).resolve();_checked(root,files,checkpoint)
        if (checkpoint.name!='epoch_99999.pth' or config['baseline_reference_filter']!='author_direct_setpoint'):
            raise ValueError('original released baseline weights and direct RML setpoints required')
        model_proof=dict(original_author_checkpoint=dict(path=str(checkpoint),sha256=file_sha(checkpoint)),
            original_neural_policy_and_online_world_model=True,GPU_started=False)
    elif args.task in BLOCKS:
        from .joint16_block_context_v97 import Joint16BlockContext
        checkpoint=root/config['context_checkpoint']['path'];_checked(root,files,checkpoint)
        model=Joint16BlockContext(checkpoint,task=args.task,root=root)
        from .joint16_checkpoint_v172 import check_training_metadata_and_roster
        metadata=model.metadata;manifest=checkpoint.parent/metadata['training_source_manifest']['path_relative_to_checkpoint']
        _checked(root,files,manifest);roster=root/'artifacts/framework/protocol/train_manifest.json';_checked(root,files,roster)
        training=check_training_metadata_and_roster(metadata,json.loads(manifest.read_text())['entries'],json.loads(roster.read_text())['episodes'])
        model_proof=dict(actual_shared16_block_adapter=model.binding,training=training,GPU_started=False)
    else:
        expected=executor_binding(root,config['baseline_repo'])
        if config.get('physmani_geometry_executor_v517')!=expected:raise ValueError('actual geometry RML executor binding changed')
        contract=policy_binding(root,config,args.task)
        if config.get('direct_RML_policy_components_v702',{}).get(args.task)!=contract:raise ValueError('freeze actual current policy components')
        model_proof=cpu_preflight(root,config,args.task,args.method,domain,files)
    out=(root/args.out).resolve();base=root/'artifacts/framework/multitask_r10'
    if not out.is_relative_to(base) or out==base or out.exists():raise ValueError('fresh isolated native case output required')
    if formal and any('demo_only' in part for part in out.relative_to(base).parts):raise ValueError('formal cases cannot write a demo namespace')
    if not formal and 'development_demo_only' not in out.parts:raise ValueError('TRAIN qualification requires an explicit development namespace')
    identity=case_identity(split=args.split,task=args.task,raw_episode_id=args.raw_episode_id if args.split=='test' else args.raw_episode_id+'__policy_aug_r0',
        method=args.method,source_row=row,config_sha256=file_sha(config_path),inputs_sha256=file_sha(lock_path),
        parent_raw_episode_id=None if args.split=='test' else args.raw_episode_id,replica=0,demo_only=not formal,exploration_seed=None)
    if not formal:
        identity.pop('sha256',None);identity.update(development_demo_only=True,fitting_source=False,formal_validation_metric=False)
        identity['sha256']=digest(identity)
    if source_lock_sha256 is not None:
        model_proof=dict(model_proof,split_source_input_lock=dict(path=str(source_lock_path.relative_to(root)),sha256=source_lock_sha256))
        identity.pop('sha256',None);identity['source_input_lock_sha256']=source_lock_sha256;identity['sha256']=digest(identity)
    return dict(args=args,config=config,config_path=config_path,lock_path=lock_path,lock_sha256=file_sha(lock_path),
        files=files,row=row,original_row=row,manifest_path=manifest_path,manifest_sha256=file_sha(manifest_path),
        out=out,identity=identity,component_domain=domain,model_preflight=model_proof,common_author_RML_profile=profile,
        common_RML_profile_sha256=fingerprint(profile),full16_policy_sha256=policy,suite_lock_pointer=suite_lock_pointer,
        validation_release=validation_release,record_demo=False,formal=formal,source_files=source_files,
        source_lock_path=source_lock_path,source_lock_sha256=source_lock_sha256)


def verify(root,p):
    verify_inputs(root,p['files'])
    if p.get('source_lock_path') is not None:
        if file_sha(p['source_lock_path'])!=p['source_lock_sha256']:raise ValueError('split source lock changed during execution')
        verify_inputs(root,p['source_files'])
    if file_sha(p['lock_path'])!=p['lock_sha256'] or file_sha(p['config_path'])!=p['identity']['config_sha256']:
        raise ValueError('case configuration or external input lock changed during execution')
    if file_sha(p['manifest_path'])!=p['manifest_sha256']:raise ValueError('actual source manifest changed during execution')
