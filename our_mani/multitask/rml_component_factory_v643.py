"""Build public RML inputs directly, without the old TRAIN-only control stack.

The same perception, conditional ring/Buzz evidence, role rules, mesh support,
TRAIN operations and shared Context predictor are retained. No former Hermite
controller/dispatcher wrapper is installed. A genuine evaluation capability is
required before constructing any evaluation component.
"""
from contextlib import ExitStack
from pathlib import Path
import json
from .evaluation import file_sha,digest
from .rml_component_domain_v640 import ComponentDomain
from .specs import TASK_SPECS
from . import factory


def load_library(root,config,task,domain):
    if not isinstance(domain,ComponentDomain):raise ValueError('sealed case component domain required')
    domain.require(task)
    path=factory.artifact_path(root,config['learned_artifacts'][task]['interaction'])
    if task.startswith('push_moving_button'):
        from .button_command_library_v641 import ButtonCommandLibrary
        library=ButtonCommandLibrary(root,path,domain=domain)
    else:
        if task=='place_cups_on_rotating_frame':
            from .normal_cup_primary_library_v109 import PrimarySequenceLibrary as Library
        elif task=='insert_onto_rotating_peg_high_speed':
            from .high_peg_primary_fitting_v231 import PrimarySequenceLibrary as Library
        else:
            from .interaction import InteractionLibrary as Library
        library=Library.load(path)
        fit=library.metadata.get('fit_configuration',{})
        if (library.metadata.get('development_support_override') is not False
            or fit.get('development_only') is not False or fit.get('minimum_completed_parent_sources',0)<8):
            raise ValueError('original completed-TRAIN support/error gates remain required')
    if library.task_spec.task_name!=task:raise ValueError('actual operation library task differs')
    return library


def identity(root,config,task,method,library):
    if method not in ('cv_geometry','context_geometry') or task not in TASK_SPECS:
        raise ValueError('only fourteen registered public CV/Context RML component providers')
    root=Path(root);parts=config['learned_artifacts'][task]
    names=('rml_component_domain_v640.py','button_command_library_v641.py','colour_component_domain_v642.py',
        'rml_component_factory_v643.py','pipeline.py','perception.py','forecasting.py','mode_observer.py',
        'interaction.py','factory.py','public_roles.py','context_se3.py','physmani_geometry_policy_v518.py')
    value=dict(schema='r10_direct_public_RML_component_identity_v643',task=task,method=method,
        learned_operation_artifacts={k:v for k,v in parts.items() if k in ('geometry','interaction','rigid_layout','colour')},
        shared_context_checkpoint=config['context_checkpoint'],operation_library_sha256=library.sha256,
        registration_config=config.get('registration_config',{}),forecast_config=config.get('forecast_config',{}),
        candidate_config=config['candidate_config'],body_orientation_observability=config['body_orientation_observability'][task],
        sources_sha256={f'our_mani/multitask/{n}':file_sha(root/'our_mani/multitask'/n) for n in names},
        old_Hermite_controller_or_TRAIN_dispatcher_installed=False,neural_value_model_loaded=False)
    return dict(value,sha256=digest(value))


def create_actor(root,env,config,method,descriptions,*,domain):
    from .interaction import CandidateConfig,MPCConfig,GripperTimingConfig
    from .pipeline import PipelineActor
    from .perception import MultiBodySE3Tracker,RegistrationConfig
    from .forecasting import MultiBodyForecastBridge,ForecastConfig
    from .training_geometry import body_templates_from_asset,declared_instance_bindings,register_collision_asset
    root=Path(root).resolve();task=env.spec.task_name
    if not isinstance(domain,ComponentDomain):raise ValueError('sealed component domain required')
    purpose=domain.require(task,method)
    if task not in TASK_SPECS or method not in ('cv_geometry','context_geometry'):
        raise ValueError('explicit core14 public geometry component provider only')
    library=load_library(root,config,task,domain);proof=identity(root,config,task,method,library)
    if config.get('direct_RML_component_identity_v643',{}).get(task,{}).get(method)!=proof:
        raise ValueError('freeze actual direct public component material and executable identity')
    asset=json.loads(factory.artifact_path(root,config['learned_artifacts'][task]['geometry']).read_text())
    finite=factory.load_finite_rotation_symmetries(root,config,task,asset)
    cups=factory.load_cup_feature_models(root,config,task,asset)
    templates=body_templates_from_asset(asset,declared_instance_bindings(env,asset,shared_permission=True),
        orientation_observability=config['body_orientation_observability'][task],
        sphere_radii=config.get('body_sphere_radii',{}).get(task),symmetry_axes=config.get('body_symmetry_axes',{}).get(task),
        axis_polarity=config.get('body_axis_polarity',{}).get(task),finite_rotation_symmetries=finite,cup_feature_models=cups)
    pointer=config['learned_artifacts'][task].get('rigid_layout')
    rigid=factory.load_rigid_layout(factory.artifact_path(root,pointer),task) if pointer else ()
    tracker=MultiBodySE3Tracker(templates,task_spec=env.spec,rigid_children=rigid,
        config=RegistrationConfig(**config.get('registration_config',{})))
    with ExitStack() as scopes:
        if env.spec.family=='put_rubbish_in_moving_bin':
            from .public_held_mesh_v33 import install_registration
            scopes.enter_context(install_registration(env.map_service,config,root,task))
        if env.spec.family=='moving_basketball_in_hoop':
            from .task_spherical_enclosure_v18 import install_registration
            scopes.enter_context(install_registration(env.map_service,config,root,task))
        register_collision_asset(env.map_service,asset,
            orientation_observability=config['body_orientation_observability'][task],
            axis_swept_assets=config.get('axis_swept_assets',{}).get(task),finite_rotation_symmetries=finite)
    predictor=method.split('_')[0];context=None
    if predictor=='context':
        from .context_se3 import MultiBodyContextDynamics
        context=MultiBodyContextDynamics(factory.artifact_path(root,config['context_checkpoint']))
    bridge=MultiBodyForecastBridge(tuple(t.role_id for t in templates),method=predictor,context=context,
        rigid_children=rigid,task_spec=env.spec,config=ForecastConfig(**config.get('forecast_config',{})))
    resolver=None
    if env.spec.family in ('insert_onto_rotating_peg','push_moving_button'):
        from .colour_component_domain_v642 import QualifiedQuantileRoleResolver
        artifact=json.loads(factory.artifact_path(root,config['learned_artifacts'][task]['colour']).read_text())
        resolver=QualifiedQuantileRoleResolver(env.spec,artifact,root=root,shared_semantic_permission=True,domain=domain)
    affordance=None
    if env.spec.family=='insert_onto_rotating_peg':
        if task=='insert_onto_rotating_peg':
            from .ring_runtime_v25 import RuntimeRingAffordance
            settings=config['ring_runtime_v25'][task]
        else:
            from .high_peg_ring_runtime_v243 import RuntimeRingAffordance
            settings=config['high_peg_ring_runtime_v243'][task]
        affordance=RuntimeRingAffordance.load(root,config['ring_affordances'][task],replay_report=settings['replay_report'])
    actor=PipelineActor(env.spec,templates,tracker,bridge,library,env.robot_model,env.map_service,
        CandidateConfig(**config['candidate_config']),MPCConfig(**config['mpc_config']),predictor=predictor,ranking='geometry',
        descriptions=descriptions,role_resolver=resolver,attachment_association_distance_m=config['attachment_association_distance_m'],
        continuation_sha256=proof['sha256'],purpose='evaluation' if purpose=='evaluation' else 'train_collection',
        gripper_timing_config=GripperTimingConfig(.02,.1),joint_limit_recovery_margin_rad=config['joint_limit_recovery_margin_rad'],
        ring_affordance=affordance)
    if env.spec.family=='put_rubbish_in_moving_bin':
        from .public_held_mesh_v33 import install
        install(actor,config,root,task)
    elif env.spec.family=='moving_basketball_in_hoop':
        from .task_spherical_enclosure_v18 import bind_geometry_port
        from .supported_sphere_attachment_v10 import install
        bind_geometry_port(actor);install(actor,config,root)
    elif env.spec.family=='beat_the_rotating_buzz':
        from .buzz_runtime_v64 import install
        install(actor,config,root,task)
    elif env.spec.family=='insert_onto_rotating_peg':
        if task=='insert_onto_rotating_peg':from .ring_operational_gauge_v60 import install
        else:from .high_peg_operational_gauge_v244 import install
        install(actor,config,root,task)
    elif env.spec.family=='remove_cups_from_rotating_frame':
        from .remove_runtime_v48 import install
        install(actor,config,root,task)
    proof=dict(proof,component_domain=domain.receipt(),actual_pipeline_purpose=actor.purpose)
    return actor,proof


def cpu_preflight(root,config,task,method,domain,files):
    """Validate the actual new loaders; no old TRAIN-only actor is constructed."""
    from .frozen_subset_batch_v132 import checked_subset
    library=load_library(root,config,task,domain);proof=identity(root,config,task,method,library)
    if config.get('direct_RML_component_identity_v643',{}).get(task,{}).get(method)!=proof:
        raise ValueError('freeze actual direct CPU model loader identity')
    checked_subset(root,files,{str(Path(root)/name):sha for name,sha in proof['sources_sha256'].items()})
    training=None
    if method=='context_geometry':
        from .context_se3 import MultiBodyContextDynamics
        from .joint16_checkpoint_v172 import check_training_metadata_and_roster
        model=MultiBodyContextDynamics(factory.artifact_path(root,config['context_checkpoint']))
        source=model.path.parent/model.metadata['training_source_manifest']['path_relative_to_checkpoint']
        roster=Path(root)/'artifacts/framework/protocol/train_manifest.json'
        checksum=model.metadata['training_source_manifest']['sha256']
        if file_sha(source)!=checksum:raise ValueError('actual shared model TRAIN source proof changed')
        checked_subset(root,files,{str(source):checksum,str(roster):file_sha(roster)})
        training=check_training_metadata_and_roster(model.metadata,json.loads(source.read_text())['entries'],
            json.loads(roster.read_text())['episodes'])
    return dict(component_identity=proof,component_domain=domain.receipt(),training=training,
        old_TRAIN_actor_constructor_executed=False,actual_model_loader_validation=True,native_started=False,GPU_started=False)
