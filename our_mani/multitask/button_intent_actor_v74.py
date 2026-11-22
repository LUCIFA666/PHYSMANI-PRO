"""TRAIN-only author-command button policy using the shared causal executor.

This is a separate action-supervision experiment, not admission of an invalid
completed-skill model. The native benchmark alone terminates press success.
"""
from pathlib import Path
from copy import deepcopy
from types import MethodType
import json
import numpy as np
from .evaluation import file_sha,digest
from .button_intent_library_v71 import ButtonIntentLibrary,TASKS
from .button_intent_candidates_v72 import ButtonIntentCandidateGenerator
from .button_intent_prefix_v72 import BoundedApproach,SCHEMA as PREFIX_SCHEMA
from .interaction import (CandidateConfig,MPCConfig,GripperTimingConfig,InteractionController,SE3InteractionMPC,_world_target,COMMAND_GOAL)
from . import factory

SCHEMA='r10_TRAIN_only_button_author_intent_causal_executor_v74'
SOURCES=('button_intent_actor_v74.py','button_intent_library_v71.py','button_intent_candidates_v72.py','button_intent_prefix_v72.py')


def binding(root,protocol,task,predictor):
    from .task_actor_backend_v69 import persistent_reference_policy_overlay
    root=Path(root).resolve()
    if task not in TASKS or predictor not in ('cv','context'):raise ValueError('exact TRAIN button geometry methods')
    model=protocol['learned_artifacts'][task]['interaction'];library=ButtonIntentLibrary(root,root/model['path'],purpose='train_development')
    settings=protocol['policy_overlay']['configurations']
    overlay=persistent_reference_policy_overlay(root,settings['feedback'],settings['persistent_reference'],settings['causal_retarget'])
    original=factory.continuation_identity(root,protocol,task,predictor,library,CandidateConfig(**protocol['candidate_config']),MPCConfig(**protocol['mpc_config']))
    value=dict(schema=SCHEMA,task=task,predictor=predictor,original_observation_collision_joint_sources=original,
        action_model=model,action_model_sha256=library.sha256,source_supervision='original_author_command_goal_not_actual_completion_TCP',
        model_support=library.metadata['independent_command_parent_support'],completed_phase_support_claimed=False,
        source_stage_guard=library.model['source_stage_guard'],public_press_completion_capability_granted=False,
        native_success_source_unchanged=True,task_deadline_seconds=10.,learned_completed_phase_clock_claimed=False,
        original_full_collision_joint_grasp_motor_gates_unchanged=True,shared_executor_overlay=overlay,
        prefix_schema=PREFIX_SCHEMA,sources={f'our_mani/multitask/{name}':file_sha(root/'our_mani/multitask'/name) for name in SOURCES},
        TRAIN_development_only=True,formal_release=False)
    return dict(value,sha256=digest(value))


def install_author_states(actor):
    """Source stage guard; never claim completed geometry/physical press."""
    c=actor.controller;m=actor.continuous_reference_manager;events=[]
    if c.goal is not None or c.feedback_step_calls:raise ValueError('install on final controller before first step')
    def phase_done(instance,state,target,*,public_geometry=False):
        goal=instance.goal
        if goal is None or goal.phase!='approach':return False
        try:
            body=instance.generator.observed_reference(state,goal.reference_id,goal.template,instance.origin)
            current=_world_target(body.pose,np.zeros(3),np.zeros(3),goal.template,1.)[0]
            return bool(np.linalg.norm(state.robot.pose.position-current.position)<=instance.library.model['source_stage_guard']['approach_to_press_distance_m'])
        except (KeyError,ValueError):return False
    def complete(instance,state):
        if instance.goal.phase!='approach':raise ValueError('this action model cannot complete a physical press skill')
        events.append(dict(time=float(state.robot.timestamp),event='original_author_public_distance_guard_enter_press_state',
            verified_completed_skill_claimed=False,native_task_success_claimed=False,task_clock_extended=False))
        instance.phase_index=1;instance.goal=instance.trajectory=instance.origin=None
        return instance._hold('author_source_stage_guard_transition_to_press')
    c._phase_done=MethodType(phase_done,c);c._complete_phase=MethodType(complete,c)
    if c._phase_done.__self__ is not c:raise ValueError('author guard installed on wrong controller')
    previous_eligible=m.retarget._eligible
    def eligible(instance,goal):return previous_eligible(goal) or (goal.phase in ('approach','press')
        and goal.template.reference_time_basis==COMMAND_GOAL and goal.template.reference_kind=='translation_only'
        and goal.template.reference_role=='button' and goal.template.reference_mode=='external')
    m.retarget._eligible=MethodType(eligible,m.retarget)
    config=deepcopy(m.approach_transit.normalized);config['version']=PREFIX_SCHEMA
    m.approach_transit=BoundedApproach(m,config)
    previous=actor.metrics_report
    def metrics(instance):return dict(previous(),button_author_intent_v74=dict(schema=SCHEMA,
        supervision=instance.controller.library.metadata,source_state_events=deepcopy(events),
        press_skill_completed_claimed=False,source_stage_guard_is_not_task_success=True,formal_release=False))
    actor.metrics_report=MethodType(metrics,actor)


def create_actor(root,env,protocol,method,descriptions,*,purpose):
    from .forecasting import ForecastConfig,MultiBodyForecastBridge
    from .perception import MultiBodySE3Tracker,RegistrationConfig
    from .training_geometry import body_templates_from_asset,declared_instance_bindings,register_collision_asset
    from .pipeline import PipelineActor
    from .public_roles import PublicRoleResolver
    from .feedback_interaction import install_feedback_controller
    from .task_reference_v37 import install_continuous_reference
    if purpose!='train_collection':raise ValueError('source-intent candidate is TRAIN-only')
    task=env.spec.task_name;predictor=method.split('_')[0];identity=binding(root,protocol,task,predictor)
    if protocol.get('continuation_identities',{}).get(task,{}).get(predictor)!=identity['sha256']:raise ValueError('freeze actual button author-intent continuation')
    pointers=protocol['learned_artifacts'][task];asset=json.loads(factory.artifact_path(root,pointers['geometry']).read_text())
    templates=body_templates_from_asset(asset,declared_instance_bindings(env,asset,shared_permission=True),
        orientation_observability=protocol['body_orientation_observability'][task])
    rigid=factory.load_rigid_layout(factory.artifact_path(root,pointers['rigid_layout']),task) if pointers.get('rigid_layout') else ()
    tracker=MultiBodySE3Tracker(templates,task_spec=env.spec,rigid_children=rigid,config=RegistrationConfig(**protocol.get('registration_config',{})))
    register_collision_asset(env.map_service,asset,orientation_observability=protocol['body_orientation_observability'][task])
    context=None
    if predictor=='context':
        from .context_se3 import MultiBodyContextDynamics
        context=MultiBodyContextDynamics(factory.artifact_path(root,protocol['context_checkpoint']))
    bridge=MultiBodyForecastBridge(tuple(t.role_id for t in templates),method=predictor,context=context,rigid_children=rigid,
        task_spec=env.spec,config=ForecastConfig(**protocol.get('forecast_config',{})))
    library=ButtonIntentLibrary(root,factory.artifact_path(root,pointers['interaction']),purpose='train_development')
    from .colour_runtime_v18 import factory_colour_scope
    with factory_colour_scope(root,protocol,task,purpose=purpose):
        from . import factory as active_factory
        # Scope updates the loader imported by the ordinary factory. Load its
        # exact typed candidate colour payload through the original resolver.
        from . import public_roles
        resolver=public_roles.PublicRoleResolver(env.spec,json.loads(factory.artifact_path(root,pointers['colour']).read_text()),shared_semantic_permission=True)
    actor=PipelineActor(env.spec,templates,tracker,bridge,library,env.robot_model,env.map_service,CandidateConfig(**protocol['candidate_config']),
        MPCConfig(**protocol['mpc_config']),predictor=predictor,ranking='geometry',descriptions=descriptions,role_resolver=resolver,
        attachment_association_distance_m=protocol['attachment_association_distance_m'],continuation_sha256=identity['sha256'],purpose=purpose,
        gripper_timing_config=GripperTimingConfig(**protocol['gripper_timing_config']),joint_limit_recovery_margin_rad=protocol['joint_limit_recovery_margin_rad'])
    old=actor.controller;generator=ButtonIntentCandidateGenerator(library,old.generator.reachability,old.generator.collision_check,old.generator.config)
    actor.controller=InteractionController(library,generator,old.mpc,predictor=predictor,ranking='geometry',continuation_sha256=identity['sha256'],
        gripper_timing_config=old.gripper_timing_config,collection_purpose='train')
    settings=identity['shared_executor_overlay']['configurations'];install_feedback_controller(actor,settings['feedback'])
    install_continuous_reference(actor,settings['persistent_reference'],retarget_config=settings['causal_retarget'])
    install_author_states(actor)
    # Retain current public trajectories for failure diagnosis; never a label.
    trace=[];previous=actor.controller.step
    def step(instance,state,forecast,remaining):
        command=previous(state,forecast,remaining)
        trace.append(dict(time=float(state.robot.timestamp),phase_index=instance.phase_index,tcp_xyzw=state.robot.pose.xyzw().tolist(),
            observed_contact=bool(state.robot.contact),observed_gripper_open=bool(state.robot.gripper_open),controller_diagnostics=deepcopy(instance.last_diagnostics)))
        return command
    actor.controller.step=MethodType(step,actor.controller);before=actor.metrics_report
    def metrics(instance):return dict(before(),executor_observed_state_trace=deepcopy(trace))
    actor.metrics_report=MethodType(metrics,actor)
    return actor,identity
