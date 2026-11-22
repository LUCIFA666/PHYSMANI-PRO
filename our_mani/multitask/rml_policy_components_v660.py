"""Explicit current source-bound RML frontend/intent/executor routes for core14."""
from pathlib import Path
import json
from .specs import TASK_SPECS
from .evaluation import file_sha

SOURCES=tuple('our_mani/multitask/'+n for n in (
    'rml_component_domain_v640.py','button_command_library_v641.py','colour_component_domain_v642.py',
    'rml_component_factory_v643.py','rml_policy_components_v660.py',
    'physmani_geometry_policy_v518.py','physmani_geometry_goals_v529.py','physmani_geometry_goals_v593.py',
    'physmani_skill_policy_v548.py','physmani_skill_executor_v534.py','physmani_public_rrt_v610.py',
    'physmani_ring_frontend_v570.py','physmani_buzz_frontend_v586.py','physmani_ring_compensation_v580.py',
    'physmani_ring_release_v590.py','physmani_terminal_opening_v620.py','physmani_buzz_curve_v602.py',
    'physmani_remove_stages_v611.py','physmani_remove_wp0_v637.py','physmani_author_stages_v574.py',
    'physmani_stage_lookahead_v594.py','physmani_high_bin_delay_v601.py','physmani_bin_filter_progress_v631.py',
    'physmani_basketball_release_v558.py','physmani_bin_clearance_v650.py',
    'physmani_remove_source_roles_v653.py','physmani_ring_release_contact_v657.py','placement_clearance_v50.py'))


def binding(root,config,task):
    if task not in TASK_SPECS:raise ValueError('core14 registered task required')
    from .physmani_terminal_opening_v620 import binding as terminal
    from .physmani_public_rrt_v610 import binding as rrt
    family=TASK_SPECS[task].family;rules={};native=config['baseline_repo'];root=Path(root)
    def require(field,expected,*,by_task=True):
        actual=config.get(field,{}).get(task) if by_task else config.get(field)
        if actual!=expected:raise ValueError('freeze actual RML component binding: '+field)
        rules[field]=expected
    if family=='insert_onto_rotating_peg':
        from .physmani_ring_compensation_v580 import binding as compensation
        from .physmani_ring_release_v590 import binding as release
        from .physmani_ring_frontend_v570 import binding as gauge
        require('public_ring_compensation_v580',compensation(root,native,task))
        require('author_ring_terminal_release_v590',release(root,native,task))
        require('public_ring_frontend_v570',gauge(root,config,task))
        require('terminal_opening_v620',terminal(root),by_task=False)
        from .physmani_ring_release_contact_v657 import binding as contact
        require('ring_release_contact_v657',contact(root,native,task))
    elif family in ('put_rubbish_in_moving_bin','place_cups_on_rotating_frame'):
        from .physmani_author_stages_v574 import binding as stages
        from .physmani_stage_lookahead_v594 import binding as delays
        require('original_author_stage_guards_v574',stages(root,native,task))
        require('author_stage_lookahead_v594',delays(root,native,task))
        if family=='put_rubbish_in_moving_bin':
            from .physmani_bin_filter_progress_v631 import binding as progress
            require('bin_filter_resolution_v631',progress(root),by_task=False)
            from .physmani_bin_clearance_v650 import binding as clearance
            require('bin_public_clearance_v650',clearance(root,config,task))
            if task.endswith('_high_speed'):
                from .physmani_high_bin_delay_v601 import binding as high
                require('high_bin_delay_v601',high(root,native,task))
    elif family=='moving_basketball_in_hoop':
        from .physmani_basketball_release_v558 import binding as ball
        require('author_basketball_release_v558',ball(root,native,task))
    elif family=='beat_the_rotating_buzz':
        from .physmani_buzz_curve_v602 import binding as curve
        from .physmani_buzz_frontend_v586 import binding as gaze
        require('public_Buzz_curve_v602',curve(root,native,task))
        require('public_buzz_frontend_v586',gaze(root,config,task))
        require('terminal_opening_v620',terminal(root),by_task=False)
        require('public_approach_RRT_v610',rrt(root),by_task=False)
    elif family=='remove_cups_from_rotating_frame':
        from .physmani_remove_stages_v611 import binding as remove
        from .physmani_remove_wp0_v637 import binding as wp0
        require('public_remove_author_v611',remove(root,native,task))
        require('remove_wp0_command_v637',wp0(root,native,task))
        from .physmani_remove_source_roles_v653 import binding as roles
        require('remove_source_roles_v653',roles(root,config,task))
        require('terminal_opening_v620',terminal(root),by_task=False)
        require('public_approach_RRT_v610',rrt(root),by_task=False)
    elif family!='push_moving_button':raise ValueError('no implicit or generic fallback RML task route')
    return dict(schema='r10_direct_core14_source_bound_RML_policy_components_v660',task=task,rules=rules,
        original_RML_parameters_and_native_outcome_unchanged=True,
        sources_sha256={name:file_sha(root/name) for name in SOURCES})


def construct(root,config,task,actor):
    binding(root,config,task)
    from .physmani_geometry_policy_v518 import PublicModelFrontend,ButtonGeometryIntents
    from .physmani_geometry_executor_v517 import PhysManiGeometryExecutor
    from .physmani_skill_executor_v534 import PhysManiSkillExecutor
    family=TASK_SPECS[task].family;root=Path(root)
    frontend_class=PublicModelFrontend;executor_class=PhysManiSkillExecutor
    if family=='insert_onto_rotating_peg':
        from .physmani_ring_frontend_v570 import PublicRingFrontend
        frontend_class=PublicRingFrontend
    elif family=='beat_the_rotating_buzz':
        from .physmani_buzz_frontend_v586 import PublicBuzzFrontend
        from .physmani_public_rrt_v610 import PublicRRTSkillExecutor
        frontend_class=PublicBuzzFrontend;executor_class=PublicRRTSkillExecutor
    elif family=='remove_cups_from_rotating_frame':
        from .physmani_public_rrt_v610 import PublicRRTSkillExecutor
        executor_class=PublicRRTSkillExecutor
    frontend=frontend_class(actor)
    if family=='push_moving_button':
        policy=ButtonGeometryIntents(frontend);executor_class=PhysManiGeometryExecutor
    elif family=='insert_onto_rotating_peg':
        from .physmani_ring_release_contact_v657 import SourceContactRingIntents
        rule=config['public_ring_compensation_v580'][task]
        policy=SourceContactRingIntents(frontend,rule,json.loads((root/rule['prior']['path']).read_text()),
            config['author_ring_terminal_release_v590'][task],contact_rule=config['ring_release_contact_v657'][task])
    elif family=='moving_basketball_in_hoop':
        from .physmani_basketball_release_v558 import AuthorBasketballReleaseIntents
        policy=AuthorBasketballReleaseIntents(frontend,config['author_basketball_release_v558'][task])
    elif family=='place_cups_on_rotating_frame':
        from .physmani_stage_lookahead_v594 import AuthorLookaheadIntents
        policy=AuthorLookaheadIntents(frontend,config['original_author_stage_guards_v574'][task],
            config['author_stage_lookahead_v594'][task])
    elif family=='put_rubbish_in_moving_bin':
        from .physmani_bin_clearance_v650 import NormalBinClearanceIntents,HighBinClearanceIntents
        args=(frontend,config['original_author_stage_guards_v574'][task],config['author_stage_lookahead_v594'][task])
        kwargs=dict(root=root,config=config,clearance_rule=config['bin_public_clearance_v650'][task])
        policy=(HighBinClearanceIntents(*args,config['high_bin_delay_v601'][task],**kwargs) if task.endswith('_high_speed')
                else NormalBinClearanceIntents(*args,**kwargs))
    elif family=='beat_the_rotating_buzz':
        from .physmani_terminal_opening_v620 import BuzzCurveIntents
        policy=BuzzCurveIntents(frontend,config['public_Buzz_curve_v602'][task])
    elif family=='remove_cups_from_rotating_frame':
        from .physmani_remove_source_roles_v653 import AuthorLiveRoleRemoveIntents
        rule=config['remove_wp0_command_v637'][task]
        policy=AuthorLiveRoleRemoveIntents(frontend,config['public_remove_author_v611'][task],rule,
            json.loads((root/rule['model']['path']).read_text()),source_role_rule=config['remove_source_roles_v653'][task])
    else:raise ValueError('missing explicit policy route')
    callbacks=dict(observe_public_frame=frontend.observe,admit_pose_action=policy.admit_pose,admit_joint_target=policy.admit_joint)
    executor=(executor_class(config,**callbacks) if family=='push_moving_button' else
              executor_class(config,skill_policy=policy,**callbacks))
    return frontend,policy,executor
