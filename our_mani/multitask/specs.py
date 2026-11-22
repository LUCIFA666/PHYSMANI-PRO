"""R10 task contracts, separated from native state and policy observations.

Native attribute/scene bindings are for reset, map partitioning and metrics.
They are not semantic segmentation or pose observations available to a policy.
Phase/stage associations describe TRAIN demonstrations, never live expert calls.
"""
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class ResetDispatch(str, Enum):
    TRANSLATION = 'author_translation'
    ROTATING = 'author_rotating'
    PEG_NO_RANDOM = 'author_peg_no_random'


@dataclass(frozen=True)
class NativeRole:
    name: str
    attribute_path: tuple = ()
    scene_name: str | None = None
    kind: str = 'body'
    required: bool = True


@dataclass(frozen=True)
class PhaseSkill:
    name: str
    TRAIN_expert_stages: tuple[str, ...]
    skill: str
    optional_after_native_completion: bool = False


@dataclass(frozen=True)
class TargetLabelSemantics:
    role: str
    native_debug_expression: str
    identity_rule: str
    dropped_roles: tuple[str, ...] = ()
    identity_may_change_within_stage: bool = False
    complete_body_state: bool = False
    sequence_rule: str = (
        'TRAIN labels only; segment at episode, body identity, phase and '
        'attachment/contact mode boundaries; never concatenate unknown identities.'
    )


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    family: str
    speed: str
    roles: tuple[NativeRole, ...]
    primary_manipulated_roles: tuple[str, ...]
    dynamic_roots: tuple[str, ...]
    graspable_roles: tuple[str, ...]
    audit_only_goal_roles: tuple[str, ...]
    native_success_all: tuple[str, ...]
    native_fatal_conditions: tuple[str, ...]
    reset_dispatch: ResetDispatch
    supported_variations: tuple[int, ...]
    phase_skills: tuple[PhaseSkill, ...]
    target_label: TargetLabelSemantics
    native_source_sha256: str
    preload_before_reset_to_demo: bool = False
    init_episode_kwargs: tuple[tuple[str, bool], ...] = ()
    metric_state_fields: tuple[str, ...] = ()
    reset_notes: tuple[str, ...] = ()

    @property
    def variation_count(self):
        return len(self.supported_variations)

    @property
    def native_source_relative_path(self):
        return 'third_party/RLBench/rlbench/tasks/' + self.task_name + '.py'

    def policy_contract(self):
        """Public task/skill names only; deliberately omit all native bindings."""
        return {
            'task_name': self.task_name,
            'family': self.family,
            'speed': self.speed,
            'skills': [phase.skill for phase in self.phase_skills],
            'required_observations': 'current/past RGB-D, declared masks and robot proprioception',
            'native_roles_are_policy_observations': False,
        }


SOURCE_SHA256 = MappingProxyType({
    'beat_the_rotating_buzz': 'b6f115a9b4c01c03c0dfb031ba1c3e8a8992b169af1be4d1dbde1316f3a057a1',
    'beat_the_rotating_buzz_high_speed': '74cbbe15c4cf13a0cc62e129179ee7b0fd6980f427b1f68cd04c0b2e465e0c48',
    'insert_onto_rotating_peg': '4ce4af367ca2a0f75de16e9804aab7c1d4ca36bafb9502cab820425313c00ddd',
    'insert_onto_rotating_peg_high_speed': '86c8775251f5f4b09e97158540e6789b98a04e4087b34b14cf22c55a59a05b95',
    'moving_basketball_in_hoop': 'a1ac3a53a5f4a148b11cbcb950710c838cf14395d1aaa9b3ba26f9e1c9b9383f',
    'moving_basketball_in_hoop_high_speed': '9a615d20eb5419ec01f47b42553b809e568b9ddd05b43885e5ca695df9260ddb',
    'place_cups_on_rotating_frame': 'ecd7eab437cbae4954550f089c65e7dbca54eaa4612d81f11e388b0abf3eed85',
    'place_cups_on_rotating_frame_high_speed': '8e8f8f53e9ad0f2704a7e6239eb83f58ce994ce393e27c8e935ffebd60b87a94',
    'push_moving_button': '45a3ac38a04b95549a9246e67ed4951276dcb189289e7d5321fa013f3211e766',
    'push_moving_button_high_speed': '602d8687d130f2486f3107d93fd8d2ca155d5154a6375c8869cda1ff9622f5b5',
    'put_rubbish_in_moving_bin': '5518be87f1763ed008f1af615ea2868563c42a64b457d4d4cfd3c877b73fe414',
    'put_rubbish_in_moving_bin_high_speed': '00ab9e38e1197ba343024e6842d5cb9a16835f69428c7aca797e8c065b75caaf',
    'remove_cups_from_rotating_frame': 'f89e93738b9e78d9229d20ca0c3242df39e4f600881617ed3091809d56220490',
    'remove_cups_from_rotating_frame_high_speed': '2c2457f6ab6e7d432828b067de4309dbfbf3f4877afcefb3294d542891cbe6c7',
})

NATIVE_COMMON_SOURCE_SHA256 = MappingProxyType({
    'third_party/RLBench/rlbench/backend/conditions.py': '73669d9811783cb2f86b91ae5ab48c6dabce5f46a16078ae506ca4f890ad17d2',
    'third_party/RLBench/rlbench/backend/task.py': 'c7b0e98564a6595d73e758b975d8a3d7f8ccae27a87cc542594f0c8ab37a61ca',
    'packages/physmani_bench/physmani_bench/rlbench_env.py': 'da0b20f4327ac4ff1510851204dd8f9f0faf2f93f7bad868f0e728ab0ad272e3',
})


def _body(name, attr=None, scene=None, kind='body'):
    path = tuple(attr) if isinstance(attr, tuple) else ((attr,) if attr else ())
    return NativeRole(name, path, scene, kind)


def _phases(*rows):
    return tuple(PhaseSkill(row[0], tuple(row[1]), row[2], row[3] if len(row) == 4 else False)
                 for row in rows)


def _build_registry():
    families = {
        'beat_the_rotating_buzz': dict(
            roles=(_body('wand', 'wand', 'wand'), _body('frame', '_frame_base', 'Cuboid'),
                   _body('right_goal', scene='right_sensor', kind='sensor'),
                   _body('forbidden_region', scene='middle_sensor', kind='sensor')),
            primary_manipulated_roles=('wand',), dynamic_roots=('frame', 'wand'),
            graspable_roles=('wand',), audit_only_goal_roles=('right_goal', 'forbidden_region'),
            native_success_all=('wand detected by right_sensor', 'nothing grasped'),
            native_fatal_conditions=('wand detected by middle_sensor: native terminal failure',),
            reset_dispatch=ResetDispatch.ROTATING, supported_variations=(0,),
            phase_skills=_phases(('approach', ('wp0',), 'approach_wand'),
                                 ('acquire', ('wp1',), 'intercept_grasp_wand'),
                                 ('follow', ('wp2', 'wp3_'), 'SE3_gap_constrained_follow'),
                                 ('release', ('wp3',), 'right_end_release_verify')),
            target_label=TargetLabelSemantics('wand', 'task.wand.get_position()',
                                             'fixed wand identity; scripted/attached modes differ'),
            metric_state_fields=('yaw_speed',),
            reset_notes=('frame rotates about local Y, unlike the other rotating families',)),
        'insert_onto_rotating_peg': dict(
            roles=(_body('ring', '_square_ring', 'square_ring'),
                   _body('frame', '_frame_base', 'square_base'),
                   *(_body('peg' + str(i), scene='pillar' + str(i)) for i in range(3)),
                   *(_body('goal_corner' + str(i), scene='success_detector' + str(i), kind='sensor') for i in range(4)),
                   _body('goal_centre', '_success_centre', 'success_centre', 'marker')),
            primary_manipulated_roles=('ring',), dynamic_roots=('frame', 'ring'),
            graspable_roles=('ring',), audit_only_goal_roles=tuple('goal_corner' + str(i) for i in range(4)) + ('goal_centre',),
            native_success_all=('ring simultaneously detected by all four success detectors',),
            native_fatal_conditions=(), reset_dispatch=ResetDispatch.PEG_NO_RANDOM,
            supported_variations=(0, 1, 2), init_episode_kwargs=(('bool_random_place', False),),
            phase_skills=_phases(('approach', ('wp0', 'wp1'), 'approach_ring'),
                                 ('acquire', ('wp1-grasp',), 'grasp_ring'),
                                 ('lift', ('wp2',), 'lift_ring'),
                                 ('align', ('wp3',), 'align_ring_hole_to_rotating_peg'),
                                 ('insert', ('wp4',), 'SE3_axis_constrained_insert'),
                                 ('release', ('wp4-open',), 'release_if_skill_requires', True)),
            target_label=TargetLabelSemantics('ring', 'task._square_ring.get_position()',
                                             'fixed ring; label does not describe rotating peg/frame'),
            metric_state_fields=('yaw_speed',),
            reset_notes=('default init_episode randomizes archived state; bool_random_place=False is mandatory',
                         'native success does not explicitly require release')),
        'moving_basketball_in_hoop': dict(
            roles=(_body('ball', 'ball', 'ball'),
                   _body('hoop', 'hoop', 'basket_ball_hoop_respondable'),
                   _body('goal', scene='success', kind='sensor')),
            primary_manipulated_roles=('ball',), dynamic_roots=('ball', 'hoop'),
            graspable_roles=('ball',), audit_only_goal_roles=('goal',),
            native_success_all=('ball detected by success sensor', 'nothing grasped',
                                'both gripper fingers open amount > .9', 'has_been_picked latch true'),
            native_fatal_conditions=(), reset_dispatch=ResetDispatch.TRANSLATION,
            supported_variations=(0, 1),
            phase_skills=_phases(('approach', ('wp0',), 'approach_moving_ball'),
                                 ('acquire', ('wp1',), 'intercept_grasp_ball'),
                                 ('lift', ('wp2',), 'lift_ball'),
                                 ('deposit', ('wp3',), 'intercept_moving_hoop_release_verify')),
            target_label=TargetLabelSemantics('ball', '[task.ball.get_position(), task.hoop.get_position()][0]',
                                             'fixed ball; held/free mode changes after first grasp', ('hoop',)),
            metric_state_fields=('_has_been_picked',),
            reset_notes=('two independent translation drivers; after first grasp ball does not resume scripted motion',)),
        'place_cups_on_rotating_frame': dict(
            roles=(*(_body('cup' + str(i), ('_cups', i), 'mug' + str(i)) for i in range(3)),
                   _body('frame', '_frame_base', 'place_cups_holder_base'),
                   *(_body('spoke' + str(i), ('_spokes', i), 'place_cups_holder_spoke' + str(i)) for i in range(3)),
                   *(_body('goal' + str(i), scene='success_detector' + str(i), kind='sensor') for i in range(3))),
            primary_manipulated_roles=('cup0',), dynamic_roots=('frame', 'cup0', 'cup1', 'cup2'),
            graspable_roles=('cup0', 'cup1', 'cup2'), audit_only_goal_roles=('goal0', 'goal1', 'goal2'),
            native_success_all=('nothing grasped', 'variation0: cup0 detected by any peg success sensor'),
            native_fatal_conditions=(), reset_dispatch=ResetDispatch.ROTATING, supported_variations=(0,),
            phase_skills=_phases(('approach', ('wp0',), 'approach_cup'),
                                 ('acquire', ('wp1',), 'grasp_cup'),
                                 ('lift', ('wp2',), 'lift_cup'),
                                 ('transport', ('wp3',), 'carry_cup'),
                                 ('align', ('wp4',), 'align_handle_to_rotating_spoke'),
                                 ('hang', ('wp5',), 'hang_release_verify')),
            target_label=TargetLabelSemantics('selected_cup', 'task._cups[task._cups_placed].get_position()',
                                             'expert-selected cup index is not stored in HDF5; resolve from TRAIN tracking',
                                             identity_may_change_within_stage=True),
            metric_state_fields=('yaw_speed', '_cups_placed', '_index'),
            reset_notes=('only variation0/one required cup is exposed by this native task',)),
        'push_moving_button': dict(
            roles=(_body('button', 'target_button', 'push_button_target'),
                   _body('plate', 'target_topPlate', 'target_button_topPlate'),
                   _body('wrap', 'target_wrap', 'target_button_wrap'),
                   _body('press_joint', 'joint', 'target_button_joint', 'joint')),
            primary_manipulated_roles=('button', 'plate'), dynamic_roots=('button',),
            graspable_roles=(), audit_only_goal_roles=('press_joint',),
            native_success_all=('press joint absolute displacement from original position > .003m',
                                'TCP z descended > .03m across last ten native task steps, simultaneously'),
            native_fatal_conditions=(), reset_dispatch=ResetDispatch.TRANSLATION,
            supported_variations=(0, 1),
            phase_skills=_phases(('approach', ('wp0',), 'approach_button'),
                                 ('press', ('wp1',), 'surface_velocity_matched_downward_press')),
            target_label=TargetLabelSemantics('button', 'task.target_button.get_position()',
                                             'fixed moving button body; not plate/joint deformation state'),
            metric_state_fields=('_tip_z_history', 'botton_init_position', '_variation_index', 'variation_index'),
            reset_notes=('green visual feedback branch is unreachable with current JointCondition tuple semantics',)),
        'put_rubbish_in_moving_bin': dict(
            roles=(_body('rubbish', 'rubbish', 'rubbish'), _body('bin', 'bin', 'bin'),
                   _body('tomato0', scene='tomato1'), _body('tomato1', scene='tomato2'),
                   _body('goal', scene='success', kind='sensor')),
            primary_manipulated_roles=('rubbish',), dynamic_roots=('rubbish', 'bin'),
            graspable_roles=('rubbish',), audit_only_goal_roles=('goal',),
            native_success_all=('rubbish detected by success sensor; no explicit empty-gripper requirement',),
            native_fatal_conditions=(), reset_dispatch=ResetDispatch.TRANSLATION,
            supported_variations=(0, 1),
            phase_skills=_phases(('approach', ('wp0',), 'identify_approach_rubbish'),
                                 ('acquire', ('wp1',), 'grasp_rubbish'),
                                 ('lift', ('wp2',), 'lift_rubbish'),
                                 ('deposit', ('wp3',), 'intercept_moving_bin_release_verify')),
            target_label=TargetLabelSemantics('bin', 'task.bin.get_position()',
                                             'fixed moving bin label, not the manipulated rubbish'),
            reset_notes=('rubbish and tomato placements are permuted in archived native reset',)),
        'remove_cups_from_rotating_frame': dict(
            roles=(*(_body('cup' + str(i), ('cups', i), 'mug' + str(i)) for i in range(3)),
                   _body('frame', '_frame_base', 'place_cups_holder_base'),
                   *(_body('spoke' + str(i), ('spokes', i), 'place_cups_holder_spoke' + str(i)) for i in range(3)),
                   *(_body('goal' + str(i), ('success_detectors', i), 'success_detector' + str(i), 'sensor') for i in range(3))),
            primary_manipulated_roles=('cup0', 'cup1', 'cup2'),
            dynamic_roots=('frame', 'cup0', 'cup1', 'cup2'), graspable_roles=('cup0', 'cup1', 'cup2'),
            audit_only_goal_roles=('goal0', 'goal1', 'goal2'),
            native_success_all=('nothing grasped', 'any cup detected by any table success sensor'),
            native_fatal_conditions=(), reset_dispatch=ResetDispatch.ROTATING, supported_variations=(0,),
            phase_skills=_phases(('approach', ('wp0', 'wp1'), 'intercept_rotating_cup'),
                                 ('acquire', ('wp2',), 'grasp_cup'),
                                 ('detach', ('wp3',), 'axis_constrained_unhook'),
                                 ('transport', ('wp4',), 'carry_cup_to_table'),
                                 ('place', ('wp5',), 'place_release_verify')),
            target_label=TargetLabelSemantics('selected_cup', 'nearest task.cups to current TCP',
                                             'nearest-cup identity is not recorded and can switch without a stage change',
                                             identity_may_change_within_stage=True),
            metric_state_fields=('yaw_speed', 'cups_removed', 'cups_to_remove'),
            reset_notes=('normal native init_episode stages arm joints; high-speed version omits this staging',
                         'only variation0 supported; do not apply the moving-pick home reset')),
    }
    result = {}
    for family, values in families.items():
        for high in (False, True):
            task = family + ('_high_speed' if high else '')
            result[task] = TaskSpec(task, family, 'high' if high else 'normal',
                                    native_source_sha256=SOURCE_SHA256[task],
                                    preload_before_reset_to_demo=(family == 'push_moving_button' and high),
                                    **values)
    return MappingProxyType(result)


TASK_SPECS = _build_registry()


def get_task_spec(task_name):
    try:
        return TASK_SPECS[task_name]
    except KeyError:
        raise ValueError('No prepared R10 TaskSpec for ' + str(task_name)) from None
