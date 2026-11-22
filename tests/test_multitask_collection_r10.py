"""CPU-only checks for causal collection boundaries and counted control."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('collection_r10', ROOT/'scripts/collect_multitask_training_r10.py')
collection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collection)
from our_mani.contracts import ObjectState, Pose, RobotState
from scipy.spatial.transform import Rotation


class TRAINOpeningTimingChecks(unittest.TestCase):
    def adapter(self):
        return collection.TRAINWaypointOpeningAdapter(dict(close_position_tolerance_m=.02,
            close_orientation_tolerance_rad=.10))

    def robot(self,position=(0.,0.,0.),angle=0.,opened=True):
        return SimpleNamespace(pose=Pose(np.asarray(position),Rotation.from_rotvec([0.,0.,angle]).as_matrix()),
                               gripper_open=opened)

    def label(self,position=(.17,0.,0.),opening=0.,phase='acquire'):
        return dict(tcp_xyzw=np.r_[position,[0.,0.,0.,1.]],opening=opening,phase=phase)

    def test_close_waits_for_position_and_orientation_then_never_retries_closed(self):
        adapter=self.adapter();label=self.label()
        actual,diagnostic=adapter.apply(label,self.robot())
        self.assertEqual(actual,1.);self.assertTrue(diagnostic['close_delayed'])
        actual,diagnostic=adapter.apply(label,self.robot((.17,0.,0.),angle=.11))
        self.assertEqual(actual,1.)
        actual,diagnostic=adapter.apply(label,self.robot((.17,0.,0.),angle=.09))
        self.assertEqual(actual,0.);self.assertTrue(diagnostic['opening_transition'])
        actual,diagnostic=adapter.apply(self.label((1.,0.,0.),phase='align'),self.robot(opened=False))
        self.assertEqual(actual,0.);self.assertFalse(diagnostic['opening_transition'])
        self.assertEqual(diagnostic['reason'],'maintain_applied_close_without_retry')

    def test_open_release_is_not_gated_against_a_predicted_or_retreat_goal(self):
        adapter=self.adapter()
        adapter.apply(self.label((0.,0.,0.)),self.robot())
        actual,diagnostic=adapter.apply(self.label((2.,0.,0.),opening=1.,phase='hang'),self.robot(opened=False))
        self.assertEqual(actual,1.);self.assertTrue(diagnostic['opening_transition'])
        self.assertEqual(diagnostic['reason'],'author_open_release_stage')

    def test_pending_close_uses_current_moving_waypoint_and_refuses_other_phase(self):
        adapter=self.adapter();adapter.apply(self.label(),self.robot())
        actual,diagnostic=adapter.apply(self.label((.4,0.,0.)),self.robot((.17,0.,0.)))
        self.assertEqual(actual,1.);self.assertAlmostEqual(diagnostic['position_error_m'],.23)
        actual,diagnostic=adapter.apply(self.label((.17,0.,0.),phase='lift'),self.robot((.17,0.,0.)))
        self.assertEqual(actual,1.);self.assertEqual(diagnostic['reason'],'close_refused_outside_acquire')

    def test_timing_configuration_is_explicit_and_finite(self):
        for bad in (None,{},dict(close_position_tolerance_m=.02),
                dict(close_position_tolerance_m=np.nan,close_orientation_tolerance_rad=.10)):
            with self.assertRaises(collection.CollectionError):collection.TRAINWaypointOpeningAdapter(bad)

    def test_collect_keeps_raw_desired_and_applied_opening_separate(self):
        env=MockEnv(terminal_after=5)
        arrays,diagnostics=collection.collect_episode(env,MockTracker(),split='train',maximum_ticks=200,
            mode_observer=collection.AttachmentModeObserver(('ball',),('ball',),.2),targets=native_targets,
            teacher=teacher_label,opening_adapter=self.adapter())
        self.assertTrue((arrays['teacher_desired_opening'][:5] == 1.).all())
        # This first pass is open; a second run requests close far from the TCP.
        def teacher():
            label=teacher_label();label['opening']=0.;return label
        env=MockEnv(terminal_after=5)
        arrays,diagnostics=collection.collect_episode(env,MockTracker(),split='train',maximum_ticks=200,
            mode_observer=collection.AttachmentModeObserver(('ball',),('ball',),.2),targets=native_targets,
            teacher=teacher,opening_adapter=self.adapter())
        self.assertTrue((arrays['teacher_desired_opening'][:5] == 0.).all())
        self.assertTrue((arrays['teacher_applied_opening'][:5] == 1.).all())
        np.testing.assert_array_equal(arrays['teacher_opening'],arrays['teacher_applied_opening'])
        self.assertTrue(arrays['teacher_opening_gate_delayed'][:5].all())
        self.assertFalse(arrays['teacher_action_valid'][-1])
        self.assertEqual(len(env.actions),5)
        self.assertTrue(all(opening == 1. for _,opening in env.actions))


class MockEnv:
    def __init__(self, terminal_after=5):
        self.ledger = SimpleNamespace(ticks=0, dt=.05)
        self.task = SimpleNamespace(t=0.)
        self.terminal_after = terminal_after
        self.actions = []

    def physics_time(self):
        return self.ledger.ticks*.05

    @property
    def native_outcome(self):
        return SimpleNamespace(terminate=self.ledger.ticks >= self.terminal_after)

    def observe(self):
        now = self.physics_time()
        robot = RobotState(now, Pose(np.zeros(3)), np.zeros(3), np.zeros(3), np.zeros(7), np.zeros(7))
        return dict(timestamp=now, **{'observation/depths': {'front': np.ones((2, 2))},
            'observation/instance_masks': {'front': np.ones((2, 2), np.int32)}}), robot

    def advance(self, pose, opening):
        if self.native_outcome.terminate:
            raise AssertionError('actuator was called after native terminal')
        self.actions.append((pose, opening))
        self.ledger.ticks += 1
        self.task.t = self.physics_time()

    def hold_tick(self):
        self.advance(Pose(np.zeros(3)), 1.)


class MockTracker:
    templates = {'ball': object()}

    def observe(self, observation, modes):
        if any('target' in key or 'teacher' in key for key in observation):
            raise AssertionError('privileged namespace leaked into sensor observation')
        now = observation['timestamp']
        pose = Pose(np.array([1.+now, 0., 0.]))
        state = ObjectState('ball', now, pose, np.ones((3, 3)), np.zeros(3), np.zeros(3))
        return [state], dict(bodies={'ball': dict(registration_valid=True, orientation_observable=False)}, rigid_groups={})


class MockPlateTracker:
    templates = {'button': object(), 'plate': object(), 'wrap': object()}

    def observe(self, observation, modes):
        now = observation['timestamp']
        states = [ObjectState(body, now, Pose(np.array([1.+now,0.,0.])), np.ones((3,3)),
                              np.zeros(3), np.zeros(3)) for body in self.templates]
        return states, dict(bodies={body:dict(registration_valid=True,orientation_observable=True)
                                  for body in self.templates},rigid_groups={})


def native_targets(ids):
    # Deliberately inconsistent poison labels must stay out of history/policy.
    return dict(positions=np.full((len(ids), 3), 666.), rotations=np.tile(np.eye(3), (len(ids), 1, 1)),
        linear_velocities=np.full((len(ids), 3), 999.), angular_velocities=np.full((len(ids), 3), 999.),
        valid=np.ones(len(ids), bool), rotation_observable=np.ones(len(ids), bool))


def teacher_label():
    return dict(tcp_xyzw=np.array([2., 0., 0., 0., 0., 0., 1.]), opening=1., stage='wp1', phase='acquire',
        manipulated_body='ball', reference_body='ball', reference_role='ball', reference_kind='translation_only', anchor_id='ball:acquire')


class CollectionChecks(unittest.TestCase):
    def run_collector(self, env, **kwargs):
        return collection.collect_episode(env, MockTracker(), maximum_ticks=200,
            mode_observer=collection.AttachmentModeObserver(('ball',), ('ball',), .2),
            targets=native_targets, **kwargs)

    def test_train_labels_do_not_change_history_and_terminal_stops_actions(self):
        env = MockEnv(terminal_after=5)
        arrays, diagnostics = self.run_collector(env, split='train', teacher=teacher_label)
        self.assertEqual(len(env.actions), 5)
        np.testing.assert_allclose(arrays['times'], np.arange(6)*.05)
        np.testing.assert_allclose(arrays['positions'][:, 0, 0], 1.+np.arange(6)*.05)
        self.assertTrue((arrays['target_positions'] == 666.).all())
        self.assertEqual(int(arrays['teacher_action_valid'].sum()), 5)
        self.assertFalse(arrays['teacher_action_valid'][-1])
        self.assertEqual(arrays['valid'].dtype, np.bool_)
        self.assertEqual(arrays['body_ids'].tolist(), ['ball'])
        self.assertEqual(len(diagnostics), 6)

    def test_val_teacher_rejected_and_frozen_policy_receives_no_targets(self):
        with self.assertRaises(collection.CollectionError):
            self.run_collector(MockEnv(), split='validation', teacher=teacher_label)

        class FrozenPolicy:
            calls = 0

            def act(self, *, states, robot, diagnostics, remaining_seconds):
                self.calls += 1
                self.assertions = (states[0].pose.position[0] < 2.,
                    'target_positions' not in diagnostics, remaining_seconds <= 10.)
                if not all(self.assertions):
                    raise AssertionError('native label leaked into frozen causal policy')
                return robot.pose, 1.

        policy = FrozenPolicy()
        arrays, _ = self.run_collector(MockEnv(5), split='validation', policy=policy)
        self.assertEqual(policy.calls, 5)
        self.assertFalse(arrays['teacher_action_valid'].any())
        self.assertTrue((arrays['teacher_stage'] == '').all())

    def test_teacher_hidden_clock_advance_rejected(self):
        env = MockEnv()

        def corrupt_teacher():
            env.advance(Pose(np.zeros(3)), 1.)
            return teacher_label()

        with self.assertRaisesRegex(collection.CollectionError, 'advanced the counted clock'):
            self.run_collector(env, split='train', teacher=corrupt_teacher)

    def test_two_hundred_tick_cap_and_hold_validation(self):
        env = MockEnv(terminal_after=1000)
        arrays, _ = self.run_collector(env, split='validation')
        self.assertEqual(env.ledger.ticks, 200)
        self.assertEqual(len(arrays['times']), 201)
        self.assertEqual(arrays['times'][-1], 10.)

    def test_public_press_masks_plate_history_and_targets_preserves_native_GT(self):
        env = MockEnv(terminal_after=5)

        def teacher():
            label = teacher_label()
            label.update(phase='approach' if env.ledger.ticks < 2 else 'press',
                manipulated_body='button', reference_body='button', reference_role='button',
                reference_kind='observed_se3')
            return label

        arrays, _ = collection.collect_episode(env, MockPlateTracker(), split='train',maximum_ticks=200,
            mode_observer=collection.AttachmentModeObserver(('button','plate','wrap'),(),.2),
            targets=native_targets,teacher=teacher)
        self.assertEqual(arrays['body_ids'].tolist(), ['button','plate','wrap'])
        self.assertTrue(arrays['valid'][:2,1].all())
        self.assertFalse(arrays['valid'][2:,1].any())
        self.assertFalse(arrays['target_valid'][2:,1].any())
        self.assertTrue(arrays['native_pose_valid'].all())
        self.assertTrue(arrays['valid'][:,[0,2]].all())
        self.assertTrue((arrays['modes'][:,1] == 0).all())

    def test_VAL_plate_requires_public_command_and_press_latches_after_move(self):
        env = MockEnv(5)

        class Policy:
            def act(self, *, states, robot, diagnostics, remaining_seconds):
                return robot.pose, 1., {'kind':'press' if env.ledger.ticks == 2 else 'move'}

        arrays, _ = collection.collect_episode(env,MockPlateTracker(),split='validation',maximum_ticks=200,
            mode_observer=collection.AttachmentModeObserver(('button','plate','wrap'),(),.2),
            targets=native_targets,policy=Policy())
        self.assertTrue(arrays['valid'][:2,1].all())
        self.assertFalse(arrays['valid'][2:,1].any())
        self.assertEqual(arrays['public_command_kind'][3], 'move')
        self.assertTrue(arrays['native_pose_valid'].all())

        class Unspecified:
            def act(self, *, states, robot, diagnostics, remaining_seconds):
                return robot.pose, 1.

        with self.assertRaisesRegex(collection.CollectionError,'explicitly log public'):
            collection.collect_episode(MockEnv(5),MockPlateTracker(),split='validation',maximum_ticks=200,
                mode_observer=collection.AttachmentModeObserver(('button','plate','wrap'),(),.2),
                targets=native_targets,policy=Unspecified())

    def test_genuine_immediate_terminal_remains_data_without_extra_teacher_or_padding(self):
        env=MockEnv(terminal_after=0)
        calls=[]

        def teacher():
            calls.append(True)
            return teacher_label()

        arrays,_=self.run_collector(env,split='train',teacher=teacher)
        self.assertEqual(len(arrays['times']),1)
        self.assertFalse(calls)
        self.assertFalse(env.actions)
        self.assertFalse(arrays['teacher_action_valid'].any())
        self.assertEqual(arrays['public_command_kind'][0],'no_action_terminal')
        eligibility=collection.context_window_eligibility(arrays)
        self.assertEqual(eligibility['usable_context_windows'],0)
        self.assertFalse(eligibility['genuine_short_terminal_is_infrastructure_failure'])
        self.assertFalse(eligibility['automatic_resampling'])

    def test_context_eligibility_distinguishes_short_vs_full_horizon(self):
        short,_=self.run_collector(MockEnv(5),split='validation')
        short_quality=collection.context_window_eligibility(short)
        self.assertEqual(short_quality['usable_context_windows'],1)
        self.assertEqual(short_quality['full_horizon_windows'],0)
        long,_=self.run_collector(MockEnv(26),split='validation')
        long_quality=collection.context_window_eligibility(long)
        self.assertEqual(long_quality['full_horizon_windows'],1)

    def test_r09_gate_refuses_partial_and_protocol_failure(self):
        base = collection.BASE/'cpu_collection_checks'
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as folder:
            path = Path(folder)/'final_report.json'
            with self.assertRaises(collection.CollectionError):
                collection.require_r09_release(path)
            report = dict(complete=True, total_test_runs=400, expected_test_runs=400,
                          protocol_acceptance_passed=True, acceptance_errors=[])
            for mutation in ({'complete': False}, {'total_test_runs': 399},
                             {'protocol_acceptance_passed': False}, {'acceptance_errors': ['clock']}):
                path.write_text(json.dumps(dict(report, **mutation)))
                with self.assertRaises(collection.CollectionError):
                    collection.require_r09_release(path)
            # Failed development performance is disclosed, never confused
            # with the separate completed hard-protocol native release gate.
            path.write_text(json.dumps(dict(report, development_performance_screen_passed=False)))
            accepted = collection.require_r09_release(path)
            self.assertFalse(accepted['development_performance_screen_passed'])
            self.assertFalse(accepted['benefit_validated_by_this_gate'])

    def test_ambiguous_contact_masks_modes_without_native_id(self):
        observer = collection.AttachmentModeObserver(('a', 'b'), ('a', 'b'), .2)
        robot = RobotState(0., Pose(np.zeros(3)), np.zeros(3), np.zeros(3), np.zeros(7), np.zeros(7), contact=True)
        states = [ObjectState(name, 0., Pose(np.zeros(3)), np.ones((3, 3)), np.zeros(3), np.zeros(3)) for name in ('a', 'b')]
        modes, invalid = observer.observe(states, robot)
        self.assertEqual(invalid, {'a', 'b'})
        self.assertNotIn('attached', modes.values())
        modes, invalid = observer.observe(states[:1], robot)
        self.assertEqual(modes['a'], 'attached')
        self.assertFalse(invalid)

    def test_peg_teacher_pillar_is_not_variation_index(self):
        env = SimpleNamespace(spec=SimpleNamespace(family='insert_onto_rotating_peg'), collection_variation=1,
            task=SimpleNamespace(var2target_state_list={1: {'target_spoke': 'pillar2'}}))
        self.assertEqual(collection.teacher_reference_labels(env)['selected_reference'], 'peg2')

    def test_teacher_references_keep_physical_ids_and_generalize_semantic_roles(self):
        from our_mani.multitask.specs import get_task_spec
        for phase in ('align', 'insert', 'release'):
            labels = [collection.reference_for_phase(get_task_spec('insert_onto_rotating_peg'), phase,
                {'manipulated_body': 'ring', 'selected_reference': 'peg' + str(i)}) for i in (0, 2)]
            self.assertEqual([label['reference_body'] for label in labels], ['peg0', 'peg2'])
            self.assertTrue(all(label['reference_role'] == 'selected_peg' for label in labels))
            self.assertEqual(labels[0]['anchor_id'], labels[1]['anchor_id'])
        labels = [collection.reference_for_phase(get_task_spec('place_cups_on_rotating_frame'), 'hang',
            {'manipulated_body': 'cup0', 'selected_reference': 'spoke' + str(i)}) for i in (0, 2)]
        self.assertTrue(all(label['reference_role'] == 'selected_spoke' for label in labels))
        self.assertEqual(labels[0]['anchor_id'], labels[1]['anchor_id'])
        for phase in ('approach', 'acquire', 'detach', 'transport', 'place'):
            labels = [collection.reference_for_phase(get_task_spec('remove_cups_from_rotating_frame'), phase,
                {'manipulated_body': 'cup' + str(i), 'selected_reference': 'spoke' + str(i)}) for i in (0, 2)]
            self.assertEqual([label['manipulated_body'] for label in labels], ['cup0', 'cup2'])
            self.assertTrue(all(label['manipulated_role'] == 'selected_cup' for label in labels))
            self.assertEqual(labels[0]['anchor_id'], labels[1]['anchor_id'])
            expected = 'selected_cup' if phase in ('approach', 'acquire') else 'selected_spoke' if phase == 'detach' else 'world'
            self.assertTrue(all(label['reference_role'] == expected for label in labels))

    def test_native_teacher_stage_bookkeeping_without_paths(self):
        from our_mani.multitask.specs import get_task_spec
        env = MockEnv()
        env.spec = get_task_spec('moving_basketball_in_hoop')
        env.development_split = 'train'
        env._native_terminal_latched = False
        env.task._bool_expert = False
        env.task.stage = 'wp0'

        def get_expert_info(task, *, bool_return_path):
            self.assertFalse(bool_return_path)
            self.assertFalse(task._bool_expert)
            return dict(trajectory=np.array([[[2.,0.,0.,0.,0.,0.,1.,1.]]]), stage='wp1')

        with patch.object(collection.importlib, 'import_module', return_value=SimpleNamespace(get_expert_info=get_expert_info)):
            teacher = collection.native_teacher(env)
            label = teacher()
            self.assertEqual(env.task.stage, 'wp1')
            self.assertEqual(label['phase'], 'acquire')
            self.assertEqual(env.ledger.ticks, 0)
            env._native_terminal_latched = True
            with self.assertRaises(collection.CollectionError):
                teacher()

    def test_rigid_layout_requires_train_and_rejects_moving_child(self):
        t = 6
        arrays = dict(times=np.arange(t)*.05, body_ids=np.asarray(['frame', 'spoke0', 'spoke1']),
            target_positions=np.zeros((t, 3, 3)), target_rotations=np.tile(np.eye(3), (t, 3, 1, 1)),
            target_valid=np.ones((t, 3), bool))
        arrays['target_positions'][:, 1, 0] = .1
        arrays['target_positions'][:, 2, 0] = np.arange(t)*.1
        artifact = collection.fit_rigid_layout(arrays, task='place_cups_on_rotating_frame', source_group='train:task:source')
        self.assertEqual([row['child'] for row in artifact['accepted']], ['spoke0'])
        self.assertEqual([row['child'] for row in artifact['rejected']], ['spoke1'])
        with self.assertRaises(collection.CollectionError):
            collection.fit_rigid_layout(arrays, task='task', source_group='validation:task:source')

    def test_cpu_plan_distinguishes_original_sources_replicas_and_empty_merge(self):
        from our_mani.multitask.specs import get_task_spec
        base = collection.BASE/'cpu_collection_checks'
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as folder:
            folder = Path(folder)
            task = 'push_moving_button'
            roles = {role.name: 'full' for role in get_task_spec(task).roles if role.kind == 'body'}
            protocol = dict(dt=.05, simulation_seconds=10., gripper_velocity=.2, current_instance_masks=True,
                information_setting='CPU mock protocol; no native run', shared_semantic_instance_binding=True,
                semantic_binding_available_to_baseline=True, shared_TRAIN_CAD_geometry=True,
                attachment_association_distance_m=.2, body_orientation_observability={task: roles})
            config = folder/'protocol.json'
            config.write_text(json.dumps(protocol))
            plan_path = folder/'plan.json'
            plan = collection.build_plan(collection.BASE/'demonstrations/manifest.json', config, plan_path,
                                         tasks=[task], replicas=2)
            self.assertEqual(len(plan['cases']), 200)
            self.assertEqual(len({c['parent_source_group'] for c in plan['cases']}), 100)
            self.assertEqual(len({c['case_id'] for c in plan['cases']}), 200)
            self.assertEqual(len({c['raw_episode_id'] for c in plan['cases']}), 200)
            self.assertFalse(plan['augmentation_budget']['native_started'])
            self.assertEqual(plan['original_demonstration_budget']['train_sources'], 1400)
            manifest = collection.merge_completed(plan_path, folder/'output')
            self.assertEqual(manifest['completed'], 0)
            self.assertEqual(manifest['expected'], 200)
            self.assertFalse(manifest['complete'])
            self.assertEqual(manifest['cache_episode_count'], 0)
            self.assertEqual(manifest['source_episode_count'], 0)

    def test_interaction_adapter_and_dynamics_loader_keep_target_namespaces(self):
        from our_mani.multitask.track_dataset import read_tracks
        base = collection.BASE/'cpu_collection_checks'
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as folder:
            folder = Path(folder)
            arrays, _ = self.run_collector(MockEnv(5), split='train', teacher=teacher_label)
            cache = folder/'body_tracks.npz'
            np.savez_compressed(cache, **arrays)
            task = 'moving_basketball_in_hoop'
            entry = dict(split='train', task=task, source_id=task+':episode1__native_aug_r0',
                raw_episode_id='episode1__native_aug_r0', parent_raw_episode_id='episode1',
                parent_source_group='train:'+task+':episode1', augmentation_id='episode1__native_aug_r0',
                source_kind='TRAIN_native_generated_augmentation',
                cache=str(cache.relative_to(ROOT)), cache_sha256=collection.sha(cache),
                observation_source='causal_RGBD_and_proprioception', target_source='TRAIN_VAL_native_supervision')
            receipt = folder/'case.json'
            receipt.write_text(json.dumps(dict(entry=entry)))
            source = collection.interaction_source_from_case(receipt)
            self.assertEqual(source.grouping_source, 'train:'+task+':episode1')
            self.assertEqual(source.source_kind, 'TRAIN_native_generated_augmentation')
            self.assertEqual(source.references[0].reference_kind, 'translation_only')
            self.assertEqual(source.references[0].manipulated_role, 'ball')
            self.assertTrue((source.body_tracks['ball'].poses_xyzw[:, :3] == 666.).all())
            manifest = folder/'manifest.json'
            manifest.write_text(json.dumps(dict(version='r10_causal_body_tracks_v1', entries=[entry])))
            _, episodes = read_tracks(ROOT, manifest)
            self.assertEqual(episodes[0].source_key, ('train', task, 'episode1'))
            self.assertTrue((episodes[0].arrays['positions'][:, 0, 0] < 2.).all())
            self.assertTrue((episodes[0].arrays['target_positions'] == 666.).all())
            entry['split'] = 'validation'
            receipt.write_text(json.dumps(dict(entry=entry)))
            with self.assertRaises(collection.CollectionError):
                collection.interaction_source_from_case(receipt)


if __name__ == '__main__':
    unittest.main()
