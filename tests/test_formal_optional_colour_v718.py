"""Real frozen-pointer admission tests; no simulator, GPU or TEST data reads."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from our_mani.multitask import formal_suite_admission_v718 as module
from our_mani.multitask import formal_suite_admission_v532 as original
from our_mani.multitask.common_rml_protocol_v530 import (
    AUTHORIZATION, LOCAL_SOURCES, NATIVE_SOURCES, PARAMETERS, fingerprint,
)
from our_mani.multitask.evaluation import file_sha
from our_mani.multitask.evaluation_roster16_v283 import TASKS, METHODS
from our_mani.multitask.joint16_tracks_v31 import BLOCKS
from our_mani.multitask.protocol import MultitaskProtocolError
from our_mani.multitask.three_method_release_v531 import policy_fingerprint


class FrozenSuite:
    """A complete TRAIN-only 1600-parent/192-run/48-route frozen fixture."""

    def __init__(self, root, current_colour='null', accepted_colour='missing',
                 required_problem=None, invalid_colour=False, model_change=False):
        self.root = root
        self.files = {}
        self.profile = dict(
            schema='r10_common_author_RML_profile_v530', methods=list(METHODS),
            parameters=dict(PARAMETERS), user_authorization=AUTHORIZATION,
            original_waypoint_filter_and_RML_algorithm=True,
            motion_metrics_required_for_all_methods=True,
            measured_acceleration6_is_diagnostic_only=True,
            native_task_success_unchanged=True,
            sources_sha256={p: 'a' * 64 for p in LOCAL_SOURCES},
            native_sources_sha256={p: 'b' * 64 for p in NATIVE_SOURCES})
        self.profile_sha = fingerprint(self.profile)
        checkpoint = self.write('checkpoint.bin', b'fresh joint16 model')
        geometry = self.write('geometry.bin', b'geometry')
        interaction = self.write('interaction.bin', b'interaction')
        layout = self.write('layout.bin', b'layout')
        colour = self.write('colour.bin', b'colour')
        if invalid_colour:
            colour = dict(colour, sha256='f' * 64)
        current = dict(geometry=geometry, interaction=interaction, rigid_layout=layout)
        accepted = deepcopy(current)
        if current_colour != 'missing':
            current['colour'] = None if current_colour == 'null' else colour
        if accepted_colour != 'missing':
            accepted['colour'] = None if accepted_colour == 'null' else colour
        if required_problem:
            key, value = required_problem
            if value == 'missing':
                current.pop(key)
                accepted.pop(key)
            else:
                current[key] = accepted[key] = value
        if model_change:
            current['geometry'] = self.write('changed_geometry.bin', b'changed')
        worker = self.write('worker.py', b'# frozen worker\n')
        consumer = self.write('consumer.py', b'# frozen native executor\n')
        input_lock = self.write('inputs.json', {'files': {}})
        config_pointers = {}
        for task in TASKS:
            config_pointers[task] = self.write('configs/' + task + '.json', dict(
                context_checkpoint=checkpoint, learned_artifacts={task: current}))
        routes = []
        for task in TASKS:
            for method in METHODS:
                routes.append(dict(task=task, method=method,
                    GPU_required=method == 'physmani', configuration=config_pointers[task],
                    frozen_inputs=input_lock, worker=worker, consumer=consumer,
                    common_RML_profile_sha256=self.profile_sha))
        self.routes = dict(routes=routes, common_author_RML_profile=self.profile,
                           shared_context_checkpoint_sha256=checkpoint['sha256'])
        self.training = dict(
            schema='r10_joint16_completed_training_independent_acceptance_v1',
            training_complete=True, TEST_read=False,
            Value_NGFF_FASTER_opened_or_started=False, checkpoint=checkpoint,
            training=dict(TRAIN_original_parents=1600, completed_epochs=30,
                exact_roster_passed=True, tasks=list(TASKS),
                buzz_parents_with_conditional_rotation=200),
            operation_artifacts=[dict(task=task,
                block_uses_original_geometry_kernel=task in BLOCKS,
                existing_TRAIN_operation_artifacts=accepted) for task in TASKS])
        self.manifest = dict(split='train', episodes=[
            dict(task=task, raw_episode_id=str(index), path='TRAIN/' + task + '/' + str(index))
            for task in TASKS for index in range(100)])
        case = self.write('case.json', {'complete': True})
        auditor = self.write('auditor.py', b'# frozen independent readiness auditor\n')
        attestation = dict(schema='r10_observed_common_author_RML_parameters_v530',
            profile_sha256=self.profile_sha, actual_parameters=dict(PARAMETERS),
            complete=True, native_parameters_mutated=False,
            actual_motion_bound_claimed=False)
        self.readiness = dict(
            schema='r10_exact16_fixed_four_common_author_RML_readiness_v532',
            policy_sha256=policy_fingerprint(self.routes),
            all16_fixed_four_runtime_ready=True, completed=192, method_case_count=192,
            source_count=64, task_ready_count=16, motion_is_diagnostic_only=True,
            common_RML_profile_sha256=self.profile_sha, TEST_read=False,
            source_sha256=auditor['sha256'],
            cases=[dict(task=task, raw=str(index), method=method,
                status='completed', native_success=False, motion_pass=False,
                common_author_RML_parameters=deepcopy(attestation), case=case)
                for task in TASKS for index in range(4) for method in METHODS],
            groups=[dict(task=task, method=method) for task in TASKS for method in METHODS])
        self.suite = dict(methods=list(METHODS), tasks=list(TASKS), readiness_auditor=auditor)
        self.refresh()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        pointer = dict(path=name, sha256=file_sha(path))
        self.files[str(path)] = pointer['sha256']
        return pointer

    def refresh(self):
        for key, name, value in (
            ('routes', 'routes.json', self.routes),
            ('training_acceptance', 'training.json', self.training),
            ('TRAIN_manifest', 'TRAIN_manifest.json', self.manifest),
            ('readiness', 'readiness.json', self.readiness)):
            self.suite[key] = self.write(name, value)

    def admit(self):
        return module.admit(self.root, self.suite, self.files)


class OptionalColourTests(unittest.TestCase):
    def fixture(self, **kwargs):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return FrozenSuite(Path(temporary.name), **kwargs)

    def test_absent_and_null_colour_are_the_same_without_changing_inputs(self):
        for current in ('missing', 'null'):
            for accepted in ('missing', 'null'):
                with self.subTest(current=current, accepted=accepted):
                    fixture = self.fixture(current_colour=current, accepted_colour=accepted)
                    before = {name: file_sha(name) for name in fixture.files}
                    # The implementation must not invoke the defective original admission.
                    with patch.object(original, 'admit', side_effect=AssertionError('old admission')):
                        result = fixture.admit()
                    self.assertTrue(result['formal_VAL_permitted'])
                    self.assertFalse(result['formal_TEST_permitted'])
                    self.assertEqual(result['fixed_TRAIN_method_cases'], 192)
                    self.assertEqual(before, {name: file_sha(name) for name in fixture.files})

    def test_present_colour_requires_equal_accepted_pointer_and_actual_frozen_hash(self):
        self.assertTrue(self.fixture(current_colour='pointer', accepted_colour='pointer').admit()['complete'])
        for current, accepted in (('pointer', 'null'), ('missing', 'pointer')):
            with self.subTest(current=current, accepted=accepted):
                with self.assertRaisesRegex(ValueError, 'operation model changed'):
                    self.fixture(current_colour=current, accepted_colour=accepted).admit()
        with self.assertRaisesRegex(ValueError, 'frozen SHA'):
            self.fixture(current_colour='pointer', accepted_colour='pointer', invalid_colour=True).admit()

    def test_required_models_cannot_be_absent_null_or_malformed(self):
        for key in ('geometry', 'interaction', 'rigid_layout'):
            for value in ('missing', None, {'path': 'geometry.bin'}):
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        self.fixture(required_problem=(key, value)).admit()

    def test_required_model_change_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, 'operation model changed'):
            self.fixture(model_change=True).admit()

    def test_training_and_fixed_four_scientific_requirements_are_unchanged(self):
        mutations = (
            ('training parents', lambda f: f.training['training'].update(TRAIN_original_parents=1599)),
            ('epochs', lambda f: f.training['training'].update(completed_epochs=29)),
            ('TRAIN rotation', lambda f: f.training['training'].update(buzz_parents_with_conditional_rotation=199)),
            ('TRAIN TEST use', lambda f: f.training.update(TEST_read=True)),
            ('forbidden model', lambda f: f.training.update(Value_NGFF_FASTER_opened_or_started=True)),
            ('192 cases', lambda f: f.readiness.update(completed=191)),
            ('64 sources', lambda f: f.readiness.update(source_count=63)),
            ('fixed original source', lambda f: f.readiness['cases'][0].update(raw='4')),
            ('48 groups', lambda f: f.readiness['groups'].pop()),
            ('attestation', lambda f: f.readiness['cases'][0]['common_author_RML_parameters']['actual_parameters'].update(max_acceleration=6.)),
            ('full TRAIN roster', lambda f: f.manifest['episodes'].pop()),
        )
        for label, mutate in mutations:
            with self.subTest(requirement=label):
                fixture = self.fixture()
                mutate(fixture)
                fixture.refresh()
                with self.assertRaises((ValueError, MultitaskProtocolError)):
                    fixture.admit()

    def test_exact_three_methods_guard_precedes_any_artifact_read(self):
        with patch.object(module, 'locked_json') as reader:
            with self.assertRaisesRegex(ValueError, 'three authorized'):
                module.admit('.', dict(methods=list(METHODS) + ['cv_value'], tasks=list(TASKS)), {})
            reader.assert_not_called()

    def test_TEST_release_gate_runs_before_any_TEST_pointer_or_data_access(self):
        fixture = self.fixture()
        # No source_manifests or TEST files exist in the fixture at all.
        with self.assertRaisesRegex(ValueError, 'VAL release'):
            module.read_formal_source(fixture.root, fixture.suite, fixture.files, 'test')

    def test_frozen_model_bytes_still_cannot_change(self):
        fixture = self.fixture()
        (fixture.root / 'geometry.bin').write_bytes(b'changed after freeze')
        with self.assertRaisesRegex(MultitaskProtocolError, 'frozen input SHA changed'):
            fixture.admit()


if __name__ == '__main__':
    unittest.main()
