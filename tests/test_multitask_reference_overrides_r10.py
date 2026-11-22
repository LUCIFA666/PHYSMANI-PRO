"""CPU audited-reference permission boundaries; original evidence is read-only."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from our_mani.multitask import reference_overrides as override
from our_mani.multitask.specs import get_task_spec
from tests.test_multitask_collection_r10 import collection,MockEnv
ROOT=Path(__file__).resolve().parents[1]


class ReferenceOverrideChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol=json.loads((collection.BASE/'full_training_config_v7_DRAFT.json').read_text())

    def test_all_six_tasks_twelve_phases_have_600_real_TRAIN_and_author_SHA_evidence(self):
        mapping,inputs=override.validate_overrides(self.protocol,root=ROOT)
        self.assertEqual(len(mapping),6);self.assertEqual(sum(map(len,mapping.values())),12)
        self.assertEqual(sum(Path(name).suffix=='.npz' for name in inputs),600)
        for task in mapping:
            source=Path(self.protocol['baseline_repo'])/get_task_spec(task).native_source_relative_path
            self.assertEqual(inputs[str(source)],get_task_spec(task).native_source_sha256)

    def test_override_changes_only_reference_kind_not_role_body_anchor_or_manipulated_role(self):
        for task,phases in override.ALLOWED.items():
            spec=get_task_spec(task)
            selected=dict(manipulated_body=spec.primary_manipulated_roles[0],selected_reference='')
            for phase in phases:
                old=collection.reference_for_phase(spec,phase,selected)
                new=collection.reference_for_phase(spec,phase,selected,protocol=self.protocol)
                old['reference_kind']='translation_only';self.assertEqual(old,new)

    def test_existing_ball_grasp_lift_and_rotating_cup_buzz_bindings_are_unmodified(self):
        for task in ('moving_basketball_in_hoop','moving_basketball_in_hoop_high_speed',
            'insert_onto_rotating_peg','place_cups_on_rotating_frame','remove_cups_from_rotating_frame','beat_the_rotating_buzz'):
            spec=get_task_spec(task);selected=dict(manipulated_body=spec.primary_manipulated_roles[0],selected_reference='peg1' if 'peg' in task else 'spoke1')
            for phase in spec.phase_skills:
                if phase.name in override.ALLOWED.get(task,{}):continue
                self.assertEqual(collection.reference_for_phase(spec,phase.name,selected),
                    collection.reference_for_phase(spec,phase.name,selected,protocol=self.protocol))

    def test_unsupported_conditional_peg_phase_and_reference_mutations_are_rejected(self):
        for task,phase,row in (
            ('insert_onto_rotating_peg','approach',dict(reference_role='ring',reference_kind='translation_only')),
            ('moving_basketball_in_hoop','lift',dict(reference_role='phase_origin',reference_kind='translation_only')),
            ('moving_basketball_in_hoop','deposit',dict(reference_role='ball',reference_kind='translation_only')),
            ('push_moving_button','press',dict(reference_role='button',reference_kind='observed_se3')),
            ('push_moving_button','press',dict(reference_role='button',reference_kind='translation_only',manipulated_role='hidden'))):
            protocol=copy.deepcopy(self.protocol);protocol['TRAIN_reference_kind_overrides'].setdefault(task,{})[phase]=row
            with self.assertRaises(override.ReferenceOverrideError):override.structural_overrides(protocol)

    def test_missing_or_changed_evidence_and_orphan_evidence_rejected(self):
        bad=copy.deepcopy(self.protocol);bad.pop('TRAIN_reference_kind_override_evidence')
        with self.assertRaises(override.ReferenceOverrideError):override.validate_overrides(bad,root=ROOT)
        bad=copy.deepcopy(self.protocol);bad['TRAIN_reference_kind_override_evidence']['sha256']='0'*64
        with self.assertRaises(override.ReferenceOverrideError):override.validate_overrides(bad,root=ROOT)
        bad=copy.deepcopy(self.protocol);bad.pop('TRAIN_reference_kind_overrides')
        with self.assertRaises(override.ReferenceOverrideError):override.validate_overrides(bad,root=ROOT)

    def test_changed_author_source_hash_is_not_accepted(self):
        original=override.sha
        def altered(path):return '0'*64 if Path(path).name=='push_moving_button.py' else original(path)
        with patch.object(override,'sha',side_effect=altered):
            with self.assertRaisesRegex(override.ReferenceOverrideError,'source changed'):
                override.validate_overrides(self.protocol,root=ROOT)

    def test_changed_original_TRAIN_cache_hash_is_not_accepted(self):
        original=override.sha
        def altered(path):return '0'*64 if Path(path).suffix=='.npz' else original(path)
        with patch.object(override,'sha',side_effect=altered):
            with self.assertRaisesRegex(override.ReferenceOverrideError,'TRAIN audit input changed'):
                override.validate_overrides(self.protocol,root=ROOT)

    def test_native_teacher_consumes_explicit_protocol_and_VAL_teacher_stays_forbidden(self):
        env=MockEnv();env.spec=get_task_spec('moving_basketball_in_hoop');env.development_split='train'
        env.protocol=self.protocol;env._native_terminal_latched=False;env.task._bool_expert=False
        def expert(task,*,bool_return_path):
            self.assertFalse(bool_return_path)
            return dict(stage='wp3',trajectory=np.array([[[0.,0.,0.,0.,0.,0.,1.,1.]]]))
        with patch.object(collection.importlib,'import_module',return_value=SimpleNamespace(get_expert_info=expert)):
            label=collection.native_teacher(env)()
        self.assertEqual(label['reference_kind'],'translation_only');self.assertEqual(label['reference_role'],'hoop')
        self.assertEqual(env.ledger.ticks,0)
        with self.assertRaises(collection.CollectionError):collection.native_teacher(SimpleNamespace(development_split='validation'))

if __name__=='__main__':unittest.main()
