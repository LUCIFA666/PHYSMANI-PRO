from pathlib import Path
from copy import deepcopy
import json,unittest
from our_mani.multitask.normal_cup_primary_export_v105 import admit_templates
from our_mani.multitask.interaction import InteractionFitConfig

ROOT=Path(__file__).resolve().parents[1];B=ROOT/'artifacts/framework/multitask_r10'


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.draft=json.loads((B/'command_goal_actual700_full_curve_CPU_draft_v1/place_cups_on_rotating_frame/real_full_curve_fit_report.json').read_text())
        full=json.loads((B/'TRAIN_artifacts_full_goalv6_semanticv9_3100_v1/report.json').read_text())
        cls.production=full['per_task']['place_cups_on_rotating_frame']['completed_skill_fitting']
        cfg=json.loads((B/'command_goal_reference_rule_v6_counted_final_setup_v1/artifact_fit_configuration_full3100.json').read_text())
        cls.config=InteractionFitConfig(**cfg['interaction'])

    def test_actual_independent_fits_admit_all_six_phases_and_preserve_failed_recovery(self):
        templates,scope=admit_templates(self.draft,self.production,self.config)
        self.assertEqual([t.phase for t in templates],['approach','acquire','lift','transport','align','hang'])
        self.assertEqual(len(scope['unadmitted_partitions']),2)
        self.assertFalse(scope['original_all_observed_partitions_gate_passed'])
        self.assertTrue(scope['admitted_partition_support_and_error_gates_passed'])
        self.assertTrue(all(t.reference_mode!='released' for t in templates))
        self.assertEqual(self.draft['phase_fits'][scope['admitted_partition_keys'][-1]]['completed_parent_count'],94)

    def test_mandatory_partition_with_less_than_eight_parents_still_fails(self):
        d=deepcopy(self.draft);p=deepcopy(self.production);key=next(k for k in d['phase_fits'] if k.startswith('hang|'))
        d['phase_fits'][key]['completed_parent_count']=p['phase_fits'][key]['completed_parent_count']=7
        with self.assertRaisesRegex(ValueError,'mandatory initial-sequence'):admit_templates(d,p,self.config)

    def test_independent_numeric_disagreement_is_not_hidden_by_export(self):
        p=deepcopy(self.production);key=next(iter(p['phase_fits']))
        p['phase_fits'][key]['position_tolerance_m']+=1e-4
        with self.assertRaisesRegex(ValueError,'numeric fit differs'):admit_templates(self.draft,p,self.config)

    def test_even_supported_released_recovery_is_unconditionally_outside_declared_domain(self):
        d=deepcopy(self.draft);p=deepcopy(self.production)
        for key in d['phase_fits']:
            if '|released|' in key:
                d['phase_fits'][key]['passed']=p['phase_fits'][key]['passed']=True
                d['phase_fits'][key]['completed_parent_count']=p['phase_fits'][key]['completed_parent_count']=100
        templates,scope=admit_templates(d,p,self.config)
        self.assertEqual(len(templates),6);self.assertEqual(len(scope['unadmitted_partitions']),2)

    def test_original_support_threshold_cannot_be_lowered(self):
        c=deepcopy(self.config)
        object.__setattr__(c,'minimum_completed_parent_sources',7)
        with self.assertRaisesRegex(ValueError,'original8'):admit_templates(self.draft,self.production,c)


if __name__=='__main__':unittest.main()
