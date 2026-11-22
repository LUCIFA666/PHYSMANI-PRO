"""One real selection log per labelled attempt; synthetic CPU values only."""
from copy import deepcopy
import unittest

import numpy as np

from our_mani.multitask.candidate_success import ContinuationAttempts,FEATURE_NAMES,CONTINUATION_POLICY
from our_mani.multitask.trace_evidence import audit_selection_usage


def record(method='cv_value',two=False):
    predictor,ranking=method.split('_');value=ranking=='value'
    ledger=ContinuationAttempts(task_name='push_moving_button',predictor=predictor,
        continuation_sha256='frozen',library_sha256='library')
    ledger.begin('approach','selected' if value else 'geometry',0.,np.zeros(len(FEATURE_NAMES)),
        probability=.7 if value else None,selected_supported=True if value else None)
    events=[dict(decision_id=0,time=0.,phase='approach',candidate_count=2,candidate_ids=['geometry','selected'],
        selected_index=1 if value else 0,geometry_index=0,selected_candidate_id='selected' if value else 'geometry',
        geometry_candidate_id='geometry',actual_override=value,episode_first_actionable_decision=True,
        value_query_this_selection=value,first_selection_policy='value_once' if value else 'geometry',
        continuation_selection_policy=CONTINUATION_POLICY,probability_is_unpenalized=True,
        all_downstream_selections_geometry=True,unexecuted_candidates_labelled=False,
        success_probability=.7 if value else None,selected_supported=True if value else None,
        calibrated_ensemble_std=.1 if value else None,fallback_reason=None)]
    if two:
        ledger.skill_completed('approach',.1);ledger.begin('press','geometry',.2,np.zeros(len(FEATURE_NAMES)))
        event=deepcopy(events[0]);event.update(decision_id=1,time=.2,phase='press',selected_index=0,
            selected_candidate_id='geometry',actual_override=False,episode_first_actionable_decision=False,
            value_query_this_selection=False,success_probability=None,selected_supported=None,calibrated_ensemble_std=None)
        events.append(event)
    attempts=ledger.finish(False,.4,'native_budget_failure')
    return dict(official_task_success=False,counted_clock=dict(seconds=.4),continuation_sha256='frozen',
        interaction_library_sha256='library',continuation_attempts=attempts,
        actor_metrics=dict(candidate_selection_events=events))


class TraceEvidenceTests(unittest.TestCase):
    def test_single_value_query_full_native_label_and_actual_override_count(self):
        result=audit_selection_usage({('push_moving_button','episode0'):{'cv_value':record(two=True)}})
        data=result['per_task_method']['push_moving_button|cv_value']
        self.assertEqual(data['candidate_attempts'],2);self.assertEqual(data['value_queries'],1)
        self.assertEqual(data['actual_override_selections'],1)
        self.assertAlmostEqual(data['selected_first_decision_probability_metrics']['Brier'],.49)
        self.assertFalse(result['success_module_benefit_established'])
        self.assertFalse(result['zero_override_proves_full_trajectory_equality'])

    def test_missing_selection_cannot_hide_unknown_override_status(self):
        data=record();data['actor_metrics']['candidate_selection_events']=[]
        with self.assertRaisesRegex(ValueError,'selection/attempt trace'):
            audit_selection_usage({('push_moving_button','episode0'):{'cv_value':data}})

    def test_second_value_query_or_penalized_probability_is_rejected(self):
        data=record(two=True);data['actor_metrics']['candidate_selection_events'][1]['value_query_this_selection']=True
        with self.assertRaisesRegex(ValueError,'only once'):
            audit_selection_usage({('push_moving_button','episode0'):{'cv_value':data}})
        data=record();data['actor_metrics']['candidate_selection_events'][0]['success_probability']=.6
        with self.assertRaisesRegex(ValueError,'probability/support differs'):
            audit_selection_usage({('push_moving_button','episode0'):{'cv_value':data}})

    def test_geometry_and_zero_candidate_episodes_need_no_fake_probability(self):
        data=record('context_geometry')
        result=audit_selection_usage({('push_moving_button','episode0'):{'context_geometry':data}})
        row=result['per_task_method']['push_moving_button|context_geometry']
        self.assertEqual(row['value_queries'],0);self.assertIsNone(row['selected_first_decision_probability_metrics'])
        data['continuation_attempts']=[];data['actor_metrics']['candidate_selection_events']=[]
        result=audit_selection_usage({('push_moving_button','episode0'):{'context_geometry':data}})
        self.assertEqual(result['per_task_method']['push_moving_button|context_geometry']['zero_candidate_episodes'],1)


if __name__=='__main__':unittest.main()
