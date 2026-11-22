"""Post-execution critic/selection evidence; labels never enter policy input."""
from collections import Counter,defaultdict

import numpy as np

from .candidate_success import CONTINUATION_POLICY,FEATURE_NAMES,FEATURE_SCHEMA_SHA256,LABEL_SCOPE
from .candidate_training import metrics


def require(value,message):
    if not value:raise ValueError(message)


def audit_selection_usage(groups):
    """Require a one-to-one actual selection/attempt ledger, including fallbacks."""
    summary={};probabilities=defaultdict(list);raw=defaultdict(list)
    for (task,source),records in groups.items():
        for method,record in records.items():
            if method=='physmani':
                require(record.get('continuation_attempts',[])==[],'baseline cannot fabricate our critic attempts')
                continue
            predictor,ranking=method.split('_');attempts=record.get('continuation_attempts')
            events=record.get('actor_metrics',{}).get('candidate_selection_events')
            require(isinstance(attempts,list) and isinstance(events,list) and len(attempts)==len(events),
                'complete actual selection/attempt trace required: '+task+'|'+method+'|'+source)
            key=task+'|'+method
            row=raw.setdefault(key,dict(episodes=0,candidate_attempts=0,zero_candidate_episodes=0,
                value_queries=0,actual_override_selections=0,override_sources=set(),fallbacks=Counter()))
            row['episodes']+=1;row['candidate_attempts']+=len(attempts);row['zero_candidate_episodes']+=not attempts
            for index,(attempt,event) in enumerate(zip(attempts,events)):
                ids=event.get('candidate_ids',[]);selected=event.get('selected_index');geometry=event.get('geometry_index')
                require(event.get('decision_id')==attempt.get('decision_id')==index
                    and event.get('candidate_count')==len(ids)>0 and type(selected) is int and type(geometry) is int
                    and 0<=selected<len(ids) and 0<=geometry<len(ids)
                    and event.get('selected_candidate_id')==ids[selected]==attempt.get('candidate_id')
                    and event.get('geometry_candidate_id')==ids[geometry]
                    and event.get('actual_override') is (selected!=geometry)
                    and event.get('episode_first_actionable_decision') is (index==0)
                    and attempt.get('episode_first_actionable_decision') is (index==0),
                    'selection indices/IDs/override or attempt identity differs')
                query=ranking=='value' and index==0
                require(event.get('value_query_this_selection') is query
                    and event.get('first_selection_policy')==('value_once' if ranking=='value' else 'geometry')
                    and event.get('continuation_selection_policy')==CONTINUATION_POLICY
                    and attempt.get('continuation_selection_policy')==CONTINUATION_POLICY
                    and event.get('probability_is_unpenalized') is True
                    and event.get('all_downstream_selections_geometry') is True
                    and event.get('unexecuted_candidates_labelled') is False,
                    'formal critic is queried only once and cannot change downstream geometry')
                require(attempt.get('task')==task and attempt.get('predictor')==predictor
                    and attempt.get('feature_schema_sha256')==FEATURE_SCHEMA_SHA256
                    and attempt.get('label_scope')==LABEL_SCOPE
                    and attempt.get('continuation_sha256')==record.get('continuation_sha256')
                    and attempt.get('interaction_library_sha256')==record.get('interaction_library_sha256')
                    and attempt.get('complete') is True and attempt.get('censored') is False
                    and type(attempt.get('official_task_success')) is bool,
                    'continuation trace has incomplete or mismatched full-native-outcome labels')
                x=np.asarray(attempt.get('features'),float)
                require(x.shape==(len(FEATURE_NAMES),) and np.isfinite(x).all()
                    and event.get('phase')==attempt.get('phase')
                    and np.isfinite(event.get('time',np.nan))
                    and abs(event['time']-attempt.get('selected_at',np.nan))<=1e-8
                    # observed_until advances when a skill, abandonment or the
                    # native episode outcome is recorded; selected_at is the
                    # immutable cutoff for the candidate feature vector.
                    and np.isfinite(attempt.get('observed_until',np.nan))
                    and event['time']<=attempt['observed_until']<=record['counted_clock']['seconds']+1e-8
                    and 0<=event['time']<=record['counted_clock']['seconds']+1e-8,
                    'selection feature/time evidence is not finite and causal')
                abandoned=attempt.get('skill_reason')=='candidate_abandoned_before_skill_completion'
                require(attempt['official_task_success']==record['official_task_success'] or
                    (abandoned and attempt['official_task_success'] is False),
                    'skill success cannot substitute for the complete frozen native continuation')
                p=event.get('success_probability');supported=event.get('selected_supported');std=event.get('calibrated_ensemble_std')
                require(p==attempt.get('predicted_probability') and supported==attempt.get('selected_supported'),
                    'stored probability/support differs from the actually selected attempt')
                if query:
                    require(isinstance(p,(int,float)) and np.isfinite(p) and 0<=p<=1 and type(supported) is bool
                        and isinstance(std,(int,float)) and np.isfinite(std) and std>=0,
                        'first value selection must record real finite unpenalized probability/support/std')
                    probabilities[key].append((int(attempt['official_task_success']),float(p),supported))
                else:
                    require(p is None and supported is None and std is None and selected==geometry,
                        'geometry/downstream selection cannot introduce a learned probability or override')
                row['value_queries']+=query;row['actual_override_selections']+=selected!=geometry
                if selected!=geometry:row['override_sources'].add(source)
                if event.get('fallback_reason') is not None:row['fallbacks'][event['fallback_reason']]+=1
    for key,row in raw.items():
        values=probabilities[key];supported=[item for item in values if item[2]]
        calculate=lambda items:metrics(np.asarray([i[0] for i in items]),np.asarray([i[1] for i in items])) if items else None
        summary[key]=dict(episodes=row['episodes'],candidate_attempts=row['candidate_attempts'],
            zero_candidate_episodes=row['zero_candidate_episodes'],value_queries=row['value_queries'],
            actual_override_selections=row['actual_override_selections'],override_sources=sorted(row['override_sources']),
            fallback_reasons=dict(row['fallbacks']),selected_first_decision_probability_metrics=calculate(values),
            supported_first_decision_probability_metrics=calculate(supported),selection_trace_complete=True)
    return dict(complete=True,diagnostics_valid=True,per_task_method=summary,
        unmatched_attempts=0,unknown_override_status_selections=0,
        unexecuted_candidate_labels_fabricated=False,
        probability_population='actually selected first value decisions, native completion with same-phase abandonment rule',
        zero_override_proves_full_trajectory_equality=False,success_module_benefit_established=False)
