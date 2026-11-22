"""Conservative learned candidate ranking, with unchanged r06 attempt continuation."""
from collections import Counter
from dataclasses import replace
import numpy as np
from ..contracts import InterceptGoal
from ..risk_v2.selector import (candidate_features,attempt_controller_hash,
    AttemptInterception as OriginalAttemptInterception)


class AttemptSelector:
    def __init__(self,owner,original,model=None,exploration='geometry',seed=0):
        self.owner,self.reachability=owner,original.reachability
        self.relative_velocity_options=original.relative_velocity_options
        self.object_position_bounds=original.object_position_bounds
        self.latency_reserve=original.latency_reserve
        self.model,self.exploration=model,exploration
        self.rng=np.random.default_rng(seed)
        self.next_goal_id=0;self.last_diagnostics={}

    def select(self,forecast,regions,robot,now,*,min_horizon=.15,stride=2):
        candidates=[];rejected=Counter()
        for region in regions:
            if region.object_id not in forecast.object_ids:
                rejected['region_target_missing']+=1;continue
            for arrival in forecast.times[1::stride]:
                duration=arrival-now-self.latency_reserve
                if duration<min_horizon:
                    rejected['insufficient_remaining_horizon']+=1;continue
                obj_pose,obj_v,obj_w=forecast.at(region.object_id,arrival)
                bounds=self.object_position_bounds.get(region.object_id)
                if bounds is not None and (np.any(obj_pose.position<bounds[0]) or np.any(obj_pose.position>bounds[1])):
                    rejected['forecast_outside_declared_target_workspace']+=1;continue
                pose,surface_v=region.in_world(obj_pose,obj_v,obj_w)
                for relative in self.relative_velocity_options:
                    if np.linalg.norm(relative)>region.max_relative_speed:
                        rejected['relative_speed_outside_region']+=1;continue
                    velocity=surface_v+pose.rotation@relative
                    check=self.reachability.check(robot,pose,velocity,obj_w,duration)
                    if not check.feasible:
                        rejected[check.reason]+=1;continue
                    effort=float(np.linalg.norm(check.joints-robot.joints))
                    geometry_score=-.10*(arrival-now)-.02*effort
                    feature=candidate_features(forecast,region,robot,self.owner.goal_pose,arrival,pose,velocity,check,
                        self.owner.config,self.owner.predictor)
                    candidates.append(dict(arrival=float(arrival),region=region,pose=pose,velocity=velocity,
                        angular_velocity=obj_w,check=check,geometry_score=geometry_score,features=feature))
        self.last_diagnostics=dict(feasible_candidates=len(candidates),rejections=dict(rejected))
        if not candidates:return None
        geometry=np.array([c['geometry_score'] for c in candidates])
        probabilities=None;fallback=None;propensity=None
        if self.model is not None:
            predictions=self.model.predict(np.stack([c['features'] for c in candidates]),self.owner.config['_episode_task'])
            supported=predictions['supported']
            probabilities=predictions['probability']
            geometry_selected=int(np.argmax(geometry))
            selected=geometry_selected
            std=np.asarray(predictions['ensemble_std'],dtype=float)
            ranking_enabled=bool(predictions.get('ranking_enabled',False))
            proposed=None;gain=None;required=None
            if not ranking_enabled:
                fallback='TRAIN_cross_validation_does_not_support_learned_ranking'
            elif not supported.any():
                fallback='all_candidates_outside_TRAIN_support_geometry_fallback'
            elif not supported[geometry_selected]:
                fallback='geometry_reference_outside_TRAIN_support'
            else:
                proposed=int(np.argmax(np.where(supported,probabilities,-np.inf)+1e-7*geometry))
                gain=float(probabilities[proposed]-probabilities[geometry_selected])
                required=float(self.owner.config['ranking_minimum_gain']+
                    self.owner.config['ranking_disagreement_multiplier']*
                    np.hypot(std[proposed],std[geometry_selected]))
                if proposed==geometry_selected:
                    fallback='learned_and_geometry_agree'
                elif gain>required:
                    selected=proposed
                else:
                    fallback='gain_below_pre_registered_disagreement_margin'
            self.last_diagnostics.update(supported_candidates=int(supported.sum()),fallback=fallback,
                learned_ranking_enabled=ranking_enabled,geometry_candidate=geometry_selected,
                proposed_candidate=proposed,selected_candidate=selected,estimated_gain=gain,
                required_gain=required,selected_bootstrap_std=float(std[selected]),
                uncertainty_interpretation='bootstrap disagreement heuristic, not a confidence interval',
                selection_rule='override geometry only when supported gain exceeds .02 + 1*combined calibrated bootstrap std')
        elif self.exploration=='uniform':
            selected=int(self.rng.integers(len(candidates)));propensity=1./len(candidates)
        elif self.exploration=='softmax':
            distribution=np.exp((geometry-geometry.max())/.08);distribution/=distribution.sum()
            selected=int(self.rng.choice(len(candidates),p=distribution));propensity=float(distribution[selected])
        else:
            selected=int(np.argmax(geometry));propensity=1.
        c=candidates[selected];probability=float(probabilities[selected]) if probabilities is not None and supported[selected] else None
        self.next_goal_id+=1
        # HandoverRegion requires a numeric template value; expose probability only in diagnostics/attempt records.
        evaluated_region=(replace(c['region'],value=probability,source='calibrated_candidate_attempt_probability')
            if probability is not None else replace(c['region'],source='geometry_template_no_probability_claim'))
        goal=InterceptGoal(self.next_goal_id,float(forecast.times[0]),c['arrival'],evaluated_region,
            c['pose'],c['velocity'],c['angular_velocity'],forecast,c['check'],
            probability if probability is not None else c['geometry_score'])
        self.last_diagnostics.update(selected_success_probability=probability,
            score_kind='calibrated_attempt_probability' if probability is not None else 'geometry',
            selected_geometry_score=float(c['geometry_score']),
            selection_utility_kind='geometry_with_conservative_learned_override' if self.model is not None else 'geometry',
            context_in_success_head=self.owner.predictor=='context')
        self.owner.start_attempt(goal,c['features'],robot,propensity,len(candidates),probability)
        return goal


class AttemptInterception(OriginalAttemptInterception):
    def __init__(self,dynamics,model,collision_scene,goal_pose,config,*,predictor,ranking,value=None):
        super().__init__(dynamics,model,collision_scene,goal_pose,config,predictor=predictor,ranking=ranking,value=None)
        self.selector=AttemptSelector(self,self.selector,model=value,
            exploration=config.get('_exploration','geometry'),seed=config.get('_collection_seed',0))
