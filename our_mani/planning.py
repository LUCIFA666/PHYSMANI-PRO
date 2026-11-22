"""Joint selection of object-relative region, absolute time, and terminal twist."""
from collections import Counter
from dataclasses import replace

import numpy as np

from .contracts import InterceptGoal


class RegionSelector:
    def __init__(self, reachability, *, time_penalty=0.15, effort_penalty=0.03,
                 latency_reserve=0.05, relative_velocity_options=((0., 0., 0.),),
                 success_value=None, minimum_success_probability=0., object_position_bounds=None):
        self.reachability = reachability
        self.time_penalty, self.effort_penalty = time_penalty, effort_penalty
        self.latency_reserve = latency_reserve
        self.relative_velocity_options = np.asarray(relative_velocity_options, dtype=float)
        self.success_value, self.minimum_success_probability = success_value, minimum_success_probability
        self.object_position_bounds = object_position_bounds or {}
        self.next_goal_id = 0
        self.last_diagnostics = {}

    def select(self, forecast, regions, robot, now, *, min_horizon=0.15, stride=2):
        candidates, rejected = [], Counter()
        for region in regions:
            if region.object_id not in forecast.object_ids:
                rejected["region_target_missing"] += 1
                continue
            for arrival in forecast.times[1::stride]:
                duration = arrival-now-self.latency_reserve
                if duration < min_horizon:
                    rejected["insufficient_remaining_horizon"] += 1
                    continue
                obj_pose, obj_v, obj_w = forecast.at(region.object_id, arrival)
                bounds = self.object_position_bounds.get(region.object_id)
                if bounds is not None and (np.any(obj_pose.position < bounds[0]) or np.any(obj_pose.position > bounds[1])):
                    rejected['forecast_outside_declared_target_workspace'] += 1
                    continue
                pose, surface_v = region.in_world(obj_pose, obj_v, obj_w)
                for relative in self.relative_velocity_options:
                    if np.linalg.norm(relative) > region.max_relative_speed:
                        rejected["relative_speed_outside_region"] += 1
                        continue
                    desired_v = surface_v + pose.rotation@relative
                    check = self.reachability.check(robot, pose, desired_v, obj_w, duration)
                    if not check.feasible:
                        rejected[check.reason] += 1
                        continue
                    effort = np.linalg.norm(check.joints-robot.joints) if check.joints is not None else 0.
                    probability = region.value
                    if self.success_value is not None:
                        probability = float(self.success_value(region=region, object_pose=obj_pose,
                            object_velocity=obj_v, object_angular_velocity=obj_w, robot_pose=pose,
                            robot_velocity=desired_v, robot_angular_velocity=obj_w))
                        if not np.isfinite(probability) or not 0 <= probability <= 1:
                            rejected['value_out_of_domain_or_invalid'] += 1
                            continue
                    if probability < self.minimum_success_probability:
                        rejected['value_below_capture_threshold'] += 1
                        continue
                    evaluated_region = replace(region, value=probability) if self.success_value is not None else region
                    score = np.log(max(probability, 1e-6))-self.time_penalty*(arrival-now)-self.effort_penalty*effort
                    candidates.append((score, arrival, evaluated_region, pose, desired_v, obj_w, check))
        self.last_diagnostics = {"feasible_candidates": len(candidates), "rejections": dict(rejected)}
        if not candidates:
            return None
        score, arrival, region, pose, v, w, check = max(candidates, key=lambda x: x[0])
        self.last_diagnostics['selected_success_probability'] = float(region.value)
        self.next_goal_id += 1
        return InterceptGoal(self.next_goal_id, float(forecast.times[0]), float(arrival), region,
                             pose, v, w, forecast, check, float(score))
