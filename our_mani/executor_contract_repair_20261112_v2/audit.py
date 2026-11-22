"""Independent finite-difference audit of actually issued position commands."""
import json
from pathlib import Path
import numpy as np


def audit_execution(path):
    rows=[json.loads(line)for line in Path(path).read_text().splitlines()]
    rows=[r for r in rows if r['executed_ticks']]
    ticks=[t for r in rows for t in r['ticks']if t['tick_completed']]
    if not ticks:return dict(complete=True,executed_ticks=0,all_reference_differences_within_limits=True)
    p=np.array([t['reference_positions']for t in ticks])
    initial=np.asarray(rows[0]['initial_measured_positions'])
    v0=np.asarray(rows[0]['initial_measured_velocities'])
    v=np.diff(np.vstack((initial,p)),axis=0)/.05
    a=np.diff(np.vstack((v0,v)),axis=0)/.05
    j=np.diff(np.vstack((np.zeros(len(initial)),a)),axis=0)/.05
    bounds=rows[0]['limits']
    limits=[np.asarray(bounds[k])for k in ('effective_reference_velocity_limits',
              'controller_reference_acceleration_limits','controller_reference_jerk_limits')]
    ratios=[np.max(np.abs(x)/limit,axis=1)for x,limit in zip((v,a,j),limits)]
    violations=[int(np.sum(np.any(np.abs(x)>limit+1e-8,axis=1)))for x,limit in zip((v,a,j),limits)]
    errors=np.array([max(abs(x)for x in t['tracking_error_positions'])for t in ticks])
    position_ok=bool(np.all(p>=np.asarray(bounds['native_joint_position_lower'])-1e-8)and
                     np.all(p<=np.asarray(bounds['native_joint_position_upper'])+1e-8))
    return dict(complete=True,executed_ticks=len(ticks),max_velocity_acceleration_jerk_ratio=[float(x.max())for x in ratios],
                velocity_acceleration_jerk_violating_ticks=violations,
                all_reference_differences_within_limits=not any(violations)and position_ok,
                all_position_commands_within_native_bounds=position_ok,
                unexpected_physics_ticks=sum(abs(r['unexpected_physics_ticks'])for r in rows),
                tracking_error_p50_p95_max_rad=np.quantile(errors,[.5,.95,1]).tolist(),
                seconds_tracking_error_over_05rad=float(np.sum(errors>.5)*.05))
