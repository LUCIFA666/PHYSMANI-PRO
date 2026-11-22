"""Read-only native success-component witnesses, never supplied to the policy."""
import numpy as np

SCHEMA='r10_button_native_success_components_metric_only_v100'


def clock(env):
    return (int(env.ledger.ticks),float(env.task.t),float(env.physics_time()),float(env.time))


def snapshot(env):
    before=clock(env);task=env.task
    condition=task.goal_condition;descent=task.descent_condition
    position=float(task.joint.get_joint_position());original=float(condition._original_pos)
    threshold=float(condition._pos);window=int(descent._window);minimum=float(descent._min_descent)
    history=np.asarray(list(task._tip_z_history),float)
    if (not np.isfinite([position,original,threshold,minimum]).all()
            or not np.isfinite(history).all() or threshold!=.003 or window!=10 or minimum!=.03):
        raise ValueError('native button success source contract changed')
    drop=None if len(history)<window else float(history[-window]-history[-1])
    joint=bool(abs(position-original)>threshold);down=bool(drop is not None and drop>minimum)
    after=clock(env)
    if before!=after:raise ValueError('button success metric read changed native clock')
    return dict(schema=SCHEMA,time=float(env.time),joint_position=position,
        original_joint_condition_position=original,absolute_displacement=abs(position-original),
        joint_threshold=threshold,joint_component_met=joint,
        native_tip_z_history=history.tolist(),window_steps=window,
        min_descent=minimum,descent_over_native_window=drop,descent_component_met=down,
        simultaneous_components_met=joint and down,native_task_success_not_replaced=True,
        metric_only=True,diagnostic_values_supplied_to_policy=False,
        additional_camera_queries=0,additional_physics_ticks=0,clock_unchanged=True)
