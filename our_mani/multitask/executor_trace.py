"""Aligned executor diagnostics from already computed public policy state."""
from copy import deepcopy
from types import MethodType
import numpy as np

SCHEMA = 'r10_public_observation_forecast_reference_execution_timeline_v1'


def install_trace(actor):
    m = actor.continuous_reference_manager
    if m.confirmations or actor.controller.feedback_step_calls:
        raise ValueError('install trace before the first counted policy tick')
    rows = []
    previous = actor.act
    def act(instance, observation, robot, remaining_seconds):
        command = previous(observation, robot, remaining_seconds)
        row = dict(time=float(robot.timestamp), remaining_seconds=float(remaining_seconds),
            phase_index=instance.controller.phase_index, command=command['command'],
            phase=command.get('phase'), reason=command.get('reason'),
            measured_TCP_xyzw=robot.pose.xyzw().tolist(), measured_joints=robot.joints.tolist(),
            measured_joint_velocity_proxy=robot.joint_velocities.tolist(),
            public_bodies={}, forecast={}, additional_physics_or_camera_calls=0)
        forecast = m.current_forecast
        for body in getattr(instance, 'previous_states', ()):
            row['public_bodies'][body.object_id] = dict(time=float(body.timestamp),
                pose_xyzw=body.pose.xyzw().tolist())
            if forecast is not None:
                predictions = []
                for delay in (0., .2, .5, 1.):
                    try:
                        pose, velocity, angular = forecast.at(body.object_id, robot.timestamp+delay)
                        predictions.append(dict(delay=delay, position=pose.position.tolist(),
                            linear_velocity=np.asarray(velocity).tolist(),
                            angular_velocity=np.asarray(angular).tolist()))
                    except (KeyError, IndexError, ValueError):
                        continue
                row['forecast'][body.object_id] = predictions
        row['forecast_observed_until'] = None if forecast is None else forecast.metadata.get('observed_until')
        issued = command.get('feedforward_execution')
        if issued is not None:
            row['reference'] = {key: deepcopy(issued.get(key)) for key in (
                'proposal_id', 'reference_kind', 'reference_from_joints', 'reference_from_velocity',
                'reference_to_joints', 'reference_to_velocity', 'plan_start', 'plan_end')}
        rows.append(row)
        return command
    actor.act = MethodType(act, actor)
    previous_metrics = actor.metrics_report
    def metrics(instance):
        return dict(previous_metrics(), executor_public_timeline=dict(schema=SCHEMA,
            events=deepcopy(rows), native_task_or_future_truth_read=False,
            additional_physics_or_camera_calls=0, commands_changed_by_observer=False))
    actor.metrics_report = MethodType(metrics, actor)
    return rows
