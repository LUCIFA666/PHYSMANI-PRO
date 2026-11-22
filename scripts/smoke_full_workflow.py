"""Real neural approach from Panda home; TRAIN/VAL development resets only."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'third_party/FASTER/packages/openpi-client/src')]
import numpy as np
from PIL import Image, ImageDraw
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from our_mani.benchmark_perception import RedBoxTracker, box_surface
from our_mani.capture_workflow import CaptureWorkflow
from our_mani.contracts import ObjectState, Pose, HandoverRegion
from our_mani.dynamics import ConstantVelocityDynamics
from our_mani.execution import FeasibleExecutor, CartesianActionLimiter
from our_mani.faster import ActionBuffer, FasterApproach
from our_mani.forecast_collision import ForecastCollisionScene
from our_mani.goal_perception import RedGoalMarker
from our_mani.local_capture import LocalConfig
from our_mani.local_learning import CandidateSuccessValue, LocalServo
from our_mani.planning import RegionSelector
from our_mani.pipeline import Stage
from our_mani.pyrep_env import LocalSimulation
from our_mani.reachability import TimedReachability
from our_mani.video import write_video
from collect_local_rollouts import initialization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18471)
    parser.add_argument('--episodes-per-task', type=int, default=1)
    parser.add_argument('--split', choices=('train', 'validation'), default='train')
    parser.add_argument('--out', default='artifacts/framework/full_workflow/neural_home_smoke_v1')
    parser.add_argument('--learned-local', action='store_true')
    parser.add_argument('--no-value', action='store_true', help='Explicit Value ablation; never used to claim the full method')
    parser.add_argument('--value-checkpoint', help='Opt-in controller-specific success model; existing runs keep their pinned Value')
    parser.add_argument('--dynamics', choices=('cv', 'driver', 'ngff_driver'), default='cv')
    parser.add_argument('--seconds', type=float, default=12.)
    parser.add_argument('--action-filter', choices=('joint_rate', 'bounded_waypoint', 'raw'), default='joint_rate')
    args = parser.parse_args()
    code_files = sorted((ROOT/'our_mani').glob('*.py'))+[Path(__file__).resolve()]
    code_sha256 = hashlib.sha256(b''.join(str(p.relative_to(ROOT)).encode()+p.read_bytes() for p in code_files)).hexdigest()
    import torch
    torch.set_num_threads(4)
    out = ROOT/args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((ROOT/f'artifacts/framework/protocol/{args.split}_manifest.json').read_text())['episodes']
    calibration = json.loads((ROOT/'artifacts/framework/local_policy/goal_marker_calibration.json').read_text())
    goal_offset = calibration['offset_train_calibrated_m']
    config = LocalConfig(max_speed=1., max_acceleration=4.)
    controller_hash = hashlib.sha256(b''.join((ROOT/'our_mani'/name).read_bytes() for name in
            ('local_capture.py', 'control.py', 'benchmark_perception.py', 'goal_perception.py', 'pyrep_env.py'))
            +json.dumps({'controller_config': asdict(config), 'servo_lead': 1.4}, sort_keys=True).encode()).hexdigest()
    learned = None
    if args.learned_local:
        checkpoint = ROOT/'artifacts/framework/local_policy/models_v1/servo.pt'
        learned = LocalServo.load(checkpoint)
        controller_hash += ':'+hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    value_path = ROOT/'artifacts/framework/local_policy'/('servo_models_v1' if learned else 'models_v1')/'value.pt'
    if args.value_checkpoint:
        if args.no_value:
            raise ValueError('--value-checkpoint and --no-value are mutually exclusive')
        value_path = ROOT/args.value_checkpoint
    value = None if args.no_value else CandidateSuccessValue(value_path, controller_sha256=controller_hash)
    results = []
    for task in ('pick_moving_target_on_the_table', 'pick_moving_target_on_the_table_high_speed'):
        for row in [r for r in rows if r['task']==task][:args.episodes_per_task]:
            env = LocalSimulation(task, row['path'], seed=421, servo_lead=1.4)
            client, workflow = None, None
            try:
                x, v, a, marker = initialization(row['path'])
                if not env.reset(target_position=x, target_velocity=v, target_acceleration=a, goal_position=marker):
                    raise RuntimeError('reset initialization failed')
                env.arm.set_joint_positions(env.home_joints.tolist(), disable_dynamics=True)
                env.arm.set_joint_target_positions(env.home_joints.tolist())
                for _ in range(3):
                    env.pyrep.step()
                tracker = RedBoxTracker()
                goal_detector = RedGoalMarker(goal_offset)
                dynamics = ConstantVelocityDynamics()
                if args.dynamics != 'cv':
                    from our_mani.adapted_dynamics import CausalDriverDynamics
                    base = None
                    if args.dynamics=='ngff_driver':
                        from our_mani.remote_dynamics import RemoteNGFF
                        base = RemoteNGFF(ROOT/'runtime/framework/ngff_dynamics.sock')
                    mode = 'ngff_driver_adapter' if base is not None else 'driver_model_without_ngff'
                    dynamics = CausalDriverDynamics(ROOT/f'artifacts/framework/domain_adaptation_h11/{mode}.pt', base)
                for _ in range(6):
                    obs, robot = env.observe()
                    obj, _ = tracker.observe(obs)
                    if hasattr(dynamics, 'observe'):
                        dynamics.observe([obj])
                    env.advance(robot.pose)
                goal_pose, goal_meta = goal_detector.observe(obs)
                client = WebsocketClientPolicy('127.0.0.1', args.port)
                buffer = ActionBuffer()
                approach = FasterApproach(client, buffer, clock=lambda: env.time)
                collision_scene = ForecastCollisionScene(env.arm, {'target':env.task.target_block})
                env.robot_model.collision_query = collision_scene.check
                env.robot_model.collision_scene = 'nominal_forecast_target_geometry_and_current_static_scene'
                reachability = TimedReachability(env.robot_model)
                regions = [HandoverRegion(f'candidate_{i}', 'target',
                            Pose.from_xyzw([dx, dy, dz, 0, 1, 0, 0]), [.04, .04, .04],
                            max_relative_speed=.14, source='TRAIN_range_templates_with_learned_controller_Value')
                           for i, (dx, dy, dz) in enumerate([(0,0,.07), (.04,0,.07), (-.04,0,.07),
                                                            (0,.04,.07), (0,-.04,.07), (0,0,.12)])]
                # Known task workspace plus 8cm one-tick reflection slack.
                # This is identical for every predictor and rejects impossible
                # future target states before expensive simulator IK.
                bounds = {'target': (np.array([-.03,-.53,.74]), np.array([.43,.53,.81]))}
                selector = RegionSelector(reachability, success_value=value, minimum_success_probability=.5 if value else 0.,
                                          object_position_bounds=bounds)
                validator = FeasibleExecutor(reachability)
                limiter = CartesianActionLimiter()
                workflow = CaptureWorkflow(dynamics, selector, regions, approach, buffer, goal_pose, config,
                                           learned_policy=learned, collision_scene=collision_scene)
                log, frames, carrying = [], [], None
                wall_start, sim_start = time.perf_counter(), env.time
                rejection_counts, lag = {}, []
                for tick in range(int(args.seconds/.05)):
                    if (ROOT/'runtime/framework/state/STOP').exists():
                        break
                    observation, robot = env.observe()
                    metric_position = env.metric_target_position().copy()
                    observation['prompt'] = 'pick up the red block and lift it up to the target'
                    try:
                        obj, perception = tracker.observe(observation)
                        if robot.contact:
                            carrying = (robot.pose.inverse_apply(obj.pose.position), robot.pose.rotation.T@obj.pose.rotation)
                    except ValueError as error:
                        obj, perception = None, {'error': str(error)}
                        if robot.contact and carrying is not None:
                            local_position, local_rotation = carrying
                            pose = Pose(robot.pose.apply(local_position), robot.pose.rotation@local_rotation)
                            obj = ObjectState('target', env.time, pose, box_surface(pose, tracker.prior.dimensions),
                                              robot.linear_velocity, robot.angular_velocity, confidence=.8)
                            perception = {'method': 'rigid_carry_from_last_visual_contact_and_current_proprioception'}
                    decision = workflow.step(observation, robot, obj, tick=tick)
                    applied, advanced, reason = False, False, 'hold'
                    if decision.command=='faster_action':
                        raw_action = decision.details['action']
                        action = limiter.filter(raw_action, robot) if args.action_filter=='bounded_waypoint' else raw_action
                        if args.action_filter=='joint_rate':
                            command, reason = validator.prepare_joint_limited(action, robot, now=env.time)
                        else:
                            command, reason = validator.prepare(action, robot, now=env.time)
                        if command is not None:
                            env.arm.set_joint_target_positions(command.joints.tolist())
                            env.gripper.actuate(1., velocity=.2)
                            env.scene.step()
                            env.time += .05
                            applied, advanced = True, True
                        else:
                            rejection_counts[reason] = rejection_counts.get(reason, 0)+1
                    elif decision.command=='local_capture':
                        applied = env.advance(decision.pose, opening=decision.details['gripper_open'])
                        advanced = True
                        reason = 'same_local_rollout_executor'
                    if not advanced:
                        env.advance(robot.pose, opening=float(not robot.contact))
                    collision = bool(env.arm.check_arm_collision())
                    log.append({'tick': tick, 'simulation_time': robot.timestamp, 'stage': decision.stage.value,
                                'command': decision.command, 'applied': applied, 'execution_reason': reason,
                                'contact': robot.contact, 'official_task_success': robot.task_success,
                                'target_position_metric_only': metric_position.tolist(),
                                'local_stage': (decision.details or {}).get('stage'),
                                'handover_check': {k:(v if not isinstance(v,float) or np.isfinite(v) else None)
                                                   for k,v in ((decision.details or {}).get('handover') or {}).items()},
                                'robot_pose':robot.pose.xyzw().tolist(),
                                'robot_linear_velocity':robot.linear_velocity.tolist(),
                                'robot_joint_positions':robot.joints.tolist(),
                                'robot_joint_velocities':robot.joint_velocities.tolist(),
                                'selected_goal_pose':workflow.goal.pose.xyzw().tolist() if workflow.goal else None,
                                'selected_arrival_time':workflow.goal.arrival_time if workflow.goal else None,
                                'perception': perception, 'collision_metric': collision})
                    if tick % 20 == 0:
                        print(json.dumps({'task':task, 'tick':tick, 'stage':decision.stage.value,
                                          'command':decision.command, 'execution_reason':reason,
                                          'selector':selector.last_diagnostics}, ensure_ascii=False), flush=True)
                    names = ('front','8','16','36')
                    pictures = observation['observation/images']
                    mosaic = np.concatenate([np.concatenate([pictures[n] for n in names[:2]],1),
                                             np.concatenate([pictures[n] for n in names[2:]],1)],0)
                    image = Image.fromarray(mosaic)
                    ImageDraw.Draw(image).text((8,8), f"REAL FASTER | {args.dynamics} | {decision.stage.value}\nt={robot.timestamp:.2f}s | {reason}",
                                               fill='white', stroke_width=1, stroke_fill='black')
                    frames.append(np.asarray(image))
                    lag.append(max(0., time.perf_counter()-wall_start-(env.time-sim_start)))
                    if collision or decision.stage in (Stage.SUCCESS, Stage.FAILED):
                        break
                    time.sleep(max(0., wall_start+(env.time-sim_start)-time.perf_counter()))
                name = task+'__'+row['raw_episode_id']
                write_video(out/(name+'.mp4'), frames)
                record = {'created_utc': datetime.now(timezone.utc).isoformat(), 'task':task,
                          'source_episode':row['raw_episode_id'], 'split':args.split,
                          'kind':'full_neural_workflow_development_reset_not_official_benchmark',
                          'executed_project_code_sha256_at_start':code_sha256,
                          'formal_test_used':False, 'neural_server':client.get_server_metadata(),
                          'local_controller_sha256':controller_hash, 'value_model':str(value_path) if value else None,
                          'value_outcome':value.value.metadata.get('outcome', 'stable_lift') if value else None,
                          'value_probability_scope':value.value.metadata.get('probability_scope', 'local handover stable lift; excludes FASTER approach') if value else None,
                          'value_input_support_sha256':value.support_sha256 if value else None,
                          'dynamics':args.dynamics, 'success':workflow.stage==Stage.SUCCESS,
                          'approach_execution_filter':args.action_filter,
                          'faster_waypoint_supplies_motion_direction':True,
                          'handovers':sum(e['to']==Stage.LOCAL.value for e in workflow.events),
                          'captured':any(r['contact'] for r in log),
                          'local_stages_executed':sorted({r['local_stage'] for r in log if r['local_stage'] is not None}),
                          'faster_actions_applied':sum(r['command']=='faster_action' and r['applied'] for r in log),
                          'rejected_neural_actions':rejection_counts, 'events':workflow.events,
                          'maximum_control_lag_seconds':max(lag), 'wall_seconds':time.perf_counter()-wall_start,
                          'simulated_seconds':env.time-sim_start, 'goal_perception':goal_meta,
                          'collision_checks':'nominal forecast target geometry plus current static scene',
                          'collision_uncertainty_margin_m':0.,
                          'all_moving_obstacles_forecast':False,
                          'declared_object_workspace_m':{k:[np.asarray(a).tolist() for a in b] for k,b in bounds.items()},
                          'handover_remaining_seconds_may_exceed_training_maximum_0_8':True,
                          'steps':log, 'approach_timings':approach.timings}
                (out/(name+'.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
                results.append({k:v for k,v in record.items() if k not in ('steps','approach_timings')})
                (out/'report.json').write_text(json.dumps({'formal_test_used':False, 'results':results}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
                print(json.dumps(results[-1], ensure_ascii=False), flush=True)
            finally:
                if workflow is not None:
                    workflow.close()
                if client is not None:
                    client._ws.close()
                env.close()


if __name__=='__main__':
    main()
