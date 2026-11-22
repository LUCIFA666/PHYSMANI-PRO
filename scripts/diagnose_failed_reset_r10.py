"""One TRAIN reset-failure diagnostic; no trajectories enter any fitting.

Keep the original forty-attempt reset intact. Its final exception restores the
task tree; reconstruct the SAME fixed variation once before querying paths.
Neither returned path is stepped and no validation function is replaced.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import collect_multitask_training_r10 as c

VERSION = 'r10_one_fixed_TRAIN_reset_failure_readonly_diagnostic_v1'


def clean(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    return value


def rng_fingerprint():
    state = np.random.get_state()
    return dict(numpy=hashlib.sha256(state[0].encode()+state[1].tobytes()
        +repr(state[2:]).encode()).hexdigest(),
        python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest())


def exception_chain(error):
    rows, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error)); rows.append(dict(type=type(error).__module__+'.'+type(error).__name__,
                                            message=str(error)))
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return rows


def prepare(plan_path):
    full_path = c.BASE/'TRAIN_full_collection_plan_v11.json'
    full = c.verify_plan(full_path)
    # Fixed before this run: an existing teacher/hold reset-failure parent.
    case = next(row for row in full['cases'] if row['task']=='beat_the_rotating_buzz'
                and row['parent_raw_episode_id']=='episode104' and row['teacher_used'])
    old = c.BASE/'TRAIN_full_collection_v11/infrastructure_attempts'/case['case_id']/'0'
    if "Can't reach waypoint 0" not in (old/'native.log').read_text(errors='replace'):
        raise ValueError('the fixed source must already have the declared reset failure')
    inputs = dict(full['inputs'])
    cache=ROOT/case['offline_demonstration']['path']
    if c.sha(cache)!=case['offline_demonstration']['sha256']:
        raise ValueError('fixed source cache changed from the full TRAIN plan')
    extra = [Path(__file__).resolve(), full_path, old/'native.log', old/'parent_receipt.json',
             cache, Path(case['source']['path'])]
    native = Path(full['protocol']['baseline_repo']).resolve()
    extra += [native/'third_party/RLBench/rlbench'/name for name in (
        'task_environment.py', 'backend/scene.py', 'backend/task.py', 'backend/waypoints.py')]
    extra += [native/'third_party/PyRep/pyrep'/name for name in (
        'robots/robot_component.py', 'robots/arms/arm.py',
        'robots/configuration_paths/arm_configuration_path.py')]
    inputs.update({str(path.resolve()): c.sha(path) for path in extra})
    source = Path(case['source']['path']); stat = source.stat()
    if dict(size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,inode=stat.st_ino,device=stat.st_dev) != case['offline_demonstration']['source_file_stat']:
        raise ValueError('original TRAIN HDF5 source stat changed')
    plan = dict(version=VERSION, created_utc=c.utc(), TRAIN_only=True, VAL_TEST_used=False,
        original_full_plan=dict(path=str(full_path.relative_to(ROOT)),sha256=c.sha(full_path)),
        original_failed_case=case, inputs_sha256=inputs,
        output=str((c.BASE/'reset_failure_diagnostic_v1').relative_to(ROOT)),
        original_reset_attempts=40, source_scene_or_seed_changed=False,
        diagnostic_native_workers=1, model_or_policy_evaluation=False,
        outputs_may_enter_fitting=False, waypoint_validation_bypassed=False,
        query_order=['original_Point_get_path_ignore_collisions_true','actual_teacher_pure_arm_quaternion'],
        actual_teacher_parameters=dict(trials=100,max_configs=10,max_time_ms=10,
            trials_per_goal=5,algorithm='RRTConnect',initial_ignore_collisions=False,
            fallback_ignore_collisions=True),
        reconstruction='After saving post-failure restored state, same native task.init_episode(0) once; no init_task/RNG reseed/physics step.',
        prohibited=['path.step','env.step','env.hold_tick','new seed','new variation',
                    'native source mutation','validation monkeypatch','dataset or formal result insertion'])
    plan['plan_sha256']=c.canonical_sha(plan)
    c.write_json(plan_path,plan)
    return plan


def verify(path):
    plan=json.loads(Path(path).read_text())
    if plan['version']!=VERSION or plan['plan_sha256']!=c.canonical_sha(
            {k:v for k,v in plan.items() if k!='plan_sha256'}):
        raise ValueError('diagnostic plan identity changed')
    for name,digest in plan['inputs_sha256'].items():
        if c.sha(name)!=digest:raise ValueError('diagnostic input changed: '+name)
    if plan['original_failed_case']['split']!='train' or plan['TRAIN_only'] is not True:
        raise ValueError('diagnostic is TRAIN only')
    return plan


def robot_state(env):
    return dict(q=list(env.arm.get_joint_positions()),
        targets=list(env.arm.get_joint_target_positions()),
        target_velocities=list(env.arm.get_joint_target_velocities()),
        tip=list(env.arm.get_tip().get_pose()), ik_target=list(env.arm._ik_target.get_pose()),
        gripper_q=list(env.gripper.get_joint_positions()))


def pose_equal(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return (a.shape==b.shape==(7,) and np.allclose(a[:3],b[:3],atol=1e-6,rtol=0)
        and Rotation.from_quat(a[3:]).inv().__mul__(Rotation.from_quat(b[3:])).magnitude()<=1e-6)


def state_equal(a,b):
    return all(np.allclose(a[k],b[k],atol=1e-6,rtol=0)
               for k in ('q','targets','target_velocities','gripper_q')) and all(
                   pose_equal(a[k],b[k]) for k in ('tip','ik_target'))


def clock(env):return [env.physics_time(),getattr(env.task,'t',None)]


def snapshot(env):
    result=dict(clock=clock(env),robot=robot_state(env),seed=env.seed,
        variation=getattr(env.task,'var_index',None),
        motion_driver=clean(getattr(env.task,'var2target_state_list',{})),
        frame_pose=list(env.task._frame_base.get_pose()),wand_pose=list(env.task.wand.get_pose()),
        waypoint0_pose=list(env.task.w0.get_pose()),waypoint0_euler=list(env.task.w0.get_orientation()),
        waypoint0_extension=env.task.w0.get_extension_string(),RNG=rng_fingerprint())
    cyclic,intervals=env.arm.get_joint_intervals();limits=np.asarray(intervals,float)
    lo=np.where(cyclic,-np.inf,limits[:,0]);hi=np.where(cyclic,np.inf,limits[:,0]+limits[:,1])
    q=np.asarray(result['robot']['q'])
    result['joint_limits']=dict(cyclic=list(cyclic),native_intervals=clean(intervals),
        lower=[None if not np.isfinite(x) else float(x) for x in lo],
        upper=[None if not np.isfinite(x) else float(x) for x in hi],
        all_actual_joints_finite_and_in_interval=bool(np.isfinite(q).all() and np.all(q>=lo) and np.all(q<=hi)))
    before=robot_state(env);ticks=clock(env)
    try:result['arm_collision']=bool(env.arm.check_arm_collision())
    except Exception as error:result['arm_collision_error']=exception_chain(error)
    if clock(env)!=ticks or not state_equal(before,robot_state(env)):
        raise RuntimeError('read-only collision query mutated robot or clock')
    return result


def query(env,label,call):
    before=robot_state(env);ticks=clock(env);rng=rng_fingerprint()
    body_before={name:list(obj.get_pose()) for name,obj in (
        ('frame',env.task._frame_base),('wand',env.task.wand),('waypoint0',env.task.w0))}
    cyclic,intervals=env.arm.get_joint_intervals();intervals=np.asarray(intervals,float)
    lo=np.where(cyclic,-np.inf,intervals[:,0]);hi=np.where(cyclic,np.inf,intervals[:,0]+intervals[:,1])
    if np.any(np.asarray(before['q'])<lo) or np.any(np.asarray(before['q'])>hi):
        raise RuntimeError('cannot safely probe or restore an actual out-of-interval joint')
    record=dict(label=label,before=before,clock_before=ticks,query_success=False)
    path=None
    try:
        path=call();points=np.asarray(path._path_points,float).reshape(-1,len(env.arm.joints))
        if not len(points) or not np.isfinite(points).all():raise RuntimeError('invalid returned native path')
        if path._rml_handle is not None:raise RuntimeError('pure query unexpectedly allocated stepped RML')
        record.update(query_success=True,path_points=len(points),first_configuration=points[0].tolist(),
            last_configuration=points[-1].tolist(),path_was_stepped=False)
    except Exception as error:
        record.update(exception_chain=exception_chain(error),traceback=traceback.format_exc())
    finally:
        after=robot_state(env);record['before_restoration']=after
        record['restore_was_needed']=not state_equal(before,after)
        # Normal public setters only, no dynamics disabling/configuration tree.
        if not np.allclose(before['q'],after['q'],atol=1e-6,rtol=0):env.arm.set_joint_positions(before['q'])
        if not np.allclose(before['targets'],after['targets'],atol=1e-6,rtol=0):env.arm.set_joint_target_positions(before['targets'])
        if not np.allclose(before['target_velocities'],after['target_velocities'],atol=1e-6,rtol=0):env.arm.set_joint_target_velocities(before['target_velocities'])
        if not pose_equal(before['ik_target'],after['ik_target']):env.arm._ik_target.set_pose(before['ik_target'])
        record['after']=robot_state(env);record['clock_after']=clock(env)
        record['clock_unchanged']=clock(env)==ticks
        record['robot_restored']=state_equal(before,record['after'])
        record['Python_numpy_RNG_unchanged']=rng_fingerprint()==rng
        record['native_Cpp_planner_RNG_not_observable']=True
        body_after={name:list(obj.get_pose()) for name,obj in (
            ('frame',env.task._frame_base),('wand',env.task.wand),('waypoint0',env.task.w0))}
        record['task_body_poses_unchanged']=all(pose_equal(body_before[k],body_after[k]) for k in body_before)
        if path is not None:
            if path._rml_handle is not None:
                from pyrep.backend import sim
                sim.simRMLRemove(path._rml_handle);path._rml_handle=None
            path.clear_visualization()
        if not all(record[k] for k in ('clock_unchanged','robot_restored','Python_numpy_RNG_unchanged','task_body_poses_unchanged')):
            raise RuntimeError('pure query failed clock/robot/RNG/body restoration guard: '+label)
    return record


def run(plan_path):
    plan=verify(plan_path);case=plan['original_failed_case']
    guard=c.require_r09_release(ROOT/'artifacts/framework/aligned_r09/final_report.json')
    out=c.destination(ROOT/plan['output'])
    if out.exists():raise ValueError('fresh diagnostic output required')
    out.mkdir(parents=True)
    report=dict(version=VERSION,created_utc=c.utc(),complete=False,process_id=os.getpid(),
        plan=dict(path=str(Path(plan_path).resolve().relative_to(ROOT)),sha256=c.sha(plan_path)),
        TRAIN_source=case,guard=guard,native_workers=1,model_or_policy_evaluation=False,
        policy_failure_label_assigned=False,outputs_may_enter_fitting=False,queries=[])
    save=lambda:c.write_json(out/'report.json',clean(report),overwrite=True)
    save();print(json.dumps(dict(stage='starting',pid=os.getpid(),source=case['parent_source_group'])),flush=True)
    env=None;failed=False
    try:
        from our_mani.multitask.simulation import AlignedDevelopmentSimulation
        from our_mani.multitask.train_teacher import plan_arm_path
        from our_mani.contracts import Pose
        from pyrep.const import ConfigurationPathAlgorithms
        from pyrep.errors import ConfigurationPathError
        from rlbench.backend.waypoints import Point
        env=AlignedDevelopmentSimulation(case['task'],case['source']['path'],source=case['source'],
            split='train',protocol=json.loads((ROOT/plan['original_full_plan']['path']).read_text())['protocol'],
            seed=0,replica=case['replica'])
        report['actual_derived_seed']=env.seed
        report['original_reset_clock_before']=clock(env)
        try:
            env.reset_development(variation=case['source']['variation'])
            report['original_reset_succeeded']=True
        except Exception as error:
            report.update(original_reset_succeeded=False,original_reset_exception_chain=exception_chain(error),
                original_reset_traceback=traceback.format_exc())
        report['original_reset_clock_after']=clock(env)
        report['post_reset_or_failure_restored_snapshot']=snapshot(env);save()
        if report['original_reset_succeeded']:
            report['reconstruction_performed']=False
        else:
            # Native Scene restores its initial task tree after the final
            # failed attempt. Reapply that SAME existing var2 state once.
            before=robot_state(env);ticks=clock(env);rng=rng_fingerprint()
            driver=c.canonical_sha(clean(env.task.var2target_state_list));seed=env.seed
            descriptions=env.task.init_episode(case['source']['variation'])
            reconstructed=dict(descriptions=descriptions,clock_unchanged=clock(env)==ticks,
                robot_unchanged=state_equal(before,robot_state(env)),
                Python_numpy_RNG_unchanged=rng_fingerprint()==rng,
                motion_driver_unchanged=c.canonical_sha(clean(env.task.var2target_state_list))==driver,
                derived_seed_unchanged=env.seed==seed)
            report.update(reconstruction_performed=True,reconstruction=clean(reconstructed))
            if not all(reconstructed[k] for k in ('clock_unchanged','robot_unchanged',
                    'Python_numpy_RNG_unchanged','motion_driver_unchanged','derived_seed_unchanged')):
                save();raise RuntimeError('same-variation reconstruction guard failed; no queries allowed')
        report['actual_fixed_source_query_snapshot']=snapshot(env);save()
        point=Point(env.task.w0,env.environment._robot)
        report['Point_flags']=dict(linear_only=point._linear_only,ignore_collisions=point._ignore_collisions,
                                  extension=point.get_ext())
        report['queries'].append(query(env,'original_Point_get_path_ignore_collisions_true',
            lambda:point.get_path(ignore_collisions=True)));save()
        goal=Pose.from_xyzw(env.task.w0.get_pose())
        report['queries'].append(query(env,'actual_teacher_pure_arm_quaternion',lambda:plan_arm_path(
            env.arm,goal,family=env.spec.family,algorithm=ConfigurationPathAlgorithms.RRTConnect,
            configuration_path_error=ConfigurationPathError)));save()
        verify(plan_path)
        report.update(complete=True,final_all_input_SHA_verified=len(plan['inputs_sha256']),
            physical_scene_legality_certified=False,mathematical_IK_unreachability_certified=False,
            no_source_scene_seed_variation_or_parameter_change=True,
            limitation='Bounded native path queries diagnose the two actual APIs; native C++ planner RNG is not controlled or replayed, and failed sampling does not prove mathematical unreachability.')
    except Exception as error:
        failed=True;report.update(diagnostic_error=exception_chain(error),diagnostic_traceback=traceback.format_exc())
    finally:
        if env is not None:
            try:env.close();report['native_cleanup_complete']=True
            except Exception as error:
                failed=True;report.update(native_cleanup_complete=False,native_cleanup_error=exception_chain(error))
        report['finished_utc']=c.utc();save()
    print(json.dumps(dict(complete=report['complete'],original_reset_succeeded=report.get('original_reset_succeeded'),
        query_successes=[row['query_success'] for row in report['queries']],report=str((out/'report.json').relative_to(ROOT)))),flush=True)
    return 1 if failed else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--run',action='store_true')
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    if args.prepare:
        plan=prepare(args.plan);print(json.dumps(dict(plan_sha256=plan['plan_sha256'],case=plan['original_failed_case']['case_id'])))
    else:raise SystemExit(run(args.plan))


if __name__=='__main__':main()
