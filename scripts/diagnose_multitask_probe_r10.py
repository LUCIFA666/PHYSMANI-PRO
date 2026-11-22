"""TRAIN-only single-case diagnostic; never a collection/evaluation receipt."""
import argparse
from datetime import datetime, timezone
import faulthandler
import importlib.util
import json
from pathlib import Path
import sys
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--maximum-ticks', type=int, default=1)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if not args.out.is_relative_to((ROOT/'artifacts/framework/multitask_r10').resolve()):
        raise ValueError('diagnostic outputs must stay in new desktop artifacts')
    args.out.mkdir(parents=True, exist_ok=False)
    module = importlib.util.spec_from_file_location('r10_collector_diagnostic', ROOT/'scripts/collect_multitask_training_r10.py')
    collector = importlib.util.module_from_spec(module); module.loader.exec_module(collector)
    plan = collector.verify_plan(args.plan)
    collector.require_r09_release(ROOT/'artifacts/framework/aligned_r09/final_report.json')
    demos = json.loads((ROOT/'artifacts/framework/aligned_r09/demo_index.json').read_text())
    if demos.get('complete') is not True or demos.get('completed') != 10:
        raise ValueError('wait for ten demos')
    case = next(c for c in plan['cases'] if c['case_id'] == args.case_id)
    if case['split'] != 'train' or not 1 <= args.maximum_ticks <= 200:
        raise ValueError('diagnostics are TRAIN only')
    from our_mani.multitask.simulation import AlignedDevelopmentSimulation
    from our_mani.contracts import Pose
    report = dict(diagnostic_only=True, formal_metrics_included=False,
        data_collection_receipt=False, source=case['source'], plan_sha256=plan['plan_sha256'],
        created_utc=datetime.now(timezone.utc).isoformat(), probe_records=[],
        modified_state_check_or_tolerances=False,
        instrumentation='wrap original model._probe; record before/after; original exceptions preserved',
        sources_sha256={p:collector.sha(ROOT/p) for p in (
            'scripts/diagnose_multitask_probe_r10.py','our_mani/multitask/maps.py',
            'our_mani/multitask/simulation.py','scripts/collect_multitask_training_r10.py')})
    def save():
        (args.out/'diagnostic.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    env = None
    try:
        print('construct', flush=True)
        env = AlignedDevelopmentSimulation(case['task'], case['source']['path'],
            source=case['source'], split='train', protocol=plan['protocol'],
            seed=plan['protocol'].get('seed',0),replica=case['replica'],
            servo_lead=plan['protocol'].get('local_servo_lead',1.4))
        env.collection_variation=int(case['source']['variation'])
        env.reset_development(variation=env.collection_variation)
        original = env.robot_model._probe
        def state():
            return dict(q=np.asarray(env.arm.get_joint_positions()).tolist(),
                targets=np.asarray(env.arm.get_joint_target_positions()).tolist(),
                ik=np.asarray(env.arm._ik_target.get_pose()).tolist(),
                task_seconds=float(env.task.t), physics_seconds=float(env.physics_time()))
        def instrumented(joints, fn):
            row=dict(before=state(), requested_joints=np.asarray(joints).tolist())
            try:
                result=original(joints,fn);row['exception']=None;return result
            except Exception as error:
                row['exception']=repr(error);raise
            finally:
                row['after']=state()
                for name in ('q','targets','ik'):
                    row[name+'_raw_max_abs_difference']=float(np.max(np.abs(
                        np.asarray(row['before'][name])-np.asarray(row['after'][name]))))
                a,b=row['before']['ik'],row['after']['ik']
                row['ik_physical_rotation_difference_rad']=float((Rotation.from_quat(a[3:]).inv()*Rotation.from_quat(b[3:])).magnitude())
                report['probe_records'].append(row);save();print(json.dumps(row),flush=True)
        env.robot_model._probe=instrumented
        teacher=collector.native_teacher(env)
        adapter=collector.TRAINWaypointOpeningAdapter(plan['protocol']['gripper_timing_config'])
        for tick in range(args.maximum_ticks):
            if env._native_terminal_latched:break
            observation,robot=env.observe()
            action=teacher()
            if tick==0:
                report['teacher_first_action']={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in action.items()}
            opening,timing=adapter.apply(action,robot)
            report['current_tick']=tick;report['current_opening_timing']=timing
            env.advance(Pose.from_xyzw(action['tcp_xyzw']),opening=opening)
            report['first_tick_completed']=True
            report['completed_ticks']=env.ledger.ticks
    except Exception as error:
        report['error']=repr(error);report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
    finally:
        if env is not None:
            try:env.close()
            except Exception as error:report['cleanup_error']=repr(error)
        save()


if __name__=='__main__':main()
