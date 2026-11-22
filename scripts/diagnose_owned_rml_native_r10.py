"""Exactly one source-fixed TRAIN virtual-RML native diagnostic, never data."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from our_mani.multitask.evaluation import digest,file_sha,atomic_json


def require(value,message):
    if not value:raise ValueError(message)


def snapshot(env):
    return dict(physics_time=env.physics_time(),task_time=float(env.task.t),ledger_ticks=env.ledger.ticks,
        joint_positions=np.asarray(env.arm.get_joint_positions()).tolist(),
        joint_targets=np.asarray(env.arm.get_joint_target_positions()).tolist(),tip_xyzw=list(env.arm.get_tip().get_pose()))


def delta(first,second):
    return dict(physics_time=second['physics_time']-first['physics_time'],task_time=second['task_time']-first['task_time'],
        ledger_ticks=second['ledger_ticks']-first['ledger_ticks'],
        maximum_joint_position_change=float(np.max(np.abs(np.asarray(second['joint_positions'])-first['joint_positions']))),
        maximum_joint_target_change=float(np.max(np.abs(np.asarray(second['joint_targets'])-first['joint_targets']))),
        tip_position_change_m=float(np.linalg.norm(np.asarray(second['tip_xyzw'][:3])-first['tip_xyzw'][:3])),
        tip_SO3_change_rad=float((Rotation.from_quat(second['tip_xyzw'][3:])*Rotation.from_quat(first['tip_xyzw'][3:]).inv()).magnitude()))


def tracked_probe(env,path,callback):
    from pyrep.backend import sim
    originals={name:getattr(sim,name) for name in ('simRMLPos','simRMLStep','simRMLRemove')}
    active=set();records={};forbidden=[];returned=None;error=None;cleanup_errors=[]
    before=snapshot(env)
    def allocate(*args,**kwargs):
        handle=originals['simRMLPos'](*args,**kwargs);active.add(handle)
        records[handle]=dict(handle=int(handle),max_velocity_acceleration_jerk=list(args[4]),samples=[],removed=False)
        return handle
    def advance(handle,*args,**kwargs):
        state,pva=originals['simRMLStep'](handle,*args,**kwargs)
        records[handle]['samples'].append((int(state),list(pva)));return state,pva
    def remove(handle):
        result=originals['simRMLRemove'](handle);active.discard(handle)
        if handle in records:records[handle]['removed']=True
        return result
    def forbid(name):
        def reject(*args,**kwargs):forbidden.append(name);raise RuntimeError('virtual probe attempted forbidden '+name)
        return reject
    with ExitStack() as stack:
        for name,function in (('simRMLPos',allocate),('simRMLStep',advance),('simRMLRemove',remove)):
            stack.enter_context(patch.object(sim,name,function))
        for obj,name in ((env.arm,'set_joint_target_positions'),(env.arm,'set_joint_positions'),
                (env.pyrep,'step'),(env,'_tick'),(env.task,'step')):
            stack.enter_context(patch.object(obj,name,forbid(name)))
        try:returned=callback(sim)
        except Exception as failure:error=dict(type=type(failure).__name__,message=str(failure),traceback=traceback.format_exc())
        finally:
            # Track original leaked local handles as well as a returned live
            # execution handle. No global remove-all or native file edit.
            outstanding_before_cleanup=sorted(map(int,active))
            for handle in list(active):
                try:remove(handle)
                except Exception as failure:cleanup_errors.append(dict(handle=int(handle),error=repr(failure)))
    after=snapshot(env);n=path._num_joints;points=np.asarray(path._path_points,float);lengths=path._get_path_point_lengths()
    dt=float(sim.simGetSimulationTimeStep());limits=np.asarray(env.arm.get_joint_upper_velocity_limits());carried=points[:n].copy();summaries=[]
    for record in records.values():
        previous=points[:n].copy();genuine=0.;carry_ratio=0.;first_genuine=None;first_carried=None
        for state,pva in record['samples']:
            position=pva[0]
            for i in range(len(lengths)-1):
                if lengths[i]<=position<=lengths[i+1]:
                    t=(position-lengths[i])/(lengths[i+1]-lengths[i]);q=points[n*i:n*(i+1)]+(points[n*(i+1):n*(i+2)]-points[n*i:n*(i+1)])*t
                    actual=float(np.max(np.abs((q-previous)/dt)/limits));old=float(np.max(np.abs((q-carried)/dt)/limits))
                    genuine=max(genuine,actual);carry_ratio=max(carry_ratio,old)
                    if first_genuine is None:first_genuine=actual;first_carried=old
                    previous=q;carried=q;break
        summaries.append(dict(handle=record['handle'],max_velocity_acceleration_jerk=record['max_velocity_acceleration_jerk'],
            virtual_steps=len(record['samples']),removed=record['removed'],
            maximum_genuine_adjacent_joint_velocity_ratio=genuine,
            author_cross_probe_carried_previous_q_max_ratio=carry_ratio,
            first_genuine_adjacent_ratio=first_genuine,first_author_cross_probe_ratio=first_carried,
            final_RML_sample=record['samples'][-1] if record['samples'] else None))
    return dict(before=before,after=after,delta=delta(before,after),exception=error,
        returned_live_handle=int(returned) if returned is not None else None,
        originally_unremoved_handles_explicitly_cleaned=outstanding_before_cleanup,
        all_created_RML_handles_removed=not active and not cleanup_errors,cleanup_errors=cleanup_errors,
        forbidden_physics_or_arm_setter_attempts=forbidden,virtual_probe_summaries=summaries)


def prepare(source_plan,destination):
    source_plan=Path(source_plan).resolve();plan=json.loads(source_plan.read_text())
    case=next(c for c in plan['cases'] if c['case_id']=='70a9c9f68520d04b')
    require(case['parent_source_group']=='train:beat_the_rotating_buzz:episode123' and case['teacher_used'] is True,'fixed failed TRAIN source differs')
    helper=ROOT/'our_mani/multitask/rml_path_repair.py'
    failure=ROOT/'artifacts/framework/multitask_r10/TRAIN_full_collection_v11/infrastructure_attempts'/case['case_id']/'0/parent_receipt.json'
    require(json.loads(failure.read_text())['status']=='infrastructure_error','fixed source is not an observed infrastructure failure')
    inputs=dict(plan['inputs']);inputs[str(source_plan)]=file_sha(source_plan);inputs[str(helper)]=file_sha(helper)
    inputs[str(Path(__file__).resolve())]=file_sha(__file__);inputs[str(failure)]=file_sha(failure)
    value=dict(version='r10_fixed_single_source_virtual_RML_native_diagnostic_v1',source_case=case,
        source_plan=dict(path=str(source_plan),sha256=file_sha(source_plan),internal_sha256=plan['plan_sha256']),
        protocol=plan['protocol'],inputs_sha256=inputs,independent_native_processes=1,
        original_then_repaired_same_path=True,physical_execution_permitted=False,
        dataset_or_fitting_input=False,no_parameter_search_or_source_expansion=True)
    value['plan_sha256']=digest(value);atomic_json(destination,value)
    print(json.dumps(dict(plan=str(destination),sha256=file_sha(destination),case_id=case['case_id'])))


def run(plan_path,out):
    plan=json.loads(Path(plan_path).read_text());require(plan['plan_sha256']==digest({k:v for k,v in plan.items() if k!='plan_sha256'}),'diagnostic plan fingerprint differs')
    for path,checksum in plan['inputs_sha256'].items():require(file_sha(path)==checksum,'diagnostic input/source changed: '+path)
    from our_mani.multitask.simulation import AlignedDevelopmentSimulation
    from our_mani.multitask.train_teacher import native_path_planner
    from our_mani.multitask.rml_path_repair import NativeRMLHandleFactory
    from our_mani.contracts import Pose
    from scripts.collect_multitask_training_r10 import native_teacher
    case=plan['source_case'];env=None;report=dict(version=plan['version'],complete=False,diagnostic_plan_sha256=plan['plan_sha256'],
        plan_file_sha256=file_sha(plan_path),case_id=case['case_id'],parent_source_group=case['parent_source_group'],
        independent_native_processes=1,dataset_or_fitting_input=False,core_or_external_files_modified=False)
    try:
        env=AlignedDevelopmentSimulation(case['task'],case['source']['path'],source=case['source'],split='train',
            protocol=plan['protocol'],seed=plan['protocol'].get('seed',0),replica=case['replica'],
            servo_lead=plan['protocol'].get('local_servo_lead',1.4))
        env.collection_variation=int(case['source']['variation']);env.reset_development(variation=env.collection_variation)
        report['reset_receipt']=env.reset_receipt;report['after_reset']=snapshot(env)
        label=native_teacher(env)();report['teacher_label']={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in label.items()}
        owned=native_path_planner(env)(Pose.from_xyzw(label['tcp_xyzw']));path=owned.path
        report['after_pure_arm_planning']=snapshot(env);report['pure_planning_delta']=delta(report['after_reset'],report['after_pure_arm_planning'])
        report['native_joint_path']=dict(points=np.asarray(path._path_points).tolist(),
            points_sha256=digest(np.asarray(path._path_points).tolist()),arc_lengths=path._get_path_point_lengths())
        report['original']=tracked_probe(env,path,lambda backend:path._get_rml_handle())
        factory=NativeRMLHandleFactory()
        def repaired(backend):factory.sim_backend=backend;return factory(path)
        report['repaired']=tracked_probe(env,path,repaired);report['repair_receipt']=factory.last_receipt
        report['final_snapshot']=snapshot(env);report['whole_diagnostic_delta']=delta(report['after_reset'],report['final_snapshot'])
        report['original_failure_reproduced']=report['original']['exception'] is not None
        report['repaired_handle_creation_succeeded']=report['repaired']['exception'] is None and factory.last_receipt['complete']
        require(all(not value for value in report['whole_diagnostic_delta'].values()),'native diagnostic altered physical clocks/robot state')
        require(all(report[name]['all_created_RML_handles_removed'] and not report[name]['forbidden_physics_or_arm_setter_attempts'] for name in ('original','repaired')),
            'native diagnostic cleanup or no-physics proof failed')
        report['complete']=True
    except Exception as error:report['diagnostic_error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        if env is not None:
            try:env.close();report['native_cleanup_complete']=True
            except Exception as error:report['native_cleanup_complete']=False;report['cleanup_error']=repr(error)
    report['all_inputs_SHA_rechecked_at_end']=all(file_sha(path)==sha for path,sha in plan['inputs_sha256'].items())
    atomic_json(out,report)
    print(json.dumps({k:report.get(k) for k in ('complete','original_failure_reproduced','repaired_handle_creation_succeeded','whole_diagnostic_delta','native_cleanup_complete','diagnostic_error')},ensure_ascii=False))
    if not report['complete']:raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--source-plan');parser.add_argument('--plan',required=True);parser.add_argument('--out')
    args=parser.parse_args()
    if args.prepare:prepare(args.source_plan,Path(args.plan))
    else:run(args.plan,Path(args.out))


if __name__=='__main__':main()
