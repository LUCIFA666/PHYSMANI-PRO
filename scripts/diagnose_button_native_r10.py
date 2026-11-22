"""TRAIN-only button native staging; diagnostic outputs never become metrics.

Runs two tasks sequentially, one isolated native child at a time. Each call is
flushed/fsynced before/after so a C++ abort retains its actual last stage. The
owned child wraps mesh reads/creation for logging only; frozen files stay intact.
"""
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
BASE=ROOT/'artifacts/framework/multitask_r10/diagnostics'
TASKS=('push_moving_button','push_moving_button_high_speed')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value,dict): return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [serial(v) for v in value]
    if hasattr(value,'tolist'): return value.tolist()
    return value


def write(path,value):
    path=Path(path).resolve()
    assert path.is_relative_to(BASE.resolve()) and not path.exists()
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(serial(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def child(plan_path,task,out):
    faulthandler.enable(all_threads=True)
    from audit_train_resets_r10 import require_completed_delivery
    from collect_multitask_training_r10 import verify_plan,train_color_annotations,native_teacher
    gate=require_completed_delivery();plan=verify_plan(plan_path)
    rows=[r for r in plan['cases'] if r['task']==task and r['split']=='train' and r['replica']==0]
    assert len(rows)==1 and task in TASKS
    row=rows[0];out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    journal=out/'stages.jsonl'
    def event(stage,state,**details):
        item=dict(stage=stage,state=state,utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**serial(details))
        line=json.dumps(item,ensure_ascii=False,allow_nan=False)
        with journal.open('a') as stream: stream.write(line+'\n');stream.flush();os.fsync(stream.fileno())
        print(line,flush=True)
    def step(name,fn):
        event(name,'begin');result=fn();event(name,'complete');return result
    write(out/'inputs.json',dict(diagnostic_only=True,TRAIN_only=True,formal_metrics_included=False,
        source=row,plan_sha256=sha(plan_path),entrypoint_sha256=sha(__file__),release_gate=gate,
        instrumentation='owned-process read-only mesh get/create logging; no source modifications'))
    env=None;old_mesh=None;old_factory=None
    try:
        from our_mani.multitask.simulation import AlignedDevelopmentSimulation
        from our_mani.multitask.training_geometry import (export_train_geometry,declared_instance_bindings,
            body_templates_from_asset,register_collision_asset)
        from our_mani.multitask.perception import MultiBodySE3Tracker
        from pyrep.objects.shape import Shape
        old_mesh=Shape.get_mesh_data
        def traced_mesh(shape):
            event('native_mesh_read','begin',name=shape.get_name(),handle=shape.get_handle())
            result=old_mesh(shape)
            event('native_mesh_read','complete',name=shape.get_name(),vertices_shape=list(result[0].shape),
                  triangles_shape=list(result[1].shape))
            return result
        Shape.get_mesh_data=traced_mesh
        env=step('environment_construct',lambda:AlignedDevelopmentSimulation(task,row['source']['path'],
            source=row['source'],split='train',protocol=plan['protocol'],seed=plan['protocol'].get('seed',0),
            replica=0,servo_lead=plan['protocol'].get('local_servo_lead',1.4)))
        env.collection_variation=int(row['source']['variation'])
        step('reset',lambda:env.reset_development(variation=env.collection_variation))
        observation,robot=step('observe',env.observe)
        asset=step('export_TRAIN_geometry',lambda:export_train_geometry(env,dict(row['source'],split='train')))
        write(out/'TRAIN_geometry.json',asset)
        bindings=step('CAD_bind',lambda:declared_instance_bindings(env,asset,shared_permission=True))
        color_observation,_=step('color_observe',env.observe)
        colors=step('color_annotations',lambda:train_color_annotations(env,color_observation,bindings,row['parent_source_group']))
        write(out/'TRAIN_color_annotations.json',colors)
        templates=step('templates',lambda:body_templates_from_asset(asset,bindings,
            orientation_observability=plan['protocol']['body_orientation_observability'][task],
            sphere_radii=plan['protocol'].get('body_sphere_radii',{}).get(task)))
        tracker=step('tracker_construct',lambda:MultiBodySE3Tracker(templates,task_spec=env.spec))
        old_factory=env.map_service._ghost_factory
        def traced_factory(vertices,triangles):
            event('native_mesh_create','begin',vertex_scalar_count=len(vertices),index_scalar_count=len(triangles))
            result=old_factory(vertices,triangles)
            event('native_mesh_create','complete',handle=result.get_handle())
            return result
        env.map_service._ghost_factory=traced_factory
        step('register_collision',lambda:register_collision_asset(env.map_service,asset))
        observation,robot=step('tracker_input_observe',env.observe)
        states,metadata=step('tracker_observe',lambda:tracker.observe(observation))
        teacher=step('nativeTeacher_construct',lambda:native_teacher(env))
        label=step('nativeTeacher_read',teacher)
        assert env.ledger.ticks==0 and env.time==0
        write(out/'read_result.json',dict(diagnostic_only=True,completed=True,task=task,
            counted_actions=0,tracker_ids=[s.object_id for s in states],teacher_label=serial(label),alignment_receipt=env.alignment_receipt()))
        event('cleanup','begin');env.close();env=None;event('cleanup','complete')
    except Exception as error:
        event('python_exception','failed',error=repr(error))
        raise
    finally:
        if old_mesh is not None: Shape.get_mesh_data=old_mesh
        if env is not None:
            if old_factory is not None: env.map_service._ghost_factory=old_factory
            event('cleanup_finally','begin');env.close();event('cleanup_finally','complete')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',default='artifacts/framework/multitask_r10/TRAIN_smoke_collection_plan_v3.json')
    parser.add_argument('--out',required=True)
    parser.add_argument('--task',choices=TASKS)
    parser.add_argument('--child',action='store_true')
    args=parser.parse_args();plan=(ROOT/args.plan).resolve();out=(ROOT/args.out).resolve()
    assert out.is_relative_to(BASE.resolve())
    if args.child:
        assert args.task;child(plan,args.task,out);return
    from audit_train_resets_r10 import require_completed_delivery
    require_completed_delivery()
    assert not out.exists();out.mkdir(parents=True)
    results=[]
    for task in (TASKS if args.task is None else (args.task,)):
        case=out/task;log=out/(task+'.native.log')
        command=['bash','scripts/run_local_sim.sh',str(Path(__file__).resolve().relative_to(ROOT)),
            '--child','--plan',str(plan),'--task',task,'--out',str(case)]
        with log.open('w') as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',
                PYTHONFAULTHANDLER='1'),stdout=stream,stderr=subprocess.STDOUT)
            code=process.wait()
        lines=(case/'stages.jsonl').read_text().splitlines() if (case/'stages.jsonl').exists() else []
        receipt=dict(task=task,diagnostic_only=True,formal_metrics_included=False,exit_code=code,
            command=command,launcher_pid=process.pid,log_sha256=sha(log),entrypoint_sha256=sha(__file__),
            last_stage=json.loads(lines[-1]) if lines else None,maximum_concurrent_native_children=1)
        write(out/(task+'.process_receipt.json'),receipt);results.append(receipt)
    write(out/'report.json',dict(diagnostic_only=True,TRAIN_only=True,results=results,
        formal_metrics_included=False,thresholds_changed=False,frozen_sources_changed=False))
    print(json.dumps(serial(results),ensure_ascii=False))


if __name__=='__main__': main()
