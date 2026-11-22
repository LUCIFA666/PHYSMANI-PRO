"""Fixed TRAIN moving blocks through the common author RML and public models."""
from pathlib import Path
import argparse
import json
import os
import random
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scripts import run_task_executor_smoke_block_v317_r10 as source
from our_mani.multitask.evaluation import atomic_json,file_sha,case_identity,native_completion_record
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.physmani_geometry_executor_v517 import binding
from our_mani.multitask.common_rml_protocol_v530 import binding as common_binding,attest
from our_mani.multitask.physmani_block_policy_v538 import PublicBlockCollision,BlockPublicFrontend,BlockGeometryIntents
from our_mani.multitask.physmani_block_receipt_v541 import ReceiptBlockView
from our_mani.multitask.physmani_skill_executor_v534 import PhysManiSkillExecutor
from our_mani.multitask.baseline_measured_windows_v345 import CountedWindowObserver

SOURCES=('our_mani/multitask/physmani_block_policy_v538.py',
    'our_mani/multitask/physmani_skill_executor_v534.py',
    'our_mani/multitask/physmani_skill_policy_v533.py',
    'our_mani/multitask/physmani_geometry_goals_v529.py',
    'our_mani/multitask/physmani_geometry_policy_v518.py',
    'our_mani/multitask/common_rml_protocol_v530.py',
    'scripts/run_task_executor_smoke_physmani_blocks_v542_r10.py',
    'our_mani/multitask/physmani_block_receipt_v541.py')


def preflight(args):
    p=source.preflight(args);config=p['config'];files=p['files']
    expected=binding(ROOT,config['baseline_repo']);profile=common_binding(ROOT,config['baseline_repo'])
    if config.get('physmani_geometry_executor_v517')!=expected or config.get('common_author_RML_profile')!=profile:
        raise ValueError('freeze unchanged author RML and the explicit public block adaptation')
    dependencies={str(ROOT/n):s for n,s in expected['sources_sha256'].items()}
    dependencies.update(expected['native_sources_sha256'])
    dependencies.update({str(Path(config['baseline_repo'])/n):s for n,s in profile['native_sources_sha256'].items()})
    dependencies.update({str(ROOT/n):file_sha(ROOT/n) for n in SOURCES})
    for name,sha in dependencies.items():
        if files.get(name)!=sha:raise ValueError('public block/common-RML input missing or changed: '+name)
    p['component_model_preflight_identity']=p['identity']
    p['identity']=case_identity(split='train',task=args.task,raw_episode_id=args.raw_episode_id+'__policy_aug_r0',
        parent_raw_episode_id=args.raw_episode_id,method=args.method,source_row=p['row'],
        config_sha256=file_sha(ROOT/args.config),inputs_sha256=file_sha(ROOT/args.frozen_inputs),demo_only=True)
    return p


def run(p):
    import numpy as np
    import torch
    from our_mani.sensor_alignment.author_rgbd import AlignedLocalSimulation,public_initial_conditions
    from scripts.collect_local_rollouts import initialization
    from scripts.evaluate_aligned_r09 import clean
    args,c,row,out=(p[k] for k in ('args','config','row','out'))
    if out.exists():raise FileExistsError('preserve every block RML attempt')
    out.mkdir(parents=True);atomic_json(out/'inputs.json',dict(identity=p['identity'],
        component_model_preflight_identity=p['component_model_preflight_identity'],files=p['files']))
    torch.set_num_threads(4);torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed'])
    env=None;record=None;started=time.perf_counter()
    try:
        raw=AlignedLocalSimulation(args.task,row['path'],protocol=c,seed=c['seed'],servo_lead=1.)
        env=ReceiptBlockView(raw)
        x,v,a,goal=initialization(row['path'])
        if not env.reset(target_position=x,target_velocity=v,target_acceleration=a,goal_position=goal):
            raise RuntimeError('original block development reset IK failed')
        env.finish_home_reset()
        parameters=attest(env,c,c['common_author_RML_profile']);window=CountedWindowObserver(env)
        collision=PublicBlockCollision(env)
        frontend=BlockPublicFrontend(ROOT,c,args.task,args.method,env.robot_model,collision)
        policy=BlockGeometryIntents(frontend)
        executor=PhysManiSkillExecutor(c,skill_policy=policy,observe_public_frame=frontend.observe,
            admit_pose_action=policy.admit_pose,admit_joint_target=policy.admit_joint)
        observation,robot=env.observe();initial=public_initial_conditions(env,observation,robot)
        frontend.observe(observation,robot);holds=[]
        while env.ledger.remaining>=env.ledger.dt-1e-8 and not env._native_terminal_latched:
            if (ROOT/'runtime/framework/multitask_r10_STOP').exists():raise InterruptedError('controlled project stop')
            action=policy.select()
            if action is None:
                holds.append(dict(time=float(env.time),reason='public_history_goal_or_skill_not_ready'))
                env.hold_tick()
                if not env._native_terminal_latched:
                    observation,robot=env.observe();frontend.observe(observation,robot)
            else:
                before=env.ledger.ticks;executor.execute(env,action)
                if not before<env.ledger.ticks<=min(before+10,200):raise ValueError('author RML decision counted-tick mismatch')
        atomic_json(out/'raw_execution_diagnostics.json',clean(dict(counted_clock=env.ledger.receipt(),
            alignment_receipt=env.alignment_receipt(),policy=policy.receipt(),
            executor=executor.geometry_receipt(),holds=holds,formal_VAL_TEST=False)))
        record=native_completion_record(p['identity'],process_id=os.getpid(),success=bool(env._native_success_latched),
            fatal=False,ticks=int(env.ledger.ticks),initial_conditions=initial,alignment_receipt=env.alignment_receipt())
        record.update(version='r10_public_blocks_on_common_author_RML_v542',
            source_raw_episode_id=args.raw_episode_id,development_demo_only=True,
            formal_validation_metric=False,formal_TEST_metric=False,automatic_retry=False,
            common_author_RML_parameters=parameters,all_stage_counted_joint_motion=window.receipt(),
            shared_context_checkpoint=c['context_checkpoint'],actor_metrics=dict(
                predictor=args.method.split('_')[0],ranking='geometry',
                physmani_geometry_executor=executor.geometry_receipt(),geometry_intents=policy.receipt(),holds=holds,
                old_reference_managers_executed=False),
            wall_seconds=time.perf_counter()-started,simulation_seconds=float(env.time),
            historical_R09_results_modified=False)
        env.close();env=None;verify_inputs(ROOT,p['files'])
    except BaseException as error:
        atomic_json(out/'infrastructure_error.json',dict(type=type(error).__name__,message=str(error),
            traceback=traceback.format_exc(),counted_as_policy_failure=False,automatic_retry=False))
        raise
    finally:
        if env is not None:env.close()
    record['cleanup_completed']=True;atomic_json(out/'case.json',clean(record));return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('config','manifest','frozen-inputs','task','raw-episode-id','out','method','split'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--development-demo',action='store_true');parser.add_argument('--check',action='store_true')
    args=parser.parse_args();p=preflight(args)
    if args.check:
        print(json.dumps(dict(complete=True,native_started=False,GPU_started=False,identity=p['identity'],
            shared_context_checkpoint=p['config']['context_checkpoint'],model_preflight=p['model_preflight'])))
        return
    r=run(p);print(json.dumps(dict(complete=True,native_success=r['official_task_success'],case=str(p['out']/'case.json'))))


if __name__=='__main__':main()
