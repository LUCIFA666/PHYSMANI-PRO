"""Fixed TRAIN buttons: CV/Context pose intents on the original PhysMani RML."""
from copy import deepcopy
from pathlib import Path
import json
import os
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scripts import run_task_executor_smoke_button_reference_v388_r10 as source
from scripts import collect_multitask_recovery_r10 as recovery
from our_mani.multitask.button_corridor_executor_native import ButtonCorridorBackend,verify as source_verify
from our_mani.multitask.physmani_geometry_executor_v517 import PhysManiGeometryExecutor,binding,SOURCES
from our_mani.multitask.physmani_geometry_policy_v518 import PublicModelFrontend,ButtonGeometryIntents
from our_mani.multitask.native_window_observer_v191 import CountedWindowObserver
from our_mani.multitask.evaluation import atomic_json,file_sha,native_completion_record
from our_mani.multitask.frozen_subset_batch_v132 import checked_subset


def preflight(args):
    if args.split!='train' or not args.development_demo or args.method not in ('cv_geometry','context_geometry'):
        raise ValueError('new RML adaptation is fixed original TRAIN development only')
    p=source.preflight(args)
    source_verify(ROOT,p)
    expected=binding(ROOT,p['config']['baseline_repo'])
    if p['config'].get('physmani_geometry_executor_v517')!=expected:
        raise ValueError('freeze original PhysMani RML and explicit geometry adaptation')
    files={str(ROOT/name):sha for name,sha in expected['sources_sha256'].items()}
    files.update(expected['native_sources_sha256'])
    for name in ('our_mani/multitask/physmani_geometry_policy_v518.py',
                 'our_mani/multitask/native_window_observer_v191.py',
                 'scripts/run_task_executor_smoke_physmani_geometry_v519_r10.py'):
        files[str(ROOT/name)]=file_sha(ROOT/name)
    checked_subset(ROOT,p['files'],files)
    p['RML_adaptation_binding']=expected
    return p


def run(p):
    args,config,out=p['args'],p['config'],p['out']
    if out.exists():
        raise FileExistsError('preserve every completed or partial RML pilot')
    out.mkdir(parents=True)
    atomic_json(out/'inputs.json',dict(identity=p['identity'],model_preflight=p['model_preflight'],
        executor=p['RML_adaptation_binding'],files=p['files']))
    import numpy as np
    import random
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
    backend=ButtonCorridorBackend()
    captures=dict(simulations=[],paths=[])
    env=None;started=time.perf_counter();record=None;failure=None
    try:
        with recovery.scoped_owned_hooks(p['shared_hook_plan'],captures):
            source.evaluator.verify_preflight(p)
            env,descriptions=backend.environment(p)
            window=CountedWindowObserver(env)
            # The established backend supplies trained perception/forecast/
            # role/library components. Its act(), step() and dispatcher are
            # not used by this new execution experiment.
            actor,component_identity=backend.actor(p,env,descriptions)
            frontend=PublicModelFrontend(actor)
            policy=ButtonGeometryIntents(frontend)
            executor=PhysManiGeometryExecutor(config,observe_public_frame=frontend.observe,
                admit_pose_action=policy.admit_pose,admit_joint_target=policy.admit_joint)
            observation,robot=env.observe()
            initial=backend.initial_conditions(env,observation,robot)
            frontend.observe(observation,robot)
            holds=[]
            while env.ledger.remaining>=env.ledger.dt-1e-8 and not env._native_terminal_latched:
                if (ROOT/'runtime/framework/multitask_r10_STOP').exists():
                    raise InterruptedError('controlled project stop')
                action=policy.select()
                if action is None:
                    holds.append(dict(time=float(env.time),reason='public_geometry_history_or_intent_unavailable'))
                    actor.command_segments.issue('hold')
                    env.hold_tick()
                    if not env._native_terminal_latched:
                        observation,robot=env.observe();frontend.observe(observation,robot)
                else:
                    before=int(env.ledger.ticks)
                    executor.execute(env,action)
                    if not before<env.ledger.ticks<=min(before+10,200):
                        raise ValueError('original PhysMani action must use1..10 counted ticks')
            alignment=env.alignment_receipt()
            record=native_completion_record(p['identity'],process_id=os.getpid(),
                success=bool(env._native_success_latched),fatal=bool(env._native_failure_latched),
                ticks=int(env.ledger.ticks),initial_conditions=initial,alignment_receipt=alignment,
                policy_failure=None)
            record.update(version='r10_TRAIN_geometry_on_original_PhysMani_RML_v519',
                method=args.method,split='train',source_raw_episode_id=args.raw_episode_id,
                parent_raw_episode_id=args.raw_episode_id,
                demo_only=True,development_demo_only=True,formal_validation_metric=False,formal_TEST_metric=False,
                formal_test_used=False,TEST_used_for_fitting=False,automatic_retry=False,
                original_source_motion_replayed=False,shared_context_checkpoint=config['context_checkpoint'],
                interaction_library_sha256=actor.controller.library.sha256,
                RML_adaptation=p['RML_adaptation_binding'],trained_component_identity=component_identity,
                actor_metrics=dict(predictor=actor.predictor,ranking='geometry',
                    physmani_geometry_executor=executor.geometry_receipt(),geometry_intents=policy.receipt(),
                    holds=holds,old_reference_managers_executed=False),
                wall_seconds=time.perf_counter()-started,simulation_seconds=float(env.time))
            env.close();env=None
        source.evaluator.verify_preflight(p)
    except BaseException as error:
        failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
                     counted_as_policy_failure=False,automatic_retry=False)
        raise
    finally:
        if env is not None:
            try:env.close()
            except Exception as error:
                if failure is not None:failure['cleanup_error']=repr(error)
                else:raise
        hooks=dict(reset=[deepcopy(getattr(e,'reset_recovery_receipt',None)) for e in captures['simulations']],
                   paths=[deepcopy(x.recovery_receipt) for x in captures['paths']])
        atomic_json(out/'owned_hook_receipt.json',source.evaluator.clean(hooks))
        if failure is not None:atomic_json(out/'infrastructure_error.json',failure)
    if len(hooks['reset'])!=1 or not hooks['reset'][0].get('complete') or any(not x.get('close_complete') for x in hooks['paths']):
        atomic_json(out/'infrastructure_error.json',dict(type='IncompleteOwnedHookReceipt',
            message='original physical reset/RML cleanup did not complete',counted_as_policy_failure=False))
        raise ValueError('original physical reset/RML cleanup did not complete')
    record['cleanup_completed']=True
    atomic_json(out/'case.json',source.evaluator.clean(record))
    return record


def main():
    args=source.parser().parse_args();p=preflight(args)
    if args.check:
        print(json.dumps(dict(complete=True,native_started=False,GPU_started=False,
            identity=p['identity'],RML_adaptation=p['RML_adaptation_binding'])))
        return
    r=run(p)
    print(json.dumps(dict(complete=True,native_success=r['official_task_success'],
                          case=str(p['out']/'case.json'))))


if __name__=='__main__':main()
