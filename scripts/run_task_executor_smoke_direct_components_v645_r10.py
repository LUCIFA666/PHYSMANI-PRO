"""Fixed TRAIN core14 skills on the user-authorized common author RML profile."""
from copy import deepcopy
from pathlib import Path
import json
import os
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import importlib
from scripts import run_task_executor_smoke_button_reference_v388_r10 as button_source
from our_mani.multitask.task_executor_registry import ADAPTERS
from our_mani.multitask.specs import TASK_SPECS
from scripts import collect_multitask_recovery_r10 as recovery
from our_mani.multitask.rml_component_domain_v640 import for_train
from our_mani.multitask.rml_component_factory_v643 import create_actor,cpu_preflight
from our_mani.multitask.rml_policy_components_v646 import construct,binding as component_binding,SOURCES as COMPONENT_SOURCES
from our_mani.multitask.evaluation import digest
from scripts import run_task_executor_smoke_joint16_v174_r10 as data_source
from our_mani.multitask.physmani_skill_executor_v534 import PhysManiSkillExecutor
from our_mani.multitask.common_rml_protocol_v530 import binding as profile_binding,attest,fingerprint
from our_mani.multitask.physmani_geometry_executor_v517 import PhysManiGeometryExecutor,binding,SOURCES
from our_mani.multitask.physmani_geometry_policy_v518 import PublicModelFrontend,ButtonGeometryIntents
from our_mani.multitask.native_window_observer_v191 import CountedWindowObserver
from our_mani.multitask.evaluation import atomic_json,file_sha,native_completion_record
from our_mani.multitask.frozen_subset_batch_v132 import checked_subset


EXTRA_SOURCES=('our_mani/multitask/physmani_geometry_policy_v518.py',
    'our_mani/multitask/physmani_geometry_goals_v529.py',
    'our_mani/multitask/physmani_skill_policy_v548.py',
    'our_mani/multitask/physmani_skill_executor_v534.py',
    'our_mani/multitask/common_rml_protocol_v530.py',
    'our_mani/multitask/native_window_observer_v191.py',
    'our_mani/multitask/task_executor_registry.py',
    'scripts/run_task_executor_smoke_direct_components_v645_r10.py')


def components(task):
    if task not in TASK_SPECS:raise ValueError('explicit registered14 task required')
    if task.startswith('push_moving_button'):
        module='button_intent_native_v91';symbol='ButtonIntentNativeBackend'
        # The current387 layer supplies the actual trained button model.
        module='button_reference_native_v387'
        driver='run_task_executor_smoke_button_reference_v388_r10'
    elif task=='put_rubbish_in_moving_bin':
        module='bin_physical_native_v235';symbol='SharedJoint16NativeBackend'
        driver='run_task_executor_smoke_bin_physical_v236_r10'
    else:
        r=ADAPTERS[task];module,symbol,driver=r.backend_module,r.backend_class,r.driver
    return (importlib.import_module('scripts.'+driver),
            getattr(importlib.import_module('our_mani.multitask.'+module),symbol)())


def preflight(args):
    if (args.split!='train' or not args.development_demo or args.method not in ('cv_geometry','context_geometry')
        or args.replica!=0 or args.exploration_seed is not None or args.record_demo or args.demo_only):
        raise ValueError('explicit fixed original TRAIN qualification, no videos/exploration')
    p=data_source.evaluator.preflight(args)
    source,values=data_source.verify_development_source(p)
    p['shared_hook_plan']=data_source.hook_plan_for(p)
    if 'development_demo_only' not in p['out'].parts:raise ValueError('explicit TRAIN-only output namespace')
    expected=binding(ROOT,p['config']['baseline_repo'])
    profile=profile_binding(ROOT,p['config']['baseline_repo'])
    if p['config'].get('physmani_geometry_executor_v517')!=expected or p['config'].get('common_author_RML_profile')!=profile:
        raise ValueError('frozen common author RML and unchanged actual native source required')
    actual={str(ROOT/n):file_sha(ROOT/n) for n in (*EXTRA_SOURCES,*COMPONENT_SOURCES,'scripts/run_task_executor_smoke_direct_components_v645_r10.py')}
    actual.update(expected['native_sources_sha256']);checked_subset(ROOT,p['files'],actual)
    contract=component_binding(ROOT,p['config'],args.task)
    if p['config'].get('direct_RML_policy_components_v646',{}).get(args.task)!=contract:
        raise ValueError('freeze actual direct frontend/intent/executor routes')
    domain=for_train(ROOT,p['files'],args.task,args.raw_episode_id,args.method)
    p['model_preflight']=cpu_preflight(ROOT,p['config'],args.task,args.method,domain,p['files'])
    p['component_domain']=domain;p['record_demo']=False
    identity=dict(p['identity'],demo_only=True,development_demo_only=True,fitting_source=False,formal_validation_metric=False)
    identity.pop('sha256',None);identity['sha256']=digest(identity);p['identity']=identity
    p['args']=deepcopy(args);p['args'].demo_only=True
    p['RML_adaptation_binding']=expected;p['common_author_RML_profile']=profile
    p['component_source_driver']='direct_public_RML_component_factory_v643'
    return p


def run(p):
    args,config,out=p['args'],p['config'],p['out']
    source=data_source;backend=data_source.evaluator.NativeBackend()
    if out.exists():
        raise FileExistsError('preserve every completed or partial RML pilot')
    out.mkdir(parents=True)
    atomic_json(out/'inputs.json',dict(identity=p['identity'],model_preflight=p['model_preflight'],
        executor=p['RML_adaptation_binding'],common_author_RML_profile=p['common_author_RML_profile'],files=p['files']))
    import numpy as np
    import random
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
    captures=dict(simulations=[],paths=[])
    env=None;started=time.perf_counter();record=None;failure=None
    try:
        with recovery.scoped_owned_hooks(p['shared_hook_plan'],captures):
            source.evaluator.verify_preflight(p)
            env,descriptions=backend.environment(p)
            common_parameters=attest(env,config,p['common_author_RML_profile'])
            window=CountedWindowObserver(env)
            # The established backend supplies trained perception/forecast/
            # role/library components. Its act(), step() and dispatcher are
            # not used by this new execution experiment.
            actor,component_identity=create_actor(ROOT,env,config,args.method,descriptions,domain=p['component_domain'])
            frontend,policy,executor=construct(ROOT,config,args.task,actor)
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
                    execution=executor.execute(env,action)
                    if hasattr(policy,'observe_owned_RML_path_result'):
                        modification=execution.get('waypoint_modification') or {}
                        policy.observe_owned_RML_path_result(dict(error=execution.get('error'),waypoint_modification={
                            k:modification[k] for k in ('selected_waypoints','cartesian_spacing_m') if k in modification}))
                    if not before<env.ledger.ticks<=min(before+10,200):
                        raise ValueError('original PhysMani action must use1..10 counted ticks')
            alignment=env.alignment_receipt()
            atomic_json(out/'raw_execution_diagnostics.json',source.evaluator.clean(dict(
                alignment_receipt=alignment,initial_conditions=initial,policy=policy.receipt(),
                executor=executor.geometry_receipt(),holds=holds,formal_VAL_TEST=False)))
            record=native_completion_record(p['identity'],process_id=os.getpid(),
                success=bool(env._native_success_latched),fatal=bool(env._native_failure_latched),
                ticks=int(env.ledger.ticks),initial_conditions=initial,alignment_receipt=alignment,
                policy_failure=None)
            record.update(version='r10_TRAIN_trained_public_skills_on_common_author_RML_v645',
                method=args.method,split='train',source_raw_episode_id=args.raw_episode_id,
                parent_raw_episode_id=args.raw_episode_id,
                demo_only=True,development_demo_only=True,formal_validation_metric=False,formal_TEST_metric=False,
                formal_test_used=False,TEST_used_for_fitting=False,automatic_retry=False,
                original_source_motion_replayed=False,shared_context_checkpoint=config['context_checkpoint'],
                interaction_library_sha256=actor.controller.library.sha256,
                RML_adaptation=p['RML_adaptation_binding'],trained_component_identity=component_identity,
                common_author_RML_parameters=common_parameters,
                component_source_driver=p['component_source_driver'],
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
    args=data_source.parser().parse_args();p=preflight(args)
    if args.check:
        print(json.dumps(dict(complete=True,native_started=False,GPU_started=False,
            identity=p['identity'],RML_adaptation=p['RML_adaptation_binding'],
            common_RML_profile_sha256=fingerprint(p['common_author_RML_profile']))))
        return
    r=run(p)
    print(json.dumps(dict(complete=True,native_success=r['official_task_success'],
                          case=str(p['out']/'case.json'))))


if __name__=='__main__':main()
