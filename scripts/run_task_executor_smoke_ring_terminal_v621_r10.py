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
from our_mani.multitask.physmani_ring_compensation_v580 import binding as compensation_binding
from our_mani.multitask.physmani_ring_release_v590 import binding as release_binding
from our_mani.multitask.physmani_terminal_opening_v620 import AuthorRingReleaseIntents,binding as terminal_binding
from our_mani.multitask.physmani_skill_executor_v534 import PhysManiSkillExecutor
from our_mani.multitask.common_rml_protocol_v530 import binding as profile_binding,attest,fingerprint
from our_mani.multitask.physmani_geometry_executor_v517 import PhysManiGeometryExecutor,binding,SOURCES
from our_mani.multitask.physmani_geometry_policy_v518 import ButtonGeometryIntents
from our_mani.multitask.physmani_ring_frontend_v570 import PublicRingFrontend as PublicModelFrontend,binding as gauge_binding
from our_mani.multitask.native_window_observer_v191 import CountedWindowObserver
from our_mani.multitask.evaluation import atomic_json,file_sha,native_completion_record
from our_mani.multitask.frozen_subset_batch_v132 import checked_subset


EXTRA_SOURCES=('our_mani/multitask/physmani_geometry_policy_v518.py',
    'our_mani/multitask/physmani_geometry_goals_v529.py',
    'our_mani/multitask/physmani_skill_policy_v533.py',
    'our_mani/multitask/physmani_skill_executor_v534.py',
    'our_mani/multitask/common_rml_protocol_v530.py',
    'our_mani/multitask/native_window_observer_v191.py',
    'our_mani/multitask/task_executor_registry.py',
    'scripts/run_task_executor_smoke_ring_terminal_v621_r10.py',
    'our_mani/multitask/physmani_ring_frontend_v570.py',
    'our_mani/multitask/physmani_ring_compensation_v580.py',
    'our_mani/multitask/physmani_ring_release_v590.py',
    'our_mani/multitask/physmani_skill_policy_v548.py',
    'our_mani/multitask/grasp_frame_compensation_native_compat.py')
EXTRA_SOURCES=tuple(dict.fromkeys((*EXTRA_SOURCES,
    'our_mani/multitask/physmani_terminal_opening_v620.py',
    'our_mani/multitask/physmani_buzz_curve_v602.py',
    'our_mani/multitask/physmani_ring_release_v590.py',
    'our_mani/multitask/physmani_remove_stages_v611.py',
    'our_mani/multitask/physmani_geometry_goals_v593.py',
    'our_mani/multitask/physmani_public_rrt_v610.py')))


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
    if args.split!='train' or not args.development_demo or args.method not in ('cv_geometry','context_geometry'):
        raise ValueError('new RML adaptation is fixed original TRAIN development only')
    source,_=components(args.task)
    p=source.preflight(args)
    if p['config'].get('public_ring_frontend_v570',{}).get(args.task)!=gauge_binding(ROOT,p['config'],args.task):
        raise ValueError('freeze current public TCP input to the original certified ring gauge')
    expected=binding(ROOT,p['config']['baseline_repo'])
    if p['config'].get('physmani_geometry_executor_v517')!=expected:
        raise ValueError('freeze original PhysMani RML and explicit geometry adaptation')
    files={str(ROOT/name):sha for name,sha in expected['sources_sha256'].items()}
    files.update(expected['native_sources_sha256'])
    profile=profile_binding(ROOT,p['config']['baseline_repo'])
    if p['config'].get('common_author_RML_profile')!=profile:
        raise ValueError('freeze the same user-authorized author RML profile for all methods')
    files.update({str(Path(p['config']['baseline_repo'])/k):v for k,v in profile['native_sources_sha256'].items()})
    for name in EXTRA_SOURCES:
        files[str(ROOT/name)]=file_sha(ROOT/name)
    checked_subset(ROOT,p['files'],files)
    proof=compensation_binding(ROOT,p['config']['baseline_repo'],args.task)
    if p['config'].get('public_ring_compensation_v580',{}).get(args.task)!=proof:
        raise ValueError('freeze existing100-TRAIN grasp compensation and original author guards')
    checked_subset(ROOT,p['files'],{str(ROOT/proof['prior']['path']):proof['prior']['sha256']})
    if p['config'].get('author_ring_terminal_release_v590',{}).get(args.task)!=release_binding(ROOT,p['config']['baseline_repo'],args.task):
        raise ValueError('freeze original author optional terminal release')
    if p['config'].get('terminal_opening_v620')!=terminal_binding(ROOT):
        raise ValueError('freeze original terminal gripper request timing')
    p['RML_adaptation_binding']=expected
    p['common_author_RML_profile']=profile
    p['component_source_driver']=source.__name__
    return p


def run(p):
    args,config,out=p['args'],p['config'],p['out']
    source,backend=components(args.task)
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
            actor,component_identity=backend.actor(p,env,descriptions)
            frontend=PublicModelFrontend(actor)
            is_button=args.task.startswith('push_moving_button')
            policy=ButtonGeometryIntents(frontend) if is_button else AuthorRingReleaseIntents(frontend,config['public_ring_compensation_v580'][args.task],
                json.loads((ROOT/config['public_ring_compensation_v580'][args.task]['prior']['path']).read_text()),config['author_ring_terminal_release_v590'][args.task])
            callbacks=dict(observe_public_frame=frontend.observe,
                admit_pose_action=policy.admit_pose,admit_joint_target=policy.admit_joint)
            executor=(PhysManiGeometryExecutor(config,**callbacks) if is_button
                      else PhysManiSkillExecutor(config,skill_policy=policy,**callbacks))
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
            atomic_json(out/'raw_execution_diagnostics.json',source.evaluator.clean(dict(
                alignment_receipt=alignment,initial_conditions=initial,policy=policy.receipt(),
                executor=executor.geometry_receipt(),frontend=frontend.receipt(),holds=holds,formal_VAL_TEST=False)))
            record=native_completion_record(p['identity'],process_id=os.getpid(),
                success=bool(env._native_success_latched),fatal=bool(env._native_failure_latched),
                ticks=int(env.ledger.ticks),initial_conditions=initial,alignment_receipt=alignment,
                policy_failure=None)
            record.update(version='r10_TRAIN_trained_public_skills_on_common_author_RML_v621',
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
                    holds=holds,public_ring_frontend=frontend.receipt(),old_reference_managers_executed=False),
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
    args=button_source.parser().parse_args();p=preflight(args)
    if args.check:
        print(json.dumps(dict(complete=True,native_started=False,GPU_started=False,
            identity=p['identity'],RML_adaptation=p['RML_adaptation_binding'],
            common_RML_profile_sha256=fingerprint(p['common_author_RML_profile']))))
        return
    r=run(p)
    print(json.dumps(dict(complete=True,native_success=r['official_task_success'],
                          case=str(p['out']/'case.json'))))


if __name__=='__main__':main()
