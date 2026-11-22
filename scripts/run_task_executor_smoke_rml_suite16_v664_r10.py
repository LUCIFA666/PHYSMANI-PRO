"""One genuine full16 common-RML TRAIN qualification or admitted VAL/TEST case.

The TRAIN resource owner only accepts this worker with --split train. Formal
cases need a separate complete suite admission and their actual split. Every
case creates and closes one native environment; there is no hidden retry.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import gc,json,os,random,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask import rml_case_preflight_v663 as contract
from our_mani.multitask.evaluation import atomic_json,digest,native_completion_record
from our_mani.multitask.joint16_tracks_v31 import BLOCKS
from our_mani.multitask.common_rml_protocol_v530 import attest


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','manifest','frozen-inputs','task','raw-episode-id','method','split','out'):
        p.add_argument('--'+name,required=True)
    for name in ('suite','suite-lock','validation-release'):p.add_argument('--'+name)
    p.add_argument('--replica',type=int,default=0);p.add_argument('--exploration-seed',type=int)
    for name in ('development-demo','record-demo','demo-only','check'):p.add_argument('--'+name,action='store_true')
    return p


def _block_environment(p):
    from our_mani.sensor_alignment.author_rgbd import AlignedLocalSimulation,AlignedOfficialSimulation
    from our_mani.multitask.physmani_block_receipt_v541 import ReceiptBlockView
    args,c,row=p['args'],p['config'],p['row'];raw=None
    try:
        if args.split=='test':
            raw=AlignedOfficialSimulation(args.task,row['path'],protocol=c,seed=int(row.get('seed',c['seed'])),servo_lead=1.)
            descriptions,_=raw.reset_official(row)
        else:
            from scripts.collect_local_rollouts import initialization
            raw=AlignedLocalSimulation(args.task,row['path'],protocol=c,seed=c['seed'],servo_lead=1.)
            x,v,a,goal=initialization(row['path'])
            if not raw.reset(target_position=x,target_velocity=v,target_acceleration=a,goal_position=goal):
                raise RuntimeError('original block development reset IK failed')
            raw.finish_home_reset();descriptions=[]
        return ReceiptBlockView(raw),descriptions
    except BaseException:
        if raw is not None:raw.close()
        raise


def _baseline(p,env,observation,robot,model=None):
    from our_mani.aligned.physmani import build_policy
    if p['args'].task in BLOCKS:
        from our_mani.sensor_alignment.physmani_executor_v6 import PhysManiAdapter
    else:
        from our_mani.multitask.physmani_executor_v7 import PhysManiAdapter
    c=p['config'];config=dict(c,baseline_output=str(p['out']/'disabled_wm_logger'))
    if model is None:
        import torch
        torch.manual_seed(env.seed)
        model=build_policy(ROOT,config,p['args'].task)
    adapter=PhysManiAdapter(ROOT,config,p['args'].task,model)
    variation=int(p['row'].get('variation',0)) if p['args'].task in BLOCKS else int(env.reset_receipt['variation'])
    adapter.reset(env,p['args'].task,variation,observation,robot)
    return adapter,model


def _geometry(p,env,descriptions):
    if p['args'].task in BLOCKS:
        from our_mani.multitask.physmani_block_policy_v538 import PublicBlockCollision,BlockPublicFrontend,BlockGeometryIntents
        from our_mani.multitask.physmani_skill_executor_v534 import PhysManiSkillExecutor
        f=BlockPublicFrontend(ROOT,p['config'],p['args'].task,p['args'].method,env.robot_model,PublicBlockCollision(env))
        policy=BlockGeometryIntents(f)
        executor=PhysManiSkillExecutor(p['config'],skill_policy=policy,observe_public_frame=f.observe,
            admit_pose_action=policy.admit_pose,admit_joint_target=policy.admit_joint)
        return None,dict(actual_shared16_block_adapter=p['model_preflight']['actual_shared16_block_adapter'],
            component_domain=p['component_domain'].receipt()),f,policy,executor
    from our_mani.multitask.rml_component_factory_v643 import create_actor
    from our_mani.multitask.rml_policy_components_v669 import construct
    actor,proof=create_actor(ROOT,env,p['config'],p['args'].method,descriptions,domain=p['component_domain'])
    f,policy,executor=construct(ROOT,p['config'],p['args'].task,actor)
    return actor,proof,f,policy,executor


def run(p):
    import numpy as np
    import torch
    from scripts import evaluate_multitask_case_r10 as evaluator
    from scripts import run_task_executor_smoke_joint16_v174_r10 as shared_source
    from scripts import collect_multitask_recovery_r10 as recovery
    from our_mani.multitask.baseline_measured_windows_v345 import CountedWindowObserver
    args,c,out=p['args'],p['config'],p['out'];block=args.task in BLOCKS
    contract.verify(ROOT,p)
    if out.exists():raise FileExistsError('preserve every previous native case attempt')
    out.mkdir(parents=True)
    atomic_json(out/'inputs.json',evaluator.clean(dict(identity=p['identity'],
        frozen_input_lock=contract.pointer(ROOT,p['lock_path']),verified_input_file_count=len(p['files']),
        input_manifest_format='immutable_external_lock_pointer_complete_source_map_retained',
        model_preflight=p['model_preflight'],component_domain=p['component_domain'].receipt(),
        actual_split=args.split,suite_lock=p['suite_lock_pointer'],full16_policy_sha256=p['full16_policy_sha256'],
        formal_source_read_before_admission=False)))
    torch.set_num_threads(4);torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed'])
    env=actor=adapter=model=None;record=failure=None;captures=dict(simulations=[],paths=[]);started=time.perf_counter()
    hooks_needed=not block and args.split!='test'
    scope=recovery.scoped_owned_hooks(shared_source.hook_plan_for(p),captures) if hooks_needed else nullcontext()
    try:
        with scope:
            if block and args.method=='physmani':
                from our_mani.aligned.physmani import build_policy
                model=build_policy(ROOT,dict(c,baseline_output=str(out/'disabled_wm_logger')),args.task)
                # Match the released legacy block runner's model-build then
                # per-episode reseed, before physical reset.
                torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed'])
            env,descriptions=(_block_environment(p) if block else evaluator.NativeBackend().environment(p))
            parameters=attest(env,c,p['common_author_RML_profile']);window=CountedWindowObserver(env)
            observation,robot=env.observe()
            if block:
                from our_mani.sensor_alignment.author_rgbd import public_initial_conditions
                initial=public_initial_conditions(env,observation,robot)
            else:initial=evaluator.NativeBackend().initial_conditions(env,observation,robot)
            holds=[];steps=[];policy_failure=None;component=None;f=policy=executor=None
            if args.method=='physmani':
                adapter,model=_baseline(p,env,observation,robot,model=model)
            else:
                actor,component,f,policy,executor=_geometry(p,env,descriptions);f.observe(observation,robot)
            while env.ledger.remaining>=env.ledger.dt-1e-8 and not env._native_terminal_latched:
                if (ROOT/'runtime/framework/multitask_r10_STOP').exists():raise InterruptedError('controlled project stop')
                before=int(env.ledger.ticks)
                if adapter is not None:
                    try:
                        try:action=np.asarray(adapter.predict(observation,robot),float)
                        except ValueError as error:
                            if str(error)=='nonfinite PhysMani action':raise evaluator.ExplicitPolicyFailure(str(error)) from error
                            raise
                        if action.shape!=(8,) or not np.isfinite(action).all() or np.round(action[7]) not in (0.,1.):
                            raise evaluator.ExplicitPolicyFailure('malformed_or_nonfinite_PhysMani_action')
                        execution=adapter.execute(env,action)
                    except evaluator.ExplicitPolicyFailure as error:
                        policy_failure=str(error);break
                    except (ValueError,RuntimeError,AssertionError) as error:
                        if not block:raise
                        # Preserve the released legacy block policy failure
                        # handling, with its original protected infrastructure
                        # classification and no retry or false task success.
                        from our_mani.sensor_alignment.physmani_executor_v6 import raise_protected_error
                        raise_protected_error(error)
                        if any(s in str(error).lower() for s in ('out of memory','illegal memory','device-side assert','cuda error','compilation')):raise
                        policy_failure=repr(error);break
                    steps.append(dict(time=float(robot.timestamp),action=action.tolist(),execution=execution))
                    observation,robot=env.observe()
                else:
                    action=policy.select()
                    if action is None:
                        holds.append(dict(time=float(env.time),reason='public_geometry_or_intent_unavailable'))
                        if actor is not None:actor.command_segments.issue('hold')
                        env.hold_tick()
                        if not env._native_terminal_latched:
                            observation,robot=env.observe();f.observe(observation,robot)
                    else:
                        execution=executor.execute(env,action)
                        if hasattr(policy,'observe_owned_RML_path_result'):
                            modification=execution.get('waypoint_modification') or {}
                            policy.observe_owned_RML_path_result(dict(error=execution.get('error'),waypoint_modification={
                                k:modification[k] for k in ('selected_waypoints','cartesian_spacing_m') if k in modification}))
                if not before<env.ledger.ticks<=min(before+10,200):raise ValueError('RML/hold must use1..10 original counted ticks')
            alignment=env.alignment_receipt();motion=window.receipt()
            metrics=(dict(baseline_steps=steps,compute_seconds=adapter.compute_seconds,original_baseline_algorithm_unchanged=True)
                if adapter is not None else dict(predictor=args.method.split('_')[0],ranking='geometry',
                physmani_geometry_executor=executor.geometry_receipt(),geometry_intents=policy.receipt(),holds=holds,
                old_reference_managers_executed=False))
            atomic_json(out/'raw_execution_diagnostics.json',evaluator.clean(dict(alignment_receipt=alignment,
                initial_conditions=initial,actor_metrics=metrics,actual_split=args.split,formal_VAL_TEST=p['formal'])))
            make_record=native_completion_record
            if block:
                from our_mani.multitask.physmani_block_record_v545 import completion_record
                make_record=completion_record
            record=make_record(p['identity'],process_id=os.getpid(),success=bool(env._native_success_latched),
                fatal=bool(env._native_failure_latched),ticks=int(env.ledger.ticks),initial_conditions=initial,
                alignment_receipt=alignment,policy_failure=policy_failure)
            record.update(version='r10_full16_actual_split_common_author_RML_v664',task=args.task,method=args.method,split=args.split,
                source_raw_episode_id=args.raw_episode_id,parent_raw_episode_id=None if args.split=='test' else args.raw_episode_id,
                development_demo_only=not p['formal'],demo_only=not p['formal'],formal_validation_metric=args.split=='validation',
                formal_TEST_metric=args.split=='test',formal_test_used=args.split=='test',TEST_used_for_fitting=False,
                automatic_retry=False,original_source_motion_replayed=False,
                shared_context_checkpoint=c.get('context_checkpoint'),trained_component_identity=component,
                common_author_RML_parameters=parameters,all_stage_counted_joint_motion=motion,
                component_domain=p['component_domain'].receipt(),actor_metrics=metrics,
                full16_frozen_execution=dict(policy_sha256=p['full16_policy_sha256'],source_row_sha256=digest(p['row']),
                    actual_component_purpose=p['component_domain'].purpose,actual_split=args.split,
                    VAL_TEST_relabelled_as_TRAIN=False,original_TEST_reset_unmodified=args.split=='test'),
                wall_seconds=time.perf_counter()-started,simulation_seconds=float(env.time),
                Value_NGFF_FASTER_used=False,historical_R09_results_modified=False)
            env.close();env=None
        contract.verify(ROOT,p)
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
            paths=[deepcopy(x.recovery_receipt) for x in captures['paths']],
            actual_split=args.split,shared_development_hooks_applied=hooks_needed,
            original_TEST_reset_unmodified=args.split=='test')
        atomic_json(out/'owned_hook_receipt.json',evaluator.clean(hooks))
        if failure is not None:atomic_json(out/'infrastructure_error.json',failure)
        del adapter,model,actor;gc.collect()
        if args.method=='physmani':torch.cuda.empty_cache()
    if hooks_needed and (len(hooks['reset'])!=1 or not hooks['reset'][0].get('complete')
        or any(not h.get('close_complete') for h in hooks['paths'])):
        atomic_json(out/'infrastructure_error.json',dict(type='IncompleteOwnedHookReceipt',
            message='actual shared physical reset/RML cleanup incomplete',counted_as_policy_failure=False))
        raise ValueError('shared physical reset/RML cleanup incomplete')
    record['cleanup_completed']=True;atomic_json(out/'case.json',evaluator.clean(record));return record


def main():
    args=parser().parse_args();p=contract.prepare(ROOT,args,Path(__file__))
    if args.check:
        print(json.dumps(dict(complete=True,native_started=False,GPU_started=False,identity=p['identity'],
            component_domain=p['component_domain'].receipt(),model_preflight=p['model_preflight'],actual_split=args.split)))
        return
    r=run(p);print(json.dumps(dict(complete=True,native_success=r['official_task_success'],case=str(p['out']/'case.json'))))


if __name__=='__main__':
    from our_mani.multitask.frozen_input_edges_v463 import evaluator_scope,declared_edge_scope
    from scripts import evaluate_multitask_case_r10 as evaluator
    with declared_edge_scope(),evaluator_scope(evaluator):main()
