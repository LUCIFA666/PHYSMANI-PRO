"""Inert by default; one fixed1204 TRAIN development case per Root invocation.

Root runs the static six CPU->Native after the formal owner releases1106.
--check never constructs Native; --execute is mandatory for any Native call.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import importlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from our_mani.multitask import train_public_held_block_development_v1204 as overlay
from scripts import run_TRAIN_block_preclose_full16_overlay_v1154_r10 as environment_parent


def parser():
    value=argparse.ArgumentParser(description=__doc__);mode=value.add_mutually_exclusive_group()
    mode.add_argument('--check',action='store_true');mode.add_argument('--execute',action='store_true')
    for name in ('task','raw-episode-id','method','out','runtime','overlay-lock','config','frozen-inputs'):
        value.add_argument('--'+name)
    value.add_argument('--split',default='train')
    return value


def process_identity():
    fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=os.getpid(),start_ticks=int(fields[19]),argv=list(sys.argv),executable=sys.executable)


def resource_guard():
    return importlib.import_module(overlay.GUARD)


def environment_proof(root):
    return environment_parent.environment_proof(root)


def _dead(process):
    try:
        fields=Path('/proc/'+str(process['pid'])+'/stat').read_text().rsplit(')',1)[1].split()
        return fields[0] in ('Z','X') or int(fields[19])!=process['start_ticks']
    except FileNotFoundError:
        return True


def require_prior_CPU(root,args,lock_ref):
    name=args.runtime+'/CPU_check/worker_sidecar.json'
    overlay._path(root,name,prefix=overlay.RUNTIME_PREFIX)
    ref=overlay.pointer(root,name)
    value=json.loads(overlay.raw(root,name,ref['sha256']));execution=value.get('execution',{})
    expected=dict(task=args.task,method=args.method,raw_episode_id=args.raw_episode_id)
    overlay.require(value.get('schema')==overlay.SCHEMA and value.get('complete') is True
        and value.get('resource_lease_scope_exited') is True
        and value.get('source_overlay_lock')==lock_ref and value.get('actual_worker')==overlay.pointer(root,overlay.WORKER)
        and all(value.get('selection',{}).get(k)==v for k,v in expected.items())
        and execution.get('mode')=='CPU_check' and execution.get('actual_split')=='train'
        and execution.get('native_started') is False and execution.get('GPU_started') is False
        and execution.get('physical_factory_constructed') is False
        and execution.get('complete') is True and _dead(value['process']),
        'this exact fixed case must first complete its own CPU preflight in an exited process')
    namespace=value['frozen_OS_environment']
    overlay.require(_dead(dict(pid=namespace['outer_wrapper_pid'],start_ticks=namespace['outer_wrapper_start_ticks'])),
        'prior CPU820 wrapper must actually exit before Native')
    return value,ref


def run_case(args,*,root=ROOT):
    overlay.require(not (args.check and args.execute),'CPU and Native modes are mutually exclusive')
    if not args.check and not args.execute:
        return dict(schema=overlay.SCHEMA,enabled=False,Native_started=False,GPU_started=False,
            source_graph_or_dataset_read=False,fixed_cases=[list(row) for row in overlay.FIXED_CASES],
            formal_TRAIN192_acceptance_granted=False,formal_VAL_TEST_qualification=False)
    root=Path(root).resolve();overlay.require(root==ROOT,'actual project root required')
    overlay.require(all(getattr(args,k,None) for k in ('task','raw_episode_id','method','out','runtime','overlay_lock')),
        'explicit fixed TRAIN selection, new output/runtime and source lock required')
    chosen=overlay.selection(root,args);locked,lock_ref=overlay.read_source_lock(root,args.overlay_lock)
    out=overlay._path(root,args.out,prefix=overlay.ARTIFACT_PREFIX,must_exist=False)
    overlay.require(not out.exists(),'fresh1204 output required; never overwrite or retry')
    base=overlay._path(root,args.runtime,prefix=overlay.RUNTIME_PREFIX,must_exist=False)
    prior_CPU=None;prior_CPU_ref=None
    if args.execute:
        overlay.require_native_ready(dict(source_overlay=locked,root_explicit_execute=True))
        prior_CPU,prior_CPU_ref=require_prior_CPU(root,args,lock_ref)
    phase=base/('CPU_check' if args.check else 'native')
    overlay.require(not phase.exists(),'fresh phase required; no repeated CPU or Native attempt')
    guard=resource_guard();prepared=None;result=None
    try:
        with guard.reserved_resources_after_TEST(root,enabled=True) as permit:
            resource=guard.require_permit(permit,root);system=environment_proof(root)
            phase.mkdir(parents=True,exist_ok=False)
            launch=dict(schema=overlay.SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),process=process_identity(),
                selection=chosen,actual_worker=overlay.pointer(root,overlay.WORKER),actual_preparer=overlay.pointer(root,overlay.SOURCE),
                source_overlay_lock=lock_ref,root_explicit_execute=args.execute is True,resource_proof=resource,
                frozen_OS_environment=system,development_only=True,formal_score_contribution=False,
                formal_TRAIN192_acceptance_granted=False,formal_VAL_TEST_qualification=False,
                borrowed_frozen_physical_graph_identity=False,automatic_retry=False,source_outcome_filtering=False)
            overlay.write_json(phase/'launch.json',launch)
            edges=importlib.import_module('our_mani.multitask.frozen_input_edges_v463')
            evaluator=importlib.import_module('scripts.evaluate_multitask_case_r10')
            with ExitStack() as lifetime:
                lifetime.enter_context(edges.declared_edge_scope());lifetime.enter_context(edges.evaluator_scope(evaluator))
                prepared=overlay.prepare(root,args);overlay.verify(root,prepared)
                if prior_CPU is not None:
                    overlay.require(prior_CPU['execution']['identity']==prepared['identity'],
                        'actual Native source/config/model identity differs from its own CPU preflight')
                common=dict(identity=prepared['identity'],original_substrate_identity=prepared['source_substrate_identity'],
                    component_domain=prepared['component_domain'].receipt(),execution_protocol=prepared['execution_protocol'],actual_split='train')
                if args.check:
                    execution=dict(common,mode='CPU_check',complete=True,native_started=False,GPU_started=False,
                        physical_factory_constructed=False,source_and_model_preflight=prepared['model_preflight'])
                else:
                    runtime=importlib.import_module('scripts.run_task_executor_smoke_author_suite16_v744_r10')
                    started=time.monotonic()
                    with overlay.runtime_scope(root,prepared,runtime,permit=permit):runtime.run(prepared)
                    ended=time.monotonic();name=args.out+'/case.json';record=json.loads(overlay.raw(root,name))
                    importlib.import_module('our_mani.multitask.evaluation').validate_completed_case(record,prepared['identity'])
                    sampler=importlib.import_module('our_mani.multitask.repaired_readiness_v1144')
                    sampler.validate_sampling_case(record,prepared['execution_protocol']['unchanged_action_outcome_sampling'],
                        block=True,case_version=overlay.SCHEMA+'_case')
                    overlay.require(record.get('version')==overlay.SCHEMA+'_case'
                        and record.get('execution_protocol')==prepared['execution_protocol']
                        and record.get('source_overlay_lock')==lock_ref and record.get('cleanup_completed') is True
                        and record.get('process_id')==os.getpid() and type(record.get('official_task_success')) is bool,
                        'clean actually identified1204 development case required')
                    restored=prepared['runtime_scope_restoration']
                    overlay.require(restored.get('complete') is True and restored.get('restored') is True,'actual owned scope must restore')
                    execution=dict(common,mode='new_public_held_block_native_TRAIN',complete=True,native_started=True,
                        case=overlay.pointer(root,name),native_success=record['official_task_success'],
                        native_failure=record['native_terminal_failure'],native_false_preserved=True,
                        runtime_scope_restoration=restored,actual_native_interval=dict(start_monotonic=started,end_monotonic=ended),
                        prior_CPU=prior_CPU_ref,new_physical_candidate=True,formal_TRAIN192_acceptance_granted=False)
            overlay.verify(root,prepared);guard.require_permit(permit,root)
            result=dict(launch,complete=True,finished_utc=datetime.now(timezone.utc).isoformat(),execution=execution,
                actual_CPU_affinity=sorted(os.sched_getaffinity(0)),TEST_used_for_fitting=False)
        result['resource_lease_scope_exited']=True
        overlay.write_json(phase/'worker_sidecar.json',result)
        return result
    except BaseException as error:
        if phase.exists():
            try:
                overlay.write_json(phase/'failure.json',dict(schema=overlay.SCHEMA,complete=False,
                    exception_type=type(error).__name__,message=str(error)[:1500],
                    identity=None if prepared is None else prepared['identity'],
                    runtime_scope_restoration=None if prepared is None else prepared['runtime_scope_restoration'],
                    counted_as_policy_failure=False,original_exception_rethrown=True,automatic_retry=False))
            except Exception:
                pass
        raise


def main(argv=None):
    result=run_case(parser().parse_args(argv))
    print(json.dumps(result,ensure_ascii=False));return result


if __name__=='__main__':main()
