"""Run exactly one graph-bound TRAIN/validation case; no queue or retry."""
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import importlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from our_mani.multitask import repaired_policy_case_v1167 as case


def parser():
    value=argparse.ArgumentParser(description=__doc__)
    for name in ('graph','config','split','task','raw-episode-id','method','out','runtime'):
        value.add_argument('--'+name,required=True)
    for name in ('suite','suite-lock','source-inputs','admission'):
        value.add_argument('--'+name)
    value.add_argument('--check',action='store_true')
    return value


def process_identity():
    fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=os.getpid(),start_ticks=int(fields[19]),argv=list(sys.argv),executable=sys.executable)


def run_case(args,*,root=ROOT):
    root=Path(root).resolve();case.require(root==ROOT,'actual project root required')
    chosen=case.selection(root,args)
    phase=chosen['runtime']/('CPU_check' if args.check else 'native')
    case.require(not phase.exists(),'fresh CPU/native phase required; no automatic retry')
    guard=importlib.import_module('our_mani.multitask.post_completed_TEST_resource_guard_v1106')
    with guard.reserved_resources_after_TEST(root,enabled=True) as permit:
        resource=guard.require_permit(permit,root)
        system=case.environment_proof(root,chosen)
        phase.mkdir(parents=True,exist_ok=False)
        launch=dict(schema=case.SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),
            process=process_identity(),graph=chosen['graph_ref'],
            **{key:chosen['graph'][key] for key in case.readiness.HASHES},
            actual_worker=chosen['declaration']['worker'],actual_preparer=chosen['declaration']['preparer'],
            selection={key:chosen[key] for key in ('task','method','split','raw_episode_id')},
            resource_proof=resource,frozen_OS_environment=system,automatic_retry=False,
            original_745_execution_acceptance_granted=False,formal_score_contribution=args.split=='validation')
        case.write_json(phase/'launch.json',launch)
        prepared=None
        try:
            edges=importlib.import_module('our_mani.multitask.frozen_input_edges_v463')
            evaluator=importlib.import_module('scripts.evaluate_multitask_case_r10')
            with ExitStack() as lifetime:
                lifetime.enter_context(edges.declared_edge_scope())
                lifetime.enter_context(edges.evaluator_scope(evaluator))
                lifetime.enter_context(case.admission_scope(root,chosen,args,permit))
                prepared=case.prepare(root,args,chosen=chosen)
                case.verify(root,prepared)
                witness=prepared['graph_witness']
                common=dict(identity=prepared['identity'],graph_witness=witness,actual_split=args.split,
                    component_domain=prepared['component_domain'].receipt(),execution_protocol=prepared['execution_protocol'])
                if args.check:
                    execution=dict(common,mode='CPU_check',complete=True,native_started=False,GPU_started=False,
                        original_substrate_identity=prepared['source_substrate_identity'],
                        source_and_model_preflight=prepared['model_preflight'],
                        physical_factory_constructed=False,graph_source_validation_only=True)
                else:
                    runtime=importlib.import_module('scripts.run_task_executor_smoke_author_suite16_v744_r10')
                    with case.runtime_scope(root,prepared,runtime):
                        runtime.run(prepared)
                    case_ref=case.pointer(root,args.out+'/case.json')
                    record,_=case.graphs.read_project_json(root,case_ref)
                    importlib.import_module('our_mani.multitask.evaluation').validate_completed_case(record,prepared['identity'])
                    case.require(record.get('version')==chosen['declaration']['case_schema']
                        and record.get('execution_protocol')==prepared['execution_protocol']
                        and record.get('graph_witness')==witness
                        and record.get('cleanup_completed') is True
                        and type(record.get('process_id')) is int and record['process_id']==os.getpid(),
                        'clean actual graph-bound case required')
                    case.readiness.validate_sampling_case(record,prepared['expected_author_outcome_protocol'],
                        block=args.task in case.BLOCKS,case_version=chosen['declaration']['case_schema'])
                    restored=prepared['runtime_scope_restoration']
                    case.require(restored.get('complete') is True and restored.get('restored') is True,
                        'all owned runtime scopes must restore')
                    execution=dict(common,mode='graph_bound_native_TRAIN' if args.split=='train' else 'graph_bound_native_validation',
                        complete=True,native_started=True,case=case_ref,
                        native_success=record['official_task_success'],native_failure=record['native_terminal_failure'],
                        native_false_preserved=True,runtime_scope_restoration=restored,
                        qualification_granted_by_worker=False)
                case.verify(root,prepared)
            guard.require_permit(permit,root)
            result=dict(launch,complete=True,finished_utc=datetime.now(timezone.utc).isoformat(),execution=execution,
                identity=prepared['identity'],graph_witness=prepared['graph_witness'],TEST_used_for_fitting=False)
            case.write_json(phase/'worker_sidecar.json',result)
            return result
        except BaseException as error:
            try:
                case.write_json(phase/'failure.json',dict(schema=case.SCHEMA,complete=False,
                    graph=chosen['graph_ref'],**{key:chosen['graph'][key] for key in case.readiness.HASHES},
                    exception_type=type(error).__name__,message=str(error)[:1500],
                    identity=None if prepared is None else prepared['identity'],
                    runtime_scope_restoration=None if prepared is None else prepared['runtime_scope_restoration'],
                    counted_as_policy_failure=False,original_exception_rethrown=True,automatic_retry=False))
            except Exception:pass
            raise


def main(argv=None):
    result=run_case(parser().parse_args(argv))
    print(json.dumps(dict(schema=case.SCHEMA,complete=True,execution=result['execution'],
        original_745_execution_acceptance_granted=False)))
    return result


if __name__=='__main__':main()
