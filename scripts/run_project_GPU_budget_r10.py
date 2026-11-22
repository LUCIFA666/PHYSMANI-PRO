#!/usr/bin/env python3
"""Prepare or explicitly launch an owned project MPS96 execution scope.

Preparing only reads metadata and writes fresh JSON. Launch is separate and
requires ROOT's exact profile/argv/environment-contract approval. This wrapper
does not approve models, datasets, validation or TEST releases.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.gpu_budget import (MPSBudgetController,RECEIPT_SCHEMA,bound,context_probe,
    descendants,identity_is_live,pointer,prepare,process_identity,require,verify_approval,write)


def handshake(controller,profile):
    argv=[profile['context_probe_python'],str(Path(__file__).resolve()),'--context-probe']
    process=subprocess.Popen(argv,env=controller.env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,text=True)
    identity=process_identity(process.pid);controller.event('context_probe_process',dict(argv=argv,identity=identity))
    try:
        available,_,_=select.select([process.stdout],[],[],30)
        require(bool(available),'owned context handshake timed out')
        result=json.loads(process.stdout.readline());controller.event('actual_context_probe',result)
        require(result['status']=='ready' and result['actual_SM_count']==96
            and result['identity']==identity and result['no_kernels_models_or_simulator'] is True,
            'actual project static partition does not report exact96SM')
        controller.validate_clients({process.pid});controller.verify_compute_processes({process.pid})
        controller.control('lspart')
    finally:
        stdout,stderr=process.communicate('\n',timeout=30)
        controller.event('context_probe_cleanup',dict(returncode=process.returncode,stdout=stdout,stderr=stderr,
            identity_no_longer_live=not identity_is_live(identity)))
    require(process.returncode==0,'context probe cleanup failed')


def run(profile_path,approval_path,out):
    profile,approval=verify_approval(ROOT,profile_path,approval_path)
    controller=MPSBudgetController(ROOT,profile,out)
    started=datetime.now(timezone.utc).isoformat();error=None;cleanup_error=None;child=None
    actual_child=None;actual_returncode=None;observed={}
    controller.event('inputs_verified_before',dict(profile=pointer(profile_path),approval=approval,
        budget_launcher_process=process_identity(os.getpid()),resource_only_existing_gates_retained=True))
    try:
        environment=controller.start();handshake(controller,profile)
        argv=profile['allowed_command_argv']
        with (Path(out)/'workload.stdout.log').open('xb') as stdout,(Path(out)/'workload.stderr.log').open('xb') as stderr:
            child=subprocess.Popen(argv,env=environment,stdout=stdout,stderr=stderr,start_new_session=True,cwd=ROOT)
            actual_child=process_identity(child.pid)
            controller.event('actual_workload_process',dict(identity=actual_child,command=argv,
                original_workload_arguments_preserved=True))
            while child.poll() is None:
                allowed=descendants(child.pid)
                # Re-read the tree after querying control if a new compiler
                # process appeared concurrently. Identity snapshots stay raw.
                client_pids=controller.clients()
                allowed|=descendants(child.pid)
                require(set(client_pids)<=allowed,'MPS client is outside the approved workload process tree')
                identities=controller.validate_clients(allowed)
                for identity in identities:
                    old=observed.get(identity['pid'])
                    require(old is None or old['start_ticks']==identity['start_ticks'],'client PID was reused during this run')
                    observed[identity['pid']]=identity
                controller.verify_compute_processes([identity['pid'] for identity in identities],owned_pids=allowed)
                time.sleep(1)
            actual_returncode=child.returncode
            controller.event('actual_workload_exit',dict(returncode=actual_returncode,identity=actual_child,
                identity_no_longer_live=not identity_is_live(actual_child)))
            require(actual_returncode==0,'approved workload returned infrastructure/nonzero exit; no retry')
    except BaseException:
        error=traceback.format_exc();controller.event('execution_error',dict(traceback=error,
            policy_failure_claimed=False,automatic_retry=False))
    finally:
        # A running CUDA workload is never asynchronously killed by this
        # budget observer. Keep evidence and wait for its normal cleanup.
        if child is not None and child.poll() is None:
            controller.event('awaiting_owned_workload_cleanup',dict(pid=child.pid,no_signal_sent=True))
            actual_returncode=child.wait()
        try:controller.close()
        except BaseException:
            cleanup_error=traceback.format_exc();controller.event('cleanup_error',dict(traceback=cleanup_error,
                foreign_process_not_terminated=True,server_retained_if_clients_remain=True))
        try:
            for ref in approval['execution_inputs']:bound(ref)
            bound(approval['approval']);bound(approval['execution_environment_contract'])
        except BaseException:
            cleanup_error=(cleanup_error or '')+traceback.format_exc()
        report=dict(schema=RECEIPT_SCHEMA,started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),
            budget_profile=pointer(profile_path),approval=approval,actual_workload_process=actual_child,
            workload_returncode=actual_returncode,observed_actual_CUDA_clients=list(observed.values()),
            complete=bool(error is None and cleanup_error is None and actual_returncode==0),
            infrastructure_error=error,cleanup_error=cleanup_error,policy_failure_claimed=False,
            automatic_retry=False,static_SM_limit96of170=True,
            original_policy_data_model_clock_physics_and_release_gates_replaced=False,
            runtime_events=[pointer(path) for path in sorted((Path(out)/'events').glob('*.json'))],
            raw_MPS_logs_directory=profile['namespace']+'/l',
            actual_MPS_log_files=[pointer(path) for path in sorted((Path(profile['namespace'])/'l').glob('*')) if path.is_file()])
        write(Path(out)/'receipt.json',report)
    return report


def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--launch',action='store_true')
    mode.add_argument('--context-probe',action='store_true')
    parser.add_argument('--metadata-dir');parser.add_argument('--namespace')
    parser.add_argument('--argv-file');parser.add_argument('--input',action='append',default=[])
    parser.add_argument('--python-executable');parser.add_argument('--profile');parser.add_argument('--ROOT-approval')
    parser.add_argument('--out')
    args=parser.parse_args()
    if args.context_probe:return context_probe()
    if args.prepare:
        require(all((args.metadata_dir,args.namespace,args.argv_file)),'prepare requires metadata-dir/namespace/argv-file')
        argv_file=(ROOT/args.argv_file).resolve();argv=json.loads(argv_file.read_text())
        result=prepare(ROOT,(ROOT/args.metadata_dir).resolve(),ROOT/args.namespace,argv,
            [(ROOT/name).resolve() for name in args.input]+[argv_file],python_executable=args.python_executable)
        print(json.dumps(result));return
    require(all((args.profile,args.ROOT_approval,args.out)),'launch requires profile/ROOT-approval/out')
    result=run((ROOT/args.profile).resolve(),(ROOT/args.ROOT_approval).resolve(),(ROOT/args.out).resolve())
    print(json.dumps(dict(complete=result['complete'],receipt=pointer((ROOT/args.out)/'receipt.json'))))
    if not result['complete']:raise SystemExit(1)


if __name__=='__main__':main()
