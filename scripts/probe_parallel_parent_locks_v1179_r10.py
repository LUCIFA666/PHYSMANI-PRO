"""Pure OS rootless-bwrap lock proof probe. Never starts a simulator or model."""
from contextlib import nullcontext
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from our_mani.multitask import parallel_repaired_resource_guard_v1179 as guard

SCHEMA = 'r10_pure_OS_parent_lock_broker_probe_v1179'
MANIFEST_SHA = '1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30'
CHILD = r'''
import json, os, sys
from pathlib import Path
from our_mani.multitask import parallel_repaired_resource_guard_v1179 as g
root=Path(sys.argv[1]); ref=json.loads(sys.argv[2]); scenario=sys.argv[3]
session=json.loads((root/ref['path']).read_bytes())
namespace=None
if sys.argv[5]=='frozen820':
    from scripts import run_frozen_system_environment_v820_r10 as environment
    marker,wrapper,current,reference,outer_read=environment.namespace_context('1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30')
    namespace=dict(actual820_namespace_verified=True,current_namespaces=current,
        reference_namespaces=reference,outer_wrapper_pid=marker['outer_wrapper_pid'],
        outer_wrapper_start_ticks=marker['outer_wrapper_start_ticks'],
        live_outer_namespace_read=outer_read,uid=os.getuid(),gid=os.getgid(),
        user_namespace=os.readlink('/proc/self/ns/user'),
        uid_map=Path('/proc/self/uid_map').read_text())
original=session['owner'].copy(); fd=session['parent_global_locks'][0]['fd']
try:
    Path('/proc/'+str(original['pid'])+'/fd/'+str(fd)).stat()
    direct='available'
except PermissionError:
    direct='EACCES'
if scenario=='wrong_server_PID':
    wrong=json.loads(sys.argv[4]);session['owner']=wrong;session['parent_broker']['owner']=wrong
try:
    a=g._verify_parent_locks(root,session,ref)
    b=g._verify_parent_locks(root,session,ref)
    result=dict(direct_parent_FD=direct,broker_passed=True,admission=a,final=b,
        fresh_distinct_nonce=a['nonce']!=b['nonce'])
except Exception as error:
    result=dict(direct_parent_FD=direct,broker_passed=False,error=type(error).__name__+': '+str(error))
result['namespace']=namespace
print(json.dumps(result,sort_keys=True))
'''


def run_probe(*, frozen820=True, scenario='success'):
    if scenario not in ('success', 'parent_unlock', 'wrong_server_PID', 'nonce_mismatch'):
        raise ValueError('explicit supported pure OS scenario required')
    temp_base = ROOT / 'runtime/framework/tmp'
    temp_base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='parent_lock_probe1179_', dir=temp_base) as tmp:
        base = Path(tmp)
        lock = base / 'CPU_only_probe.lock'
        with lock.open('xb') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            owner = guard._process()
            session = dict(owner=owner, boot_id=guard._boot(), test_only=True,
                parent_broker=guard._broker_descriptor(owner),
                parent_global_locks=guard._lock_proof(ROOT, owner, [str(lock.relative_to(ROOT))]),
                guard=guard._source(ROOT, guard.SOURCE), parent_guard=guard._source(ROOT, guard.PARENT_SOURCE),
                jobs=[], scientific_admission_granted=False, resource_only=True)
            ref = guard._fresh(ROOT, str((base/'session.json').relative_to(ROOT)), session)
            original_response = guard._broker_response
            def wrong_nonce(*args, **kwargs):
                response = original_response(*args, **kwargs)
                response['nonce'] = '0' * 64
                return response
            mutation = patch.object(guard, '_broker_response', side_effect=wrong_nonce) if scenario == 'nonce_mismatch' else nullcontext()
            with mutation, guard._parent_broker(ROOT, session, ref):
                if scenario == 'parent_unlock':
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                command = [str(ROOT/'.conda-env/bin/python'), '-B', '-c', CHILD,
                    str(ROOT), json.dumps(ref), scenario, json.dumps(guard._process(os.getppid())),
                    'frozen820' if frozen820 else 'minimal_bwrap']
                if frozen820:
                    argv = [str(ROOT/'.conda-env/bin/python'), '-B',
                        'scripts/run_frozen_system_environment_v820_r10.py',
                        '--manifest-sha256', MANIFEST_SHA, '--', *command]
                else:
                    argv = ['/usr/bin/bwrap', '--bind', '/', '/', '--dev-bind', '/dev', '/dev', '--', *command]
                process = subprocess.run(argv, cwd=ROOT, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
                if process.returncode:
                    raise RuntimeError('pure OS probe process failed: '+process.stderr)
                result = json.loads(process.stdout)
                result.update(schema=SCHEMA, scenario=scenario, frozen820=frozen820,
                    returncode=process.returncode, Native_started=False, GPU_started=False,
                    scientific_admission_granted=False, global_permissions_changed=False,
                    test_only=True)
                return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--scenario', default='success', choices=['success', 'parent_unlock', 'wrong_server_PID', 'nonce_mismatch', 'all'])
    args = parser.parse_args(argv)
    if not args.execute:
        return dict(schema=SCHEMA, default_inert=True, Native_started=False)
    if args.scenario == 'all':
        results = {name: run_probe(frozen820=True, scenario=name) for name in
            ('success', 'parent_unlock', 'wrong_server_PID', 'nonce_mismatch')}
        checks = dict(actual820_EACCES_reproduced=all(x['direct_parent_FD'] == 'EACCES' for x in results.values()),
            actual820_namespace_verified=all(x['namespace']['actual820_namespace_verified'] is True for x in results.values()),
            actual820_broker_passed=results['success']['broker_passed'] is True,
            fresh_admission_final_nonces=results['success'].get('fresh_distinct_nonce') is True,
            parent_unlock_rejected=results['parent_unlock']['broker_passed'] is False
                and 'FLOCK' in results['parent_unlock'].get('error', ''),
            wrong_server_PID_rejected=results['wrong_server_PID']['broker_passed'] is False
                and 'SO_PEERCRED' in results['wrong_server_PID'].get('error', ''),
            nonce_mismatch_rejected=results['nonce_mismatch']['broker_passed'] is False
                and 'nonce' in results['nonce_mismatch'].get('error', ''))
        if not all(checks.values()):
            raise RuntimeError('actual820 pure OS probe suite failed: '+json.dumps(dict(checks=checks, results=results)))
        return dict(schema=SCHEMA, complete=True, checks=checks, results=results,
            Native_started=False, GPU_started=False, global_permissions_changed=False,
            scientific_admission_granted=False, test_only=True)
    return run_probe(frozen820=True, scenario=args.scenario)


if __name__ == '__main__':
    print(json.dumps(main(), sort_keys=True))
