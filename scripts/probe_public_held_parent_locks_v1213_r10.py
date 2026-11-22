"""Pure OS rootless-bwrap lock proof probe. Never starts a simulator or model."""
from contextlib import ExitStack, nullcontext
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from our_mani.multitask import public_held_repaired_resource_guard_v1213 as guard

SCHEMA = 'r10_pure_OS_parent_lock_broker_probe_v1213'
MANIFEST_SHA = '1c33a2e35c1134b90aebe896da6176a997242e8aa10dd6c9b08c9e4d5730df30'
CHILD = r'''
import json, os, sys
from pathlib import Path
from our_mani.multitask import public_held_repaired_resource_guard_v1213 as g
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
    global_missing=0
    iterations=25 if scenario=='churn' else 1
    for _ in range(iterations):
        b=g._verify_parent_locks(root,session,ref)
        if scenario=='churn':
            for row in session['parent_global_locks']:
                if not g._flock_rows(session['owner']['pid'],root/row['path']):global_missing+=1
    result=dict(direct_parent_FD=direct,broker_passed=True,admission=a,final=b,
        fresh_distinct_nonce=a['nonce']!=b['nonce'],verification_iterations=iterations+1,
        global_table_diagnostic_missing_rows=global_missing,
        per_FD_records_all_valid=all(x['FD_stat_before_after_verified'] is True for x in b['per_FD_lock_evidence']),
        independent_conflicts_all_confirmed=all(x['confirmed'] is True and x['conflict_errno'] in (11,) for x in b['independent_conflict_evidence']))
except Exception as error:
    result=dict(direct_parent_FD=direct,broker_passed=False,error=type(error).__name__+': '+str(error))
result['namespace']=namespace
print(json.dumps(result,sort_keys=True))
'''
CHURN = r'''
import fcntl,sys,time,json
from pathlib import Path
root=Path(sys.argv[1]);end=time.monotonic()+12;cycles=0
paths=[root/(str(i)+'.lock') for i in range(40)]
while time.monotonic()<end:
    opened=[]
    for p in paths:
        f=p.open('r+b');fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);opened.append(f)
    for f in opened[::2]:f.close()
    for f in opened[1::2]:f.close()
    cycles+=1
print(json.dumps(dict(cycles=cycles)))
'''


def run_probe(*, frozen820=True, scenario='success'):
    if scenario not in ('success', 'parent_unlock', 'wrong_server_PID', 'nonce_mismatch', 'churn'):
        raise ValueError('explicit supported pure OS scenario required')
    temp_base = ROOT / 'runtime/framework/tmp'
    temp_base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='parent_lock_probe1213_', dir=temp_base) as tmp:
        base = Path(tmp)
        with ExitStack() as files:
            locks, streams = [], []
            for index in range(11):
                lock = base / ('CPU_only_probe_' + str(index) + '.lock')
                stream = files.enter_context(lock.open('xb'))
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locks.append(str(lock.relative_to(ROOT)))
                streams.append(stream)
            owner = guard._process()
            session = dict(owner=owner, boot_id=guard._boot(), test_only=True,
                parent_broker=guard._broker_descriptor(owner),
                parent_global_locks=guard._lock_proof(ROOT, owner, locks),
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
                children = []
                if scenario == 'churn':
                    for index in range(4):
                        directory = base / ('churn_' + str(index))
                        directory.mkdir()
                        for i in range(40):(directory / (str(i) + '.lock')).touch()
                        children.append(subprocess.Popen([str(ROOT/'.conda-env/bin/python'), '-B', '-c', CHURN,
                            str(directory)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, close_fds=True))
                if scenario == 'parent_unlock':
                    fcntl.flock(streams[0].fileno(), fcntl.LOCK_UN)
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
                churn_results=[]
                for child in children:
                    out, err=child.communicate(timeout=20)
                    if child.returncode:raise RuntimeError('pure OS churn process failed: '+err)
                    churn_results.append(json.loads(out))
                if process.returncode:
                    raise RuntimeError('pure OS probe process failed: '+process.stderr)
                result = json.loads(process.stdout)
                result['churn_children']=churn_results
                result.update(schema=SCHEMA, scenario=scenario, frozen820=frozen820,
                    returncode=process.returncode, Native_started=False, GPU_started=False,
                    scientific_admission_granted=False, global_permissions_changed=False,
                    test_only=True)
                return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--scenario', default='success', choices=['success', 'parent_unlock', 'wrong_server_PID', 'nonce_mismatch', 'churn', 'all'])
    args = parser.parse_args(argv)
    if not args.execute:
        return dict(schema=SCHEMA, default_inert=True, Native_started=False)
    if args.scenario == 'all':
        results = {name: run_probe(frozen820=True, scenario=name) for name in
            ('success', 'parent_unlock', 'wrong_server_PID', 'nonce_mismatch', 'churn')}
        checks = dict(actual820_EACCES_reproduced=all(x['direct_parent_FD'] == 'EACCES' for x in results.values()),
            actual820_namespace_verified=all(x['namespace']['actual820_namespace_verified'] is True for x in results.values()),
            actual820_broker_passed=results['success']['broker_passed'] is True,
            actual820_per_FD_fdinfo_verified=results['success'].get('per_FD_records_all_valid') is True,
            actual820_independent_conflicts_verified=results['success'].get('independent_conflicts_all_confirmed') is True,
            actual820_churn_per_FD_and_conflicts_stable=results['churn'].get('broker_passed') is True
                and results['churn'].get('per_FD_records_all_valid') is True
                and results['churn'].get('independent_conflicts_all_confirmed') is True
                and len(results['churn'].get('churn_children',[]))==4,
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
