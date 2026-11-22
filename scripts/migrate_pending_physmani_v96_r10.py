"""Atomically replace only unstarted baseline requests under a brief owner pause."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,signal,sys,time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from scripts.run_native_resource_owner_v19_r10 import validate_request

OWNER=ROOT/'runtime/framework/native_resource_owner_v19'
SETUP=ROOT/'artifacts/framework/multitask_r10/physmani_task_smoke_v96_setup'
PID=1093047;START=4980878


def checked_owner():
    p=Path('/proc')/str(PID);stat=(p/'stat').read_text().rsplit(')',1)[1].split()
    assert int(stat[19])==START
    argv=(p/'cmdline').read_bytes().split(b'\0')
    assert b'scripts/run_native_resource_owner_v19_r10.py' in argv
    return stat[0]


def main():
    receipt=SETUP/'queue_migration.json'
    if receipt.exists():raise FileExistsError('preserve previous migration')
    assert json.loads((SETUP/'CPU_acceptance.json').read_text())['complete']
    plan=json.loads((SETUP/'plan.json').read_text())
    archive=OWNER/'archived_requests/physmani_v96_migration'
    archive.mkdir(parents=True,exist_ok=False)
    checked_owner();moved=[];skipped=[];paused=False;started=time.monotonic()
    try:
        os.kill(PID,signal.SIGSTOP);paused=True
        for _ in range(100):
            if checked_owner()=='T':break
            time.sleep(.01)
        else:raise RuntimeError('owned process did not confirm stopped state')
        requests={}
        for p in (OWNER/'requests').glob('*.json'):
            row=json.loads(p.read_text())
            assert row['job_id'] not in requests
            requests[row['job_id']]=(p,row)
        status=json.loads((OWNER/'status.json').read_text())
        for row in plan['requests']:
            old_id,new_id=row['original_job_id'],row['job_id']
            if (OWNER/'jobs'/old_id).exists() or old_id in status['active_jobs']:
                skipped.append(dict(job_id=old_id,reason='already_started_or_terminal_preserved'))
                continue
            assert new_id not in requests and not (OWNER/'jobs'/new_id).exists()
            old_path,old=requests[old_id]
            argv=list(old['argv']);expected=list(argv)
            expected[2]=row['argv'][2]
            expected[expected.index('--frozen-inputs')+1]=row['argv'][row['argv'].index('--frozen-inputs')+1]
            expected[expected.index('--out')+1]=row['argv'][row['argv'].index('--out')+1]
            assert expected==row['argv']
            new=dict(old,job_id=new_id,argv=row['argv'],source_sha256=file_sha(ROOT/row['argv'][2]))
            validate_request(new)
            target=old_path.with_name(old_path.name.replace('physmani_v85_','physmani_v96_'))
            assert target!=old_path and not target.exists()
            archived=archive/old_path.name;sha=file_sha(old_path)
            old_path.rename(archived)
            moved.append(dict(original=old_path,archived=archived,new=target,original_sha256=sha,
                old_job_id=old_id,new_job_id=new_id))
            atomic_json(target,new)
        atomic_json(receipt,dict(schema='r10_unstarted_PhysMani_same_policy_queue_migration_v96',
            created_utc=datetime.now(timezone.utc).isoformat(),owner_pid=PID,owner_start_ticks=START,
            owner_pause_seconds=time.monotonic()-started,
            children_not_signalled=True,existing_jobs_not_restarted=True,automatic_retry=False,
            unchanged_checkpoint=True,unchanged_configuration=True,unchanged_policy_and_actuator=True,
            migrated=[dict(original_request=str(x['original'].relative_to(ROOT)),
                archived_request=str(x['archived'].relative_to(ROOT)),
                new_request=str(x['new'].relative_to(ROOT)),original_sha256=x['original_sha256'],
                old_job_id=x['old_job_id'],new_job_id=x['new_job_id']) for x in moved],skipped=skipped,
            source=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=file_sha(__file__))))
    except BaseException:
        for x in reversed(moved):
            x['new'].unlink(missing_ok=True)
            assert file_sha(x['archived'])==x['original_sha256']
            x['archived'].rename(x['original'])
        raise
    finally:
        if paused:
            checked_owner();os.kill(PID,signal.SIGCONT)
    print(json.dumps(dict(migrated=len(moved),preserved_started=len(skipped),owner_resumed=True)),flush=True)


if __name__=='__main__':main()
