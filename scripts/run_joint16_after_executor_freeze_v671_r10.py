"""One locked joint16 Context training dependency; no native work or retry.

The existing native owner can continue its CPU geometry jobs. The complete
executor freeze is mandatory before this fresh30-epoch training. Its
baseline GPU queue must be empty before this separate single-GPU reservation
starts. Future GPU baseline requests must wait for this reservation to end.
"""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha

BASE=ROOT/'artifacts/framework/multitask_r10'
RUNTIME=ROOT/'runtime/framework/joint16_after_executor_training_v671'
OUT=BASE/'context_joint16_after_executor_v671'
MANIFEST=BASE/'joint16_buzz_hold_channel_v77/manifest.json'


def start_ticks(pid):return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
def now():return datetime.now(timezone.utc).isoformat()


def main():
    lock=(ROOT/'runtime/framework/joint16_hold77_training_GPU.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if RUNTIME.exists() or OUT.exists():raise FileExistsError('preserve previous training run or failure')
    RUNTIME.mkdir();files={};started=time.perf_counter();child=None;child_start=None
    try:
        freeze_path=BASE/'rml_executor_freeze_v670/report.json'
        freeze=json.loads(freeze_path.read_text())
        if (freeze.get('schema')!='r10_full16_executor_first_TRAIN_runtime_freeze_v670'
            or freeze.get('complete') is not True or freeze.get('completed')!=48
            or freeze.get('requested_new_unified_training_permitted') is not True
            or freeze.get('source_sha256')!=file_sha(ROOT/'scripts/audit_executor_freeze_v670_r10.py')):
            raise ValueError('complete unchanged executor freeze required before requested new training')
        from our_mani.multitask.input_stat_cache_v79 import verify_inputs
        verify_inputs(ROOT,freeze['inputs_sha256'])
        files[str(freeze_path)]=file_sha(freeze_path)
        files.update(freeze['inputs_sha256'])
        proof_path=BASE/'joint16_after_executor_v671_CPU_check/CPU_check.json'
        proof=json.loads(proof_path.read_text())
        assert proof['complete'] and proof['TRAIN_parents']==1600 and proof['offline_VAL_parents']==315 and proof['tasks']==16
        assert proof['optimizer_steps']==0 and proof['GPU_used'] is False
        assert file_sha(MANIFEST)==proof['manifest']['sha256']
        for name,sha in proof['source_code_sha256'].items():
            p=ROOT/name;assert file_sha(p)==sha;files[str(p.resolve())]=sha
        for p in (MANIFEST,proof_path,Path(__file__),ROOT/'scripts/run_aligned_gpu.sh',ROOT/'our_mani/multitask/motion_labels.py'):
            files[str(p.resolve())]=file_sha(p)
        owner=ROOT/'runtime/framework/native_resource_owner_v401'
        owner_launch=json.loads((owner/'launch.json').read_text())
        assert start_ticks(owner_launch['pid'])==owner_launch['start_ticks']
        status=json.loads((owner/'status.json').read_text());assert not status['stop_requested'] and status['allocated_GPU_slots']==0
        pending=[]
        for path in (owner/'requests').glob('*.json'):
            request=json.loads(path.read_text())
            if request['job_id'] not in status['completed'] and request['use_gpu']:pending.append(request['job_id'])
        assert not pending,pending
        for p in Path('/proc').iterdir():
            if not p.name.isdigit() or int(p.name)==os.getpid():continue
            try:argv=[s.decode() for s in (p/'cmdline').read_bytes().split(b'\0') if s]
            except (OSError,UnicodeError):continue
            assert not ('--train' in argv and any(Path(a).name=='train_joint16_context_v73_r10.py' for a in argv)), 'joint16 training already running'
        argv=['bash','scripts/run_aligned_gpu.sh','scripts/train_joint16_context_v73_r10.py','--train',
            '--manifest',str(MANIFEST.relative_to(ROOT)),'--out',str(OUT.relative_to(ROOT)),'--device','cuda']
        launch=dict(schema='r10_joint16_full1600_GPU_training_reservation_v671',created_utc=now(),
            supervisor_pid=os.getpid(),supervisor_start_ticks=start_ticks(os.getpid()),argv=argv,
            original_TRAIN_parents=1600,offline_VAL_parents=315,epochs=30,windows_per_parent=64,batch_size=256,
            shared_models=1,original_tasks=16,max_GPU_training_slots=1,native_slots=0,
            native_owner_GPU_jobs_active=0,native_owner_GPU_requests_pending=[],
            native_geometry_jobs_at_start=status['active_jobs'],executor_qualification_or_formal_release_granted=False,
            training_is_required_by_latest_user_order_after_executor_freeze=True,
            executor_freeze=dict(path=str(freeze_path.relative_to(ROOT)),sha256=file_sha(freeze_path)),
            previous_v157_checkpoint_reused_as_final_training=False,
            verified_source_files_sha256=files,TEST_read=False,automatic_retry=False)
        atomic_json(RUNTIME/'launch.json',launch)
        with (RUNTIME/'TRAIN_stdout.log').open('w') as log:
            child=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            child_start=start_ticks(child.pid)
            current=dict(stage='training',updated_utc=now(),supervisor_pid=os.getpid(),supervisor_start_ticks=start_ticks(os.getpid()),
                launcher_pid=child.pid,launcher_start_ticks=start_ticks(child.pid),GPU_reserved=True,GPU_slots=1,native_slots=0,
                output=str(OUT.relative_to(ROOT)),automatic_retry=False,formal_VAL_TEST=False)
            atomic_json(RUNTIME/'status.json',current)
            atomic_json(RUNTIME/'child_launch.json',current)
            while child.poll() is None:
                state=json.loads((owner/'status.json').read_text())
                if state['allocated_GPU_slots']!=0:
                    # Do not signal another owner's native evaluation. End only
                    # this owned training launcher group if a reservation conflict occurs.
                    import signal
                    os.killpg(child.pid,signal.SIGTERM)
                    raise RuntimeError('native owner acquired GPU while joint16 reservation active; training ended, no automatic retry')
                current['updated_utc']=now()
                if (OUT/'status.json').exists():
                    progress=json.loads((OUT/'status.json').read_text())
                    current['training_stage']=progress.get('stage');current['epoch']=progress.get('epoch')
                atomic_json(RUNTIME/'status.json',current,overwrite=True)
                time.sleep(5.)
            code=child.wait()
        assert code==0,f'training launcher exited {code}'
        trained=json.loads((OUT/'status.json').read_text());meta=trained['metadata']
        assert trained['stage']=='complete' and meta['training_complete'] and meta['completed_epochs']==30
        assert len(meta['jointly_trained_tasks'])==16 and meta['training_source_manifest']['source_episode_count']==1600
        assert meta['VAL_source_episodes']==315 and meta['TEST_used_for_fitting'] is False
        assert file_sha(OUT/'dynamics.pt')==trained['checkpoint_sha256']
        for name,sha in files.items():assert file_sha(name)==sha
        terminal=dict(complete=True,returncode=code,finished_utc=now(),seconds=time.perf_counter()-started,
            checkpoint=dict(path=str((OUT/'dynamics.pt').relative_to(ROOT)),sha256=trained['checkpoint_sha256']),
            TRAIN_parents=1600,offline_VAL_parents=315,tasks=16,epochs=30,GPU_reserved=False,
            original_sources_not_modified=True,automatic_retry=False,formal_VAL_TEST=False)
        atomic_json(RUNTIME/'terminal.json',terminal)
        atomic_json(RUNTIME/'status.json',dict(stage='complete',**terminal),overwrite=True)
        print(json.dumps(terminal),flush=True)
    except BaseException as error:
        if child is not None and child.poll() is None:
            import signal
            if start_ticks(child.pid)!=child_start:raise RuntimeError('owned training launcher identity changed') from error
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30.)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=30.)
        failure=dict(complete=False,stage='failed',finished_utc=now(),type=type(error).__name__,message=str(error),
            GPU_reserved=False,native_started=False,automatic_retry=False,original_sources_not_modified=True,formal_VAL_TEST=False)
        atomic_json(RUNTIME/'failed.json',failure)
        atomic_json(RUNTIME/'status.json',failure,overwrite=True)
        raise
    finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
