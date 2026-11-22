"""Queue each actual-argv CPU-verified full16 TRAIN smoke exactly once.

Independent cases can run while later CPU checks finish. A failed preflight
never starts native work. Existing outputs/owner requests are never retried.
"""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.case_output_isolation_v238 import require_development_output
from scripts.run_native_resource_owner_v19_r10 import validate_request
BASE=ROOT/'runtime/framework/queue_rml_suite16_smoke_v674'
SETUP=ROOT/'artifacts/framework/multitask_r10/rml_suite16_smoke_v666_setup'
OWNER=ROOT/'runtime/framework/native_resource_owner_v401'


def ticks(pid):return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
def utc():return datetime.now(timezone.utc).isoformat()


def main():
    lock=(ROOT/'runtime/framework/queue_rml_suite16_smoke_v674.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if BASE.exists() or (SETUP/'queued_current_owner.json').exists():raise FileExistsError('preserve the existing one-time queue supervisor')
    # The plan and lock must already have been emitted by the running CPU
    # preparation. Waiting here never reads a VAL/TEST source.
    while not (SETUP/'plan.json').exists():
        if (ROOT/'runtime/framework/multitask_r10_STOP').exists():raise InterruptedError('controlled project stop')
        time.sleep(2.)
    plan=json.loads((SETUP/'plan.json').read_text());jobs=plan['planned'];files=json.loads((SETUP/'frozen_inputs.json').read_text())['files']
    if len(jobs)!=48 or plan.get('source_outcome_filtering') is not False:raise ValueError('exact original first full16x3 plan required')
    verify_inputs(ROOT,files)
    launch=json.loads((OWNER/'launch.json').read_text())
    if ticks(launch['pid'])!=launch['start_ticks']:raise ValueError('sole live native owner changed')
    for r in (SETUP/'plan.json',SETUP/'frozen_inputs.json'):
        if not r.is_file():raise ValueError('frozen CPU plan missing')
    BASE.mkdir();atomic_json(BASE/'launch.json',dict(pid=os.getpid(),start_ticks=ticks(os.getpid()),created_utc=utc(),
        owner_pid=launch['pid'],owner_start_ticks=launch['start_ticks'],plan_sha256=file_sha(SETUP/'plan.json'),
        input_lock_sha256=file_sha(SETUP/'frozen_inputs.json'),source_sha256=file_sha(__file__),automatic_retry=False))
    handled=set();queued=[];failed=[]
    while len(handled)<48:
        state=json.loads((OWNER/'status.json').read_text())
        if state['stop_requested'] or (ROOT/'runtime/framework/multitask_r10_STOP').exists():raise InterruptedError('controlled project stop')
        if ticks(launch['pid'])!=launch['start_ticks']:raise ValueError('native owner identity changed')
        for index,row in enumerate(jobs):
            name=row['job_id']
            if name in handled:continue
            failure=SETUP/(name+'_CPU_failed.json');proof_path=SETUP/(name+'_CPU_preflight.json')
            if failure.exists():
                handled.add(name);failed.append(name);continue
            if not proof_path.exists():continue
            proof=json.loads(proof_path.read_text());identity=proof.get('identity',{})
            argv=row['argv'];config=ROOT/argv[argv.index('--config')+1];output_arg=argv[argv.index('--out')+1]
            if (proof.get('complete') is not True or proof.get('native_started') is not False or proof.get('GPU_started') is not False
                or proof.get('actual_split')!='train' or identity.get('task')!=row['task'] or identity.get('method')!=row['method']
                or identity.get('parent_raw_episode_id')!=row['raw'] or identity.get('config_sha256')!=file_sha(config)
                or identity.get('frozen_inputs_sha256')!=file_sha(SETUP/'frozen_inputs.json')):
                raise ValueError('CPU receipt does not match actual immutable case argv: '+name)
            output=require_development_output(ROOT,output_arg,task=row['task'],raw=row['raw'],method=row['method'])
            existing=[json.loads(p.read_text()) for owner in (OWNER,ROOT/'runtime/framework/native_resource_owner_v19') for p in (owner/'requests').glob('*.json')]
            if output.exists() or any('--out' in r['argv'] and r['argv'][r['argv'].index('--out')+1]==output_arg for r in existing):
                raise FileExistsError('case output already exists or was already submitted: '+name)
            request=dict(schema='r10_owned_native_TRAIN_request_v19',kind='executor_smoke',job_id=name,argv=argv,
                use_gpu=row['method']=='physmani',timeout_seconds=1800,source_sha256=file_sha(ROOT/argv[2]))
            validate_request(request);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            request_path=OWNER/'requests'/f'0041_{stamp}_{index:04d}_{name}.json'
            atomic_json(request_path,request)
            atomic_json(SETUP/(name+'_queued_v674.json'),dict(queued_once=True,job_id=name,owner_pid=launch['pid'],
                owner_start_ticks=launch['start_ticks'],CPU_preflight_sha256=file_sha(proof_path),
                request_sha256=file_sha(request_path),automatic_retry=False,formal_VAL_TEST=False))
            handled.add(name);queued.append(name)
            print(json.dumps(dict(queued=len(queued),CPU_failed=len(failed),total=48,job_id=name)),flush=True)
        atomic_json(BASE/'status.json',dict(updated_utc=utc(),queued=len(queued),CPU_failed=failed,handled=len(handled),
            expected=48,automatic_retry=False,formal_VAL_TEST=False),overwrite=True)
        if len(handled)<48:time.sleep(2.)
    if not failed:
        atomic_json(SETUP/'queued_current_owner.json',dict(queued_once=True,owner_pid=launch['pid'],owner_start_ticks=launch['start_ticks'],
            request_jobs=queued,automatic_retry=False,formal_VAL_TEST=False,actual_individual_CPU_proofs_preserved=True))
    atomic_json(BASE/'terminal.json',dict(complete=not failed,finished_utc=utc(),queued=len(queued),CPU_failed=failed,
        native_outcomes_not_filtered=True,automatic_retry=False))


if __name__=='__main__':main()
