"""Queue a fully checked fixed TRAIN batch once, with per-case output isolation."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.case_output_isolation_v238 import require_development_output

B=ROOT/'artifacts/framework/multitask_r10'
SPECS={
    'button_author_intent_v361_setup': (('push_moving_button','push_moving_button_high_speed'),1,'button_author_intent_v361','0011u'),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup',choices=tuple(SPECS),required=True)
    args=parser.parse_args();setup=B/args.setup
    tasks,count,label,priority=SPECS[args.setup]
    if (setup/'queued_first_sources.json').exists():raise FileExistsError('batch already queued; inspect existing jobs')
    acceptance=json.loads((setup/'CPU_acceptance.json').read_text())
    if acceptance['complete'] is not True:raise ValueError('complete actual argv CPU proof required')
    checks=acceptance['actual_argv_preflights'];methods=('cv_geometry','context_geometry')
    roster=json.loads((ROOT/'artifacts/framework/protocol/train_manifest.json').read_text())['episodes']
    original={task:[r['raw_episode_id'] for r in roster if r['task']==task][:count] for task in tasks}
    expected={(task,raw,method) for task in tasks for raw in original[task] for method in methods}
    actual=[(r['task'],r['raw'],r['method']) for r in checks]
    if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('exact original fixed source/method grid required')
    files=json.loads((setup/'frozen_inputs.json').read_text())['files'];verify_inputs(ROOT,files)
    owner=ROOT/'runtime/framework/native_resource_owner_v19';pid=1093047;start=4980878
    stat=Path(f'/proc/{pid}/stat').read_text()
    if int(stat.rsplit(')',1)[1].split()[19])!=start:raise ValueError('original native owner handle differs')
    if 'run_native_resource_owner_v19_r10.py' not in Path(f'/proc/{pid}/cmdline').read_bytes().decode():
        raise ValueError('owner command differs')
    status=json.loads((owner/'status.json').read_text())
    if status['stop_requested']:raise ValueError('owner is stopping')
    existing={json.loads(p.read_text())['job_id'] for p in (owner/'requests').glob('*.json')}
    seen_outputs=set();planned=[]
    for index,row in enumerate(checks,1):
        argv=row['argv'];task=row['task'];raw=row['raw'];method=row['method']
        if argv[:2]!=['bash','scripts/run_local_sim.sh'] or not argv[2].startswith('scripts/run_task_executor_smoke_'):
            raise ValueError('checked local executor smoke worker required')
        for key,value in (('--task',task),('--raw-episode-id',raw),('--method',method),('--split','train')):
            if argv[argv.index(key)+1]!=value:raise ValueError('checked argv and source grid differ')
        if argv[argv.index('--config')+1]!=str((setup/'configuration.json').relative_to(ROOT)):
            raise ValueError('actual config differs from checked setup')
        worker=ROOT/argv[2]
        if files.get(str(worker))!=file_sha(worker):raise ValueError('actual worker differs from frozen input lock')
        output=require_development_output(ROOT,argv[argv.index('--out')+1],task=task,raw=raw,method=method)
        if output in seen_outputs or output.exists():raise ValueError('output duplicate or already used')
        seen_outputs.add(output)
        candidates=[setup/(task+'_'+raw+'_'+method+'_CPU_preflight.json'),setup/(task+'_'+method+'_CPU_preflight.json')]
        existing_proofs=[p for p in candidates if p.exists()]
        if len(existing_proofs)!=1:raise ValueError('one exact actual case CPU proof required')
        proof=json.loads(existing_proofs[0].read_text())
        if not proof['complete'] or proof['native_started'] or proof['GPU_started']:
            raise ValueError('actual case CPU proof incomplete')
        job=f'{label}_{task}_{raw}_{method.split("_")[0]}'
        if job in existing or job in status['completed'] or job in status['active_jobs']:
            raise ValueError('job already submitted or executed')
        path=owner/'requests'/f'{priority}{index}_{job}.json'
        request=dict(schema='r10_owned_native_TRAIN_request_v19',job_id=job,kind='executor_smoke',
            argv=argv,use_gpu=False,timeout_seconds=1800,source_sha256=file_sha(worker))
        planned.append((path,request,row))
    rows=[]
    for path,request,row in planned:
        atomic_json(path,request)
        rows.append(dict(request=str(path.relative_to(ROOT)),job_id=request['job_id'],
            task=task,raw=row['raw'],method=row['method']))
    atomic_json(setup/'queued_first_sources.json',dict(owner_pid=pid,owner_start_ticks=start,
        queued_once=True,automatic_retry=False,cases=rows,native_started_by_owner_only=True,
        GPU_started=False,formal_VAL_TEST=False,distinct_correct_case_outputs_verified=True,
        source_script=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=file_sha(__file__))))
    print(json.dumps(dict(queued=len(rows),jobs=[r['job_id'] for r in rows],frozen_inputs=len(files))),flush=True)


if __name__=='__main__':main()
