"""Same four original TRAIN sources, restore the author's public3cm release rule."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.physmani_basketball_release_v558 import binding
from scripts.run_task_executor_smoke_physmani_basketball_v559_r10 import EXTRA_SOURCES


def main():
    base=ROOT/'artifacts/framework/multitask_r10';old=base/'physmani_skills_v550_setup';setup=base/'physmani_basketball_release_v560_setup'
    if setup.exists():raise FileExistsError('preserve every release-rule preparation')
    rows=[r for r in json.loads((old/'CPU_acceptance.json').read_text())['actual_argv_preflights'] if r['task'].startswith('moving_basketball_in_hoop')]
    if len(rows)!=4:raise ValueError('same original two tasks and two methods required')
    files=json.loads((old/'frozen_inputs.json').read_text())['files'];configs={}
    for row in rows:
        task=row['task'];c=json.loads((ROOT/row['argv'][row['argv'].index('--config')+1]).read_text())
        c['author_basketball_release_v558']={task:binding(ROOT,c['baseline_repo'],task)};configs[task]=c
        native=Path(c['baseline_repo'])/'third_party/RLBench/rlbench/tasks'/(task+'.py');files[str(native)]=file_sha(native)
    for n in (*EXTRA_SOURCES,'scripts/prepare_basketball_release_v560_r10.py','tests/test_physmani_basketball_release_v558.py'):
        files[str(ROOT/n)]=file_sha(ROOT/n)
    verify_inputs(ROOT,files);setup.mkdir()
    for task,c in configs.items():
        path=setup/(task+'_configuration.json');atomic_json(path,c);files[str(path)]=file_sha(path)
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,formal_VAL_TEST=False))
    jobs=[]
    for row in rows:
        argv=list(row['argv']);argv[2]='scripts/run_task_executor_smoke_physmani_basketball_v559_r10.py'
        argv[argv.index('--config')+1]=str((setup/(row['task']+'_configuration.json')).relative_to(ROOT))
        argv[argv.index('--frozen-inputs')+1]=str((setup/'frozen_inputs.json').relative_to(ROOT))
        argv[argv.index('--out')+1]=str((base/'physmani_basketball_release_v560'/'development_demo_only'/row['task']/row['raw']/row['method']).relative_to(ROOT))
        jobs.append(dict(row,argv=argv,job_id='physmani_basketball_release_v560_'+row['task']+'_'+row['raw']+'_'+row['method'].split('_')[0]))
    def check(row):return row,subprocess.run(row['argv']+['--check'],cwd=ROOT,text=True,capture_output=True)
    passed=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(check,r) for r in jobs]):
            row,r=future.result()
            if r.returncode:
                atomic_json(setup/(row['job_id']+'_CPU_failed.json'),dict(stdout=r.stdout,stderr=r.stderr,automatic_retry=False,native_started=False));r.check_returncode()
            proof=json.loads(r.stdout)
            if not proof['complete'] or proof['native_started'] or proof['GPU_started']:raise ValueError('CPU-only actual argv required')
            atomic_json(setup/(row['job_id']+'_CPU_preflight.json'),proof);passed.append(row)
            print(json.dumps(dict(CPU_pass=len(passed),total=4,task=row['task'],method=row['method'])),flush=True)
    verify_inputs(ROOT,files)
    atomic_json(setup/'CPU_acceptance.json',dict(complete=True,actual_argv_preflights=passed,
        original_sources_unchanged=True,original_native_task_success_unchanged=True,native_started=False,formal_VAL_TEST=False))
    print(json.dumps(dict(complete=True,checks=4,release_rule='original_author_current_public3cm')),flush=True)


if __name__=='__main__':main()
