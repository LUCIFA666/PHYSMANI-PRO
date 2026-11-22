"""Repair missing CPU-only dependency edges, preserving all v536 proof attempts.

The worker/configuration/model bytes stay frozen. Passed actual argv proofs
are reused unchanged; failed argv proofs get a new explicit lock/output and
an actual CPU preflight. No native job is retried by this preparation.
"""
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs


def main():
    base=ROOT/'artifacts/framework/multitask_r10';parent=base/'physmani_skills_v536_setup'
    setup=base/'physmani_skills_v537_setup'
    if setup.exists():raise FileExistsError('preserve original lock correction')
    proof=json.loads((parent/'CPU_acceptance.json').read_text())
    failed=proof['failed_preflights'];passed=proof['actual_argv_preflights']
    if len(failed)!=2 or {r['task'] for r in failed}!={'put_rubbish_in_moving_bin'}:
        raise ValueError('only the two observed missing normal-bin driver edges may be corrected here')
    for row in failed:
        error=json.loads((parent/(row['job_id']+'_CPU_failed.json')).read_text())
        if 'missing or conflicting in explicit frozen lock' not in error['stderr'] or 'run_task_executor_smoke_bin_physical_v236_r10.py' not in error['stderr']:
            raise ValueError('unexpected preflight failure must be diagnosed independently')
    if len(passed)!=26:raise ValueError('all other26 original preflights required')
    files=json.loads((parent/'frozen_inputs.json').read_text())['files']
    for p in (ROOT/'scripts/run_task_executor_smoke_bin_physical_v236_r10.py',Path(__file__)):
        files[str(p)]=file_sha(p)
    verify_inputs(ROOT,files);setup.mkdir()
    atomic_json(setup/'frozen_inputs.json',dict(files=files,approved_for_execution=True,formal_VAL_TEST=False,
        missing_CPU_driver_edge_corrected=True,worker_and_model_bytes_unchanged=True))
    accepted=list(passed);reused=[]
    for row in passed:
        p=parent/(row['job_id']+'_CPU_preflight.json')
        reused.append(dict(job_id=row['job_id'],proof=dict(path=str(p.relative_to(ROOT)),sha256=file_sha(p)),actual_argv_unchanged=True))
    for row in failed:
        argv=list(row['argv']);argv[argv.index('--frozen-inputs')+1]=str((setup/'frozen_inputs.json').relative_to(ROOT))
        argv[argv.index('--out')+1]=argv[argv.index('--out')+1].replace('/physmani_skills_v536/','/physmani_skills_v537/')
        r=subprocess.run(argv+['--check'],cwd=ROOT,text=True,capture_output=True)
        if r.returncode:
            atomic_json(setup/(row['job_id']+'_CPU_failed.json'),dict(stdout=r.stdout,stderr=r.stderr,automatic_retry=False,native_started=False))
            r.check_returncode()
        checked=json.loads(r.stdout)
        if not checked['complete'] or checked['native_started'] or checked['GPU_started']:raise ValueError('CPU-only actual argv required')
        atomic_json(setup/(row['job_id']+'_CPU_preflight.json'),checked);accepted.append(dict(row,argv=argv))
        print(json.dumps(dict(CPU_new_pass=row['method'],reused_passed=26,total=28)),flush=True)
    verify_inputs(ROOT,files)
    atomic_json(setup/'CPU_acceptance.json',dict(complete=True,actual_argv_preflights=accepted,
        reused_unchanged_actual_argv_proofs=reused,original_fixed_TRAIN_sources=True,
        original_failed_CPU_attempts_retained=True,native_started=False,formal_VAL_TEST=False))
    print(json.dumps(dict(complete=True,actual_argv_cases=28,old_CPU_passes_reused=26,new_CPU_checks=2)),flush=True)


if __name__=='__main__':main()
