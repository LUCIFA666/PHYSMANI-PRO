"""Preserve the initial CPU preparation failure and finish missing read-only checks."""
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs


def main():
    setup=ROOT/'artifacts/framework/multitask_r10/physmani_task_smoke_v85_setup'
    if (setup/'CPU_acceptance.json').exists():raise FileExistsError('preserve completed acceptance')
    initial=setup/'preparation_failed.json'
    assert initial.exists()
    files=json.loads((setup/'frozen_inputs.json').read_text())['files'];verify_inputs(ROOT,files)
    plan=json.loads((setup/'plan.json').read_text());checks=[];supplemental=[]
    for row in plan['requests']:
        if not row['unstarted_at_preparation']:continue
        p=setup/(row['task']+'_CPU_preflight.json')
        if not p.exists():
            result=subprocess.run(row['argv']+['--check'],cwd=ROOT,text=True,capture_output=True)
            log=setup/(row['task']+'_supplemental_CPU_stdout.log');log.write_text(result.stdout)
            err=setup/(row['task']+'_supplemental_CPU_stderr.log');err.write_text(result.stderr)
            result.check_returncode()
            atomic_json(p,json.loads(result.stdout));supplemental.append(row['task'])
        d=json.loads(p.read_text())
        assert d['complete'] and d['check_only'] and not d['native_started'] and not d['GPU_started']
        assert d['identity']['task']==row['task'] and d['identity']['source_raw_episode_id']==row['raw_episode_id']
        assert d['identity']['frozen_inputs_sha256']==file_sha(setup/'frozen_inputs.json')
        checks.append(dict(task=row['task'],raw_episode_id=row['raw_episode_id'],argv=row['argv'],
            receipt=dict(path=str(p.relative_to(ROOT)),sha256=file_sha(p))))
    verify_inputs(ROOT,files)
    atomic_json(setup/'CPU_acceptance.json',dict(complete=True,created_utc=datetime.now(timezone.utc).isoformat(),
        actual_pending_argv_checks=checks,frozen_inputs=len(files),native_started=False,GPU_started=False,
        queue_migrated=False,policy_and_actuator_unchanged=True,formal_release=False,
        initial_CPU_preparation_failure=dict(path=str(initial.relative_to(ROOT)),sha256=file_sha(initial)),
        original_failure_stderr_not_retained_cause_not_asserted=True,
        supplemental_readonly_CPU_checks=supplemental,source_files_unchanged_since_initial_preparation=True,
        no_native_case_retry=True,source=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=file_sha(__file__))))
    print(json.dumps(dict(complete=True,checks=len(checks),supplemental_readonly_checks=supplemental,
        initial_failure_preserved=True,native_started=False,GPU_started=False)),flush=True)


if __name__=='__main__':main()
