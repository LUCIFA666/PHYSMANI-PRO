"""Keep frozen success repair; use source-faithful baseline v4 after acceptance.

All currently running collectors finish before the held original manager is
replaced. Sampler, dynamics, success models and deployed our-policy code remain
unchanged. This wrapper versions only baseline execution and its reporting.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import run_success_repair_r06 as original
from our_mani.benchmark_protocol import file_sha256,verify_lock
BASE=ROOT/'artifacts/framework/aligned_r06'

def freeze():
    path=BASE/'comparison_v4_inputs.json'
    if path.exists():
        lock=json.loads(path.read_text());verify_lock(ROOT,lock);return lock
    files=['scripts/resume_success_repair_r06_v4.py','scripts/summarize_aligned_r06_v4.py',
        'our_mani/ablations/physmani_executor_v4.py','scripts/evaluate_physmani_aligned_r06_v4.py',
        'config/framework/aligned_protocol_r06_baseline_v4.json','tests/test_physmani_executor_v4.py',
        'scripts/smoke_physmani_executor_v4.py','artifacts/framework/aligned_r06/baseline_executor_v4_smoke/report.json',
        'artifacts/framework/aligned_r06/acceptance_tolerances.json']
    lock=dict(created_utc=datetime.now(timezone.utc).isoformat(),files={p:file_sha256(ROOT/p) for p in files},
        revision='baseline executor/source parity only; success collection/controller/training inputs unchanged',
        baseline_selection='author source and TRAIN protocol parity; no TEST outcome tuning',formal_TEST_started=False)
    original.write(path,lock);return lock

def main():
    held=json.loads((BASE/'status.json').read_text()).get('original_orchestrator_pid')
    # Wait for every shard and the native v4 protocol check, without duplicating
    # any running sampler. Existing frozen r06 data are resumed verbatim.
    while True:
        reports=[json.loads(p.read_text()) for p in BASE.glob('collection_*/report.json')]
        probe_path=BASE/'baseline_executor_v4_smoke/report.json'
        probe=json.loads(probe_path.read_text()) if probe_path.exists() else {}
        if probe_path.exists() and not probe.get('passed'):raise RuntimeError('baseline v4 native protocol smoke failed')
        if len(reports)==16 and all(r.get('complete') for r in reports) and probe.get('passed'):break
        if (ROOT/'runtime/framework/success_r06_STOP').exists():raise InterruptedError('r06 audit stop requested')
        time.sleep(5)
    lock=freeze()
    verify_lock(ROOT,json.loads((BASE/'collection_inputs.json').read_text()))
    if held:
        try:
            command=Path(f'/proc/{held}/cmdline').read_bytes().decode().replace('\0',' ')
            if 'scripts/run_success_repair_r06.py' not in command:raise RuntimeError('handoff PID identity changed')
            os.kill(held,signal.SIGTERM);os.kill(held,signal.SIGCONT)
        except FileNotFoundError:pass
    run=original.run
    def versioned_run(arguments,name,gpu=False,sim=True):
        verify_lock(ROOT,lock)
        args=list(arguments)
        if args and args[0]=='scripts/evaluate_physmani_aligned_r05_v3.py':
            args[0]='scripts/evaluate_physmani_aligned_r06_v4.py'
            args=[a.replace('config/framework/aligned_protocol_r06.json','config/framework/aligned_protocol_r06_baseline_v4.json').replace('_physmani_v3','_physmani_v4') for a in args]
        elif args and args[0]=='scripts/summarize_aligned_r06.py':
            args[0]='scripts/summarize_aligned_r06_v4.py'
        run(args,name,gpu=gpu,sim=sim)
        verify_lock(ROOT,lock)
    original.run=versioned_run
    original.main()

if __name__=='__main__':main()
