"""Submit checked shared-executor cases once through the existing resource owner."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.case_output_isolation_v238 import require_development_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    if not args.label.replace('_', '').isalnum():
        raise ValueError('simple artifact label required')
    setup = ROOT/'artifacts/framework/multitask_r10'/(args.label+'_setup')
    receipt = setup/'queued_first_sources.json'
    if receipt.exists():
        raise FileExistsError('already submitted; inspect existing jobs')
    accepted = json.loads((setup/'CPU_acceptance.json').read_text())
    if accepted.get('complete') is not True or accepted['label'] != args.label:
        raise ValueError('complete exact-label CPU acceptance required')
    files = json.loads((setup/'frozen_inputs.json').read_text())['files']
    verify_inputs(ROOT, files)
    owner = ROOT/'runtime/framework/native_resource_owner_v19'
    pid, start = 1093047, 4980878
    if (int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]) != start
            or b'run_native_resource_owner_v19_r10.py' not in Path(f'/proc/{pid}/cmdline').read_bytes()):
        raise ValueError('original resource owner identity changed')
    status = json.loads((owner/'status.json').read_text())
    if status['stop_requested']:
        raise ValueError('resource owner stopping')
    existing = [json.loads(p.read_text()) for p in (owner/'requests').glob('*.json')]
    existing_jobs = {row['job_id'] for row in existing}
    existing_outputs = {row['argv'][row['argv'].index('--out')+1]
        for row in existing if '--out' in row['argv']}
    planned, outputs = [], set()
    for index, row in enumerate(accepted['actual_argv_preflights']):
        argv = row['argv']; task, raw, method = row['task'], row['raw'], row['method']
        proof = json.loads((setup/row['proof']).read_text())
        if not proof['complete'] or proof['native_started'] or proof['GPU_started']:
            raise ValueError('native case lacks exact CPU-only preflight')
        if argv[:3] != ['bash', 'scripts/run_local_sim.sh', 'scripts/run_continuous_executor_smoke_r10.py']:
            raise ValueError('unexpected shared executor entrypoint')
        for key, expected in (('--task', task), ('--raw-episode-id', raw), ('--method', method),
                              ('--split', 'train'), ('--config', str((setup/'configuration.json').relative_to(ROOT)))):
            if argv[argv.index(key)+1] != expected:
                raise ValueError('checked source/command mismatch')
        output_arg = argv[argv.index('--out')+1]
        output = require_development_output(ROOT, output_arg, task=task, raw=raw, method=method)
        job = args.label+'_'+task+'_'+raw+'_'+method.split('_')[0]
        if job in existing_jobs or output.exists() or output_arg in existing_outputs or output in outputs:
            raise ValueError('job or output already submitted or used')
        outputs.add(output)
        request = dict(schema='r10_owned_native_TRAIN_request_v19', job_id=job,
            kind='executor_smoke', argv=argv, use_gpu=False, timeout_seconds=1800,
            source_sha256=file_sha(ROOT/argv[2]))
        planned.append((owner/'requests'/f'0020_{args.label}_{index}_{job}.json', request))
    for path, request in planned:
        atomic_json(path, request)
    atomic_json(receipt, dict(queued_once=True, owner_pid=pid, owner_start_ticks=start,
        automatic_retry=False, cases=[r for _, r in planned], formal_VAL_TEST=False))
    print(json.dumps(dict(queued=len(planned), jobs=[r['job_id'] for _, r in planned])), flush=True)


if __name__ == '__main__':
    main()
