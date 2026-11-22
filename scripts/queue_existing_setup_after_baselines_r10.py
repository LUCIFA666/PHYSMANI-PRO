"""Submit a fresh verified TRAIN setup after baseline priorities, retaining native2/GPU1."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_native_resource_owner_v19_r10 import validate_request
from our_mani.multitask.evaluation import atomic_json, file_sha
from our_mani.multitask.input_stat_cache_v79 import verify_inputs
from our_mani.multitask.case_output_isolation_v238 import require_development_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup', required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    if any(not v.replace('_', '').isalnum() for v in (args.setup, args.label)):
        raise ValueError('simple setup and job labels required')
    setup = ROOT/'artifacts/framework/multitask_r10'/args.setup
    receipt = setup/'queued_current_owner.json'
    if receipt.exists() or (setup/'queued_first_sources.json').exists():
        raise FileExistsError('setup already queued; inspect original execution')
    accepted = json.loads((setup/'CPU_acceptance.json').read_text())
    if accepted.get('complete') is not True:
        raise ValueError('complete actual-argv acceptance required')
    verify_inputs(ROOT, json.loads((setup/'frozen_inputs.json').read_text())['files'])
    owner = ROOT/'runtime/framework/native_resource_owner_v401'
    launch, status = (json.loads((owner/name).read_text()) for name in ('launch.json', 'status.json'))
    stat = Path(f'/proc/{launch["pid"]}/stat').read_text()
    if int(stat.rsplit(')', 1)[1].split()[19]) != launch['start_ticks'] or status['stop_requested']:
        raise ValueError('current resource owner identity unavailable')
    existing = []
    for directory in (ROOT/'runtime/framework/native_resource_owner_v19', owner):
        existing.extend(json.loads(p.read_text()) for p in (directory/'requests').glob('*.json'))
    requests, outputs = [], set()
    for row in accepted['actual_argv_preflights']:
        argv = row['argv']; task, raw, method = row['task'], row['raw'], row['method']
        output_arg = argv[argv.index('--out')+1]
        output = require_development_output(ROOT, output_arg, task=task, raw=raw, method=method)
        if (output.exists() or output in outputs or any('--out' in r['argv']
                and r['argv'][r['argv'].index('--out')+1] == output_arg for r in existing)):
            raise ValueError('case output already used or submitted')
        outputs.add(output)
        request = dict(schema='r10_owned_native_TRAIN_request_v19', kind='executor_smoke',
            job_id=args.label+'_'+task+'_'+raw+'_'+method.split('_')[0], argv=argv,
            use_gpu=method == 'physmani', timeout_seconds=1800,
            source_sha256=file_sha(ROOT/argv[2]))
        validate_request(request)
        requests.append(request)
    for index, request in enumerate(requests):
        atomic_json(owner/'requests'/f'0040_{args.label}_{index}_{request["job_id"]}.json', request)
    atomic_json(receipt, dict(queued_once=True, owner_pid=launch['pid'], owner_start_ticks=launch['start_ticks'],
        request_jobs=[r['job_id'] for r in requests], source_sha256=file_sha(__file__),
        original_policy_and_actual_CPU_proofs_unchanged=True, automatic_retry=False, formal_VAL_TEST=False))
    print(json.dumps(dict(queued=len(requests), jobs=[r['job_id'] for r in requests])), flush=True)


if __name__ == '__main__':
    main()
