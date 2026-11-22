"""Fresh centre-only continuation after the preserved deterministic-pooling failure.

The closed 40-parent data are reused. No capture, automatic retry, or Native
evaluation is performed. Child CLIs hold the canonical Native/GPU mutexes.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import signal

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('frozen_large_continuation', ROOT / 'scripts/run_large_network_training_continuation_20261104_v1.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
SCHEMA = 'centre_large_v2_continuation_20261104_v1'
DEFAULT = ROOT / 'config/framework' / SCHEMA / 'plan.json'
STAGES = base.STAGES[6:]
LOCK = ROOT / 'runtime/framework' / (SCHEMA + '.lock')


def validate_previous(config):
    previous = base.read(base.checked_pointer(config['previous_root_exit']))
    if previous.get('root_session') != 23027 or previous.get('actual_process_returncode') != 1 or previous.get('actual_wait_completed') is not True:
        raise ValueError('Actual Root wait for the preserved failed predecessor is required')
    terminal = base.read(base.checked_pointer(previous['terminal_status']))
    failure = base.read(base.checked_pointer(previous['failure']))
    if terminal.get('complete') is not False or terminal.get('error') != 'Stage actual nonzero exit; no retry: centre_GPU_preflight' or failure.get('error') != terminal['error']:
        raise ValueError('Only the explicitly diagnosed predecessor failure is admitted')
    ledger = [json.loads(line) for line in base.checked_pointer(previous['stage_ledger']).read_text().splitlines() if line.strip()]
    if tuple(row['stage'] for row in ledger) != base.STAGES[:6]:
        raise ValueError('The exact six completed predecessor stages are required')
    for row in ledger:
        if row.get('actual_process_returncode') != 0 or row.get('actual_wait_completed') is not True or row.get('source_unchanged') is not True or row.get('unreaped_descendants'):
            raise ValueError('Predecessor stages must be actually waited and clean')
    child_path = base.checked_pointer(previous['failed_child'])
    child = base.read(child_path)
    if child.get('stage') != 'centre_GPU_preflight' or child.get('actual_process_returncode') != 1 or child.get('actual_wait_completed') is not True or child.get('source_unchanged') is not True or child.get('unreaped_descendants'):
        raise ValueError('The failed preflight must be actually reaped without source edits')
    log = child_path.with_name('stdout.log').read_text()
    if 'adaptive_avg_pool2d_backward_cuda does not have a deterministic implementation' not in log:
        raise ValueError('This successor only corrects the recorded deterministic-pooling failure')
    old_config = base.load_config(base.checked_pointer(config['previous_config']))
    admission, plan = base.validate_CF_admission(old_config, old_config['CF_final_admission'])
    base.validate_QA_admission(old_config, old_config['QA_admission'], Path(old_config['output_root']) / 'artifacts/CF_first_fit_QA', admission['capture_plan'])
    closed = base.queue_audit(old_config, admission, plan, count=40)
    receipt = base.read(base.checked_pointer(config['compiled_receipt']))
    if receipt.get('complete') is not True or receipt.get('decoded_fit_samples') != 208 or receipt.get('decoded_selection_samples') != 208 or receipt.get('decoded_reserved_samples') != 0:
        raise ValueError('The same complete compiled cohort is required, with reserved labels unopened')
    manifest = base.checked_pointer(receipt['manifest'])
    if base.pointer(manifest) != config['manifest']:
        raise ValueError('Compiled manifest identity mismatch')
    if (manifest.parent / 'reserved_centre_evaluation_consumption.request.json').exists():
        # Allowed after this owner's own reserved prediction; never at startup.
        claim = base.read(manifest.parent / 'reserved_centre_evaluation_consumption.request.json')
        own = Path(config['output_root']) / 'artifacts/centre_reserved_fresh_centre_VAL/request.json'
        if not own.exists() or base.read(own) != claim:
            raise ValueError('Reserved cohort was consumed by another owner')
    return dict(previous_root_exit=config['previous_root_exit'], all40_closed=len(closed) == 40, manifest=base.pointer(manifest))


def load_config(path, expected=None):
    path = Path(path).resolve(strict=True)
    if expected is not None and base.sha(path) != expected:
        raise ValueError('External frozen successor config SHA required')
    config = base.read(path)
    if config.get('schema') != SCHEMA or tuple(config.get('stages', ())) != STAGES or config.get('Native_VAL_TEST_authorized') is not False:
        raise ValueError('Fixed centre-only successor protocol required')
    for key in ('python', 'output_root', 'CF_output_root'):
        if not Path(config[key]).is_absolute():
            raise ValueError('Absolute paths required')
    if ROOT / 'artifacts/framework/local_mpc_20261102' not in Path(config['output_root']).parents:
        raise ValueError('Independent successor output required')
    for key in ('centre_delivery', 'encoder', 'method_sources', 'manifest', 'previous_root_exit', 'previous_config', 'compiled_receipt'):
        base.checked_pointer(config[key])
    if config['encoder']['sha256'] != 'dbf998242ad13fea26ede9c4b17a0e2bfd0daeb581331d21342af9a6b6d06bfa':
        raise ValueError('Original encoder initializer required')
    config['_config_path'] = str(path)
    config['_config_sha256'] = base.sha(path)
    base.verify_sources(config)
    return config


def execute(path, digest):
    config = load_config(path, digest)
    output = Path(config['output_root'])
    with base.driver_lock(LOCK):
        if output.exists():
            raise FileExistsError('Fresh successor owner only; no output retry')
        base.assert_running(config)
        closure = validate_previous(config)
        output.mkdir(parents=True, exist_ok=False)
        artifact = output / 'artifacts'
        artifact.mkdir()
        (output / 'owners').mkdir()
        base.enable_subreaper()
        base.write_new(output / 'request.json', dict(config=base.pointer(path), source_map=config['method_sources'], prior=closure, started_utc=base.now(), Native_started=False))
        previous_signals = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_signals[signum] = signal.signal(signum, lambda number, frame: base.SIGNAL_REQUEST.update(signal=number))
        status = dict(complete=False, completed_stages=[], previous_root_exit=config['previous_root_exit'], compiled_manifest=config['manifest'], current_network_end_to_end_ready=False, model_or_runtime_qualified=False, Native_VAL_TEST_started=False)

        def run(name, args, cpu=False):
            base.verify_sources(config)
            value = base.run_stage(config, name, args, cpu=cpu)
            if value.get('actual_process_returncode') != 0 or value.get('actual_wait_completed') is not True or value.get('source_unchanged') is not True or value.get('unreaped_descendants'):
                raise ValueError('Actual clean child wait/source required')
            status['completed_stages'].append(name)
            base.append(output / 'stage_ledger.jsonl', value)

        try:
            encoder = config['encoder']
            data = ['--manifest', config['manifest']['path'], '--encoder-checkpoint', encoder['path'], '--expected-encoder-sha256', encoder['sha256']]
            trainer = 'scripts/train_current_cube_dual_crop_centre_large_20261104_v2.py'
            preflight = artifact / 'centre_GPU_preflight'
            run(STAGES[0], [trainer, '--gpu-preflight'] + data + ['--output', str(preflight), '--device', 'cuda:0'])
            base.require_complete(preflight / 'gpu_preflight.json', passed=True, actual_optimizer_steps=1, model_saved=False, updated_weights_discarded=True)
            fit = artifact / 'centre_fit'
            run(STAGES[1], [trainer, '--train'] + data + ['--preflight', str(preflight / 'gpu_preflight.json'), '--expected-preflight-sha256', base.sha(preflight / 'gpu_preflight.json'), '--output', str(fit), '--device', 'cuda:0'])
            trained = base.require_complete(fit / 'fit_status.json', training_complete=True, epochs_completed=40, actual_optimizer_steps=520)
            selected = base.checked_pointer(trained['selected_checkpoint'])
            if selected != fit / 'selected.pt':
                raise ValueError('Actual selected checkpoint sibling required')
            evaluator = 'scripts/evaluate_current_cube_dual_crop_centre_large_20261104_v2.py'
            for role, count, start in [('selection', 208, 2), ('reserved_fresh_centre_VAL', 104, 4)]:
                prediction = artifact / ('centre_' + role)
                run(STAGES[start], [evaluator, '--predict', '--role', role, '--manifest', config['manifest']['path'], '--selected', str(selected), '--expected-sha256', base.sha(selected), '--encoder-checkpoint', encoder['path'], '--output', str(prediction)], cpu=True)
                frozen = base.require_complete(prediction / 'prediction_freeze.json', complete=True, role=role, samples=count, torch_num_threads=2, target_label_array_decodes_before_freeze=0)
                for pointer in frozen['files'].values():
                    base.checked_pointer(pointer)
                run(STAGES[start + 1], [evaluator, '--score', '--output', str(prediction)], cpu=True)
                report = base.read(prediction / 'report.json')
                if report.get('role') != role or len(report.get('rows', [])) != count or report.get('model_qualified') is not False:
                    raise ValueError('Fixed complete report required, without automatic runtime admission')
            base.verify_sources(config)
            status.update(complete=True, selected_centre=base.pointer(selected), finished_utc=base.now())
            base.write_new(output / 'completion.json', status)
            return status
        except BaseException as error:
            status.update(error_type=type(error).__name__, error=str(error))
            base.write_new(output / 'failure.json', status)
            raise
        finally:
            base.write_new(output / 'terminal_status.json', dict(status, finished_utc=base.now()))
            for signum, handler in previous_signals.items():
                signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--config-sha256')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.execute:
        if not args.config_sha256:
            parser.error('--execute requires externally frozen --config-sha256')
        result = execute(args.config, args.config_sha256)
    else:
        config = load_config(args.config, args.config_sha256)
        result = dict(complete=True, preparation_only=True, prior=validate_previous(config), stages=list(STAGES), child_processes_started=0, GPU_used=False, Native_started=False, model_forward_calls=0)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
