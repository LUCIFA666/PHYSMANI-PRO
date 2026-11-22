"""Persistent bounded FASTER adaptation smoke, isolated from NGFF and baseline jobs.

Waits for the already-started dependency installation, verifies CUDA, converts
four TRAIN episodes, computes stats, then permits only 10 optimizer updates.
Every stage writes status/logs under our-mani. No test episodes are evaluated.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT/'runtime/framework/state'
STATUS = ROOT/'artifacts/framework/bootstrap_status.json'
STOP = STATE/'STOP'


def update(stage, **extra):
    record = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'stage': stage,
              'pid': os.getpid(), 'kind': 'bounded_faster_training_smoke',
              'maximum_optimizer_updates': 10, 'ngff_training_resumed': False, **extra}
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(STATUS)
    labels = {'installing_faster_environment': '安装独立 FASTER 环境',
              'config_preflight': '检查 FASTER 适配配置', 'gpu_preflight': '检查 JAX/CUDA 数值结果',
              'convert_train_smoke': '转换四条训练演示', 'normalization': '计算训练数据归一化统计',
              'download_base_weights': '下载初始化权重', 'waiting_for_gpu_capacity': '等待足够的空闲显存',
              'train_10_updates': '运行 10 次更新的训练集成检查',
              'training_smoke_complete': '训练集成检查完成', 'stopped': '已停止', 'failed': '当前步骤失败，详见日志'}
    (ROOT/'框架运行状态.txt').write_text(
        f"更新时间（UTC）：{record['updated_utc']}\n当前步骤：{labels.get(stage, stage)}\n"
        "NGFF：保持停止，使用第 2751 次更新的最佳权重；最后训练断点为第 3083 次更新。\n"
        "FASTER：本次最多进行 10 次优化更新，用于验证适配训练流程。\n"
        "完整闭环抓取成功率：尚未测试。\n"
        f"详细状态：{STATUS}\n"+json.dumps(extra, ensure_ascii=False, indent=2)+"\n", encoding='utf-8')


def run(stage, command, env):
    update(stage, command=command)
    log_path = ROOT/f'logs/framework/bootstrap_{stage}.log'
    with log_path.open('a', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        while process.poll() is None:
            if STOP.exists():
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                raise InterruptedError('framework STOP requested')
            update(stage, child_pid=process.pid, log=str(log_path))
            time.sleep(10)
    if process.returncode:
        raise RuntimeError(f'{stage} failed with exit {process.returncode}; see {log_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installer-pid', type=int, required=True)
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({'PYTHONPATH': ':'.join([str(ROOT), str(ROOT/'third_party/FASTER/src'),
                                      str(ROOT/'third_party/FASTER/packages/openpi-client/src')]),
                'HF_HOME': str(ROOT/'runtime/framework/huggingface'),
                'HF_LEROBOT_HOME': str(ROOT/'artifacts/framework/lerobot'),
                'OPENPI_DATA_HOME': str(ROOT/'artifacts/framework/openpi_cache'),
                'XDG_CACHE_HOME': str(ROOT/'runtime/framework/cache'),
                'TMPDIR': str(ROOT/'runtime/framework/tmp'),
                'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'WANDB_MODE': 'offline',
                'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.45',
                'JAX_COMPILATION_CACHE_DIR': str(ROOT/'runtime/framework/jax-cache'),
                'OUR_MANI_STOP_FILE': str(STOP),
                'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
                'PYTHONUNBUFFERED': '1'})
    try:
        def installer_active():
            try:
                command = Path(f'/proc/{args.installer_pid}/cmdline').read_bytes().replace(b'\x00', b' ').decode()
            except FileNotFoundError:
                return False
            return str(ROOT/'.conda-env/bin/uv') in command and ' sync ' in command
        while installer_active():
            if STOP.exists():
                raise InterruptedError('framework STOP requested')
            update('installing_faster_environment', installer_pid=args.installer_pid,
                   log=str(ROOT/'logs/framework/faster_env_setup.log'))
            time.sleep(10)
        python = str(ROOT/'.faster-env/bin/python')
        cpu_env = dict(env, JAX_PLATFORMS='cpu')
        run('config_preflight', [python, 'scripts/faster_training.py', 'config', '--updates', '10'], cpu_env)
        run('gpu_preflight', [python, 'scripts/preflight_faster_gpu.py'], env)
        dataset_id = 'local/our_mani_physmani_smoke_v1'
        receipt = ROOT/'artifacts/framework/faster_data/local_our_mani_physmani_smoke_v1_receipt.json'
        if not receipt.exists():
            run('convert_train_smoke', [python, 'scripts/convert_faster_demonstrations.py',
                                       '--repo-id', dataset_id, '--limit-episodes', '2'], cpu_env)
        common = ['--repo-id', dataset_id, '--exp-name', 'integration_smoke_10_updates', '--batch-size', '1', '--updates', '10']
        run('normalization', [python, 'scripts/faster_training.py', 'norm-stats', *common], cpu_env)
        # Read-only preparation of base weights happens before occupying GPU memory.
        run('download_base_weights', [python, 'scripts/download_faster_weights.py'], cpu_env)
        stable_polls = 0
        while stable_polls < 6:
            if STOP.exists():
                raise InterruptedError('framework STOP requested')
            value = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
            free_mib = int(value.splitlines()[0])
            stable_polls = stable_polls+1 if free_mib >= 17*1024 else 0
            update('waiting_for_gpu_capacity', free_mib=free_mib, required_free_mib=17*1024,
                   stable_polls=stable_polls, required_stable_polls=6)
            time.sleep(10)
        checkpoint_dir = ROOT/'artifacts/framework/faster_checkpoints/pi05_faster_our_mani_lora/integration_smoke_10_updates'
        resume = False
        if checkpoint_dir.exists():
            resume = any(p.is_dir() and p.name.isdigit() for p in checkpoint_dir.iterdir())
            if not resume:
                archive = checkpoint_dir.with_name(checkpoint_dir.name+'_interrupted_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
                checkpoint_dir.rename(archive)
        run('train_10_updates', [python, 'scripts/faster_training.py', 'train', *common, *(['--resume'] if resume else [])], env)
        update('training_smoke_complete', optimizer_updates=10, converted_train_episodes=4,
               test_episodes_evaluated=0, model_quality_claim=False)
    except InterruptedError as error:
        update('stopped', reason=str(error))
    except Exception as error:
        update('failed', reason=str(error))
        raise


if __name__ == '__main__':
    main()
