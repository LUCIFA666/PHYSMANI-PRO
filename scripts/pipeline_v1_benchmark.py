"""Wait for TRAIN/VAL, freeze v1, check TRAIN integration, then evaluate TEST."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from our_mani.benchmark_protocol import file_sha256,verify_lock

OUT=ROOT/'artifacts/framework/pipeline_v1'
LOCK=OUT/'frozen_test_inputs.json'
STOP=ROOT/'runtime/framework/pipeline_v1_STOP'
PORT=18473


def write(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def status(stage,**extra):
    write(OUT/'benchmark_status.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'version':'pipeline_v1_context_simnorm','stage':stage,**extra})


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def wait_dependencies():
    while True:
        if STOP.exists():raise InterruptedError('v1 STOP')
        training=read(OUT/'training_status.json')
        faster=read(ROOT/'artifacts/framework/faster_pipeline_status.json')
        for item in (training,faster):
            if item.get('stage') in ('failed','stopped'):
                raise RuntimeError(f'dependency stopped: {item}')
        ready=training.get('stage')=='context_dynamics_MPC_and_success_ready_for_frozen_TEST'
        faster_ready=faster.get('stage')=='bounded_faster_training_and_val_checks_complete'
        if ready and faster_ready:return
        status('waiting_for_TRAIN_VAL_only_dependencies',formal_test_used=False,
            context_stage=training.get('stage'),faster_stage=faster.get('stage'))
        time.sleep(10)


def freeze():
    if LOCK.exists():
        lock=read(LOCK);verify_lock(ROOT,lock)
        return lock
    config=read(ROOT/'config/framework/pipeline_v1.json')
    best=read(ROOT/'artifacts/framework/faster_best_validation.json')
    config.update(faster_checkpoint=best['checkpoint'],faster_sha256=best['checkpoint_sha256'])
    receipt_path=Path(best['checkpoint'])/'our_mani_receipt.json'
    receipt=read(receipt_path)
    files=[ROOT/config[k] for k in ('dynamics','success','goal_calibration','test_manifest')]
    files += [(ROOT/config['success']).with_name('value_support.json')]
    files += sorted((ROOT/'our_mani').glob('*.py'))
    files += [ROOT/'config/framework/pipeline_v1.json',ROOT/'scripts/evaluate_pipeline_v1.py',
        ROOT/'scripts/serve_faster.py',ROOT/'config/framework/paper_reference_comparison.json',receipt_path]
    files += [ROOT/'third_party/FASTER/src/openpi'/name for name in
        ('policies/our_mani_policy.py','policies/policy.py','models/pi0_faster.py')]
    # Verify every immutable Orbax file before any TEST inference. The server
    # must later identify this same pinned receipt, never a mutable latest path.
    for entry in receipt['files']:
        path=Path(best['checkpoint'])/entry['path']
        if file_sha256(path)!=entry['sha256']:
            raise ValueError(f'pinned FASTER weights changed: {path}')
        files.append(path)
    lock={'created_utc':datetime.now(timezone.utc).isoformat(),'config':config,
        'authorization':'用户要求：用 PhysMani 数据集与 context + SimNorm pipeline v1 对比',
        'checkpoint_selection':'dynamics TRAIN-heldout, success TRAIN-heldout/calibration, FASTER fixed VAL XYZ error; no TEST selection',
        'test_tuning_permitted':False,'files':{str(p.relative_to(ROOT)):file_sha256(p) for p in files}}
    write(LOCK,lock)
    return lock


def run(stage,command):
    status(stage,formal_test_used=stage=='official_TEST_80_running',command=command)
    with (ROOT/f'logs/framework/{stage}.log').open('a') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    server=None
    try:
        wait_dependencies()
        status('freezing_inputs_before_TEST',formal_test_used=False)
        lock=freeze()
        with (ROOT/'logs/framework/pipeline_v1_faster_server.log').open('a') as log:
            server=subprocess.Popen([str(ROOT/'.faster-env/bin/python'),'scripts/serve_faster.py',
                '--checkpoint',lock['config']['faster_checkpoint'],'--port',str(PORT)],cwd=ROOT,
                env={**os.environ,'XLA_PYTHON_CLIENT_PREALLOCATE':'false','OMP_NUM_THREADS':'4',
                     'OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'},stdout=log,stderr=subprocess.STDOUT)
            for _ in range(90):
                if STOP.exists():raise InterruptedError('v1 STOP during serving setup')
                if server.poll() is not None:raise RuntimeError('FASTER server exited; see v1 server log')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/healthz',timeout=2) as response:
                        if response.status==200:break
                except OSError:time.sleep(2)
            else:raise TimeoutError('v1 FASTER server startup exceeded 180 seconds')
            common=['bash','scripts/run_local_sim.sh','scripts/evaluate_pipeline_v1.py','--lock',str(LOCK.relative_to(ROOT)),
                    '--port',str(PORT)]
            run('pipeline_v1_TRAIN_integration_check',[*common,'--mode','development',
                '--out','artifacts/framework/pipeline_v1/development_home'])
            # Infrastructure completion, not development success, is the gate.
            # A weak full policy is still evaluated and all failures retained.
            session=read(ROOT/'config/framework/current_run.json')
            session['formal_test']='authorized_frozen_pipeline_v1_context_simnorm_evaluation'
            write(ROOT/'config/framework/current_run.json',session)
            run('official_TEST_80_running',[*common,'--mode','test','--out','artifacts/framework/pipeline_v1/official_test'])
        status('complete',formal_test_used=True,report='artifacts/framework/pipeline_v1/official_test/report.json')
        report=read(OUT/'official_test/report.json')
        lines=['Context + SimNorm pipeline v1 与 PhysMani 论文结果对比',
            '基线数据来自论文 Table 1，未在本地复现；ours 是单一冻结检查点。',
            '采用官方 TEST episode、variation 和复位顺序；ours 20Hz 控制及抓爪执行器不同，预算为名义10秒。']
        for task,row in report['per_task'].items():
            paper=row['paper_reported_PhysMani'];ci=row['wilson_95']
            lines.append(f'{task}: ours={row["successes"]}/{row["completed"]} ({100*row["success_rate"]:.1f}%), Wilson95%=[{100*ci[0]:.1f},{100*ci[1]:.1f}]%; PhysMani论文={paper["mean_percent"]}±{paper["std_percent"]}%')
        lines += ['论文±是最佳3/最后5检查点的标准差，与ours的二项置信区间含义不同。',
                  '完整日志：artifacts/framework/pipeline_v1/official_test/report.json']
        (ROOT/'v1与PhysMani首轮对比.txt').write_text('\n'.join(lines)+'\n')
        subprocess.run([sys.executable,'scripts/overnight_report.py'],cwd=ROOT,check=True)
    except Exception as error:
        status('failed',error=repr(error),formal_test_used=(OUT/'official_test').exists())
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:server.kill();server.wait()


if __name__=='__main__':main()
