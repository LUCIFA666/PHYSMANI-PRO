"""Scheduled evidence snapshots while repairing and evaluating success models."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/framework/aligned_r06'

def read(p):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return {}

def main():
    now=datetime.now(timezone.utc);status=read(BASE/'status.json')
    if (BASE/'collection_inputs.json').exists():
        audit=subprocess.run([sys.executable,'scripts/audit_candidate_data_r06.py'],cwd=ROOT,capture_output=True,text=True)
        if audit.returncode:
            (ROOT/'runtime/framework/success_r06_STOP').write_text('candidate data audit failed\n')
            status.update(stage='data_quality_needs_attention',description='候选数据审计未通过，已阻止继续采集/拟合，待修复。',audit_error=audit.stderr[-1500:])
            (BASE/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    lines=['r06 成功率模块修复进度',f'北京时间：{now.astimezone(ZoneInfo("Asia/Shanghai")):%Y-%m-%d %H:%M:%S}',
        f"阶段：{status.get('stage')}",status.get('description',''),
        'r05 对比已按用户要求停止；原始权重、结果保留。FASTER 排除，NGFF 训练不重启。']
    collections={}
    for predictor in ('cv','context'):
        for split in ('train','validation'):
            reports=[read(p) for p in BASE.glob(f'collection_{predictor}_{split}_*/report.json')]
            row={k:sum(r.get(k,0) for r in reports) for k in ('completed','expected','attempts','positive','source_episodes')}
            row['expected']= (200 if split=='train' else 40)*3
            collections[predictor+'_'+split]=row
            lines.append(f"{predictor} {split}：{row['completed']}/{row['expected']} 完整执行，候选尝试 {row['attempts']}，成功 {row['positive']}。")
    models={}
    for predictor in ('cv','context'):
        report=read(BASE/f'{predictor}_success/report.json')
        if report:
            models[predictor]={k:report.get(k) for k in ('train_attempts','validation_attempts','per_task','calibration_source_crossfit')}
            lines.append(predictor+'概率模型：'+json.dumps(models[predictor],ensure_ascii=False))
    evaluation={}
    for directory in sorted(list(BASE.glob('validation_*'))+list(BASE.glob('test_*'))):
        if not directory.is_dir():continue
        rows=[read(p) for p in directory.glob('pick*__episode*.json') if not p.name.endswith('infrastructure_error.json')]
        rows=[r for r in rows if 'official_task_success' in r]
        if not rows:continue
        evaluation[directory.name]={t:dict(n=len(rr:=[r for r in rows if r['task']==t]),successes=sum(r['official_task_success'] for r in rr)) for t in sorted({r['task'] for r in rows})}
        lines.append(directory.name+'：'+json.dumps(evaluation[directory.name],ensure_ascii=False))
    gate=read(BASE/'success_acceptance.json')
    if gate:lines.append('成功率模块验收：'+json.dumps(gate,ensure_ascii=False))
    try:gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):gpu='unavailable'
    lines.extend([f'GPU：{gpu}','训练/VAL指标用于开发，正式TEST完成前不宣称超过基线。'])
    directory=BASE/'progress';directory.mkdir(parents=True,exist_ok=True);name=now.strftime('%Y%m%dT%H%M%SZ')
    snapshot=dict(created_utc=now.isoformat(),status=status,collections=collections,models=models,evaluation=evaluation,gate=gate,gpu=gpu)
    (directory/(name+'.json')).write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    body='\n'.join(lines)+'\n';(directory/(name+'.txt')).write_text(body)
    for path in (ROOT/'十分钟进度报告.txt',ROOT/'最新项目状态.txt'):
        tmp=path.with_suffix('.r06.tmp');tmp.write_text(body);tmp.replace(path)
    print(body,flush=True)
    if status.get('stage')=='complete':
        subprocess.run(['systemctl','--user','stop','our-mani-r05-report.timer'],check=False)

if __name__=='__main__':main()
