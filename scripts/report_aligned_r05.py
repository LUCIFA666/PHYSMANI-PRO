"""Evidence-based r05 progress snapshots, scheduled every ten minutes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/framework/aligned_r05'

def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None

def main():
    now = datetime.now(timezone.utc)
    status = read(BASE/'status.json') or {'stage':'initializing'}
    lines = ['r05 十分钟进度报告', f'北京时间：{now.astimezone(ZoneInfo("Asia/Shanghai")):%Y-%m-%d %H:%M:%S}',
             f'阶段：{status.get("stage")}', status.get('description',''),
             '保留：Context/SimNorm、候选截获、碰撞筛选、成功率选择、MPC；FASTER 不使用。']
    reports = {}
    for path in sorted(BASE.glob('**/report.json')):
        report = read(path)
        if not report:
            continue
        key = str(path.relative_to(BASE))
        reports[key] = {k:v for k,v in report.items() if k in ('stage','complete','completed','expected','per_task','passed','failures','valid_initializations','valid_source_episodes','official_task_success_immediate','blockers')}
        if path.parent.name.startswith(('validation_','test_')):
            cases=[read(p) for p in path.parent.glob('pick*__episode*.json') if not p.name.endswith('infrastructure_error.json')]
            cases=[r for r in cases if r and 'official_task_success' in r]
            if cases:
                reports[key]['completed']=len(cases)
                reports[key]['expected']=80 if path.parent.name.startswith('test_') else 40
                reports[key]['complete']=len(cases)==reports[key]['expected']
                reports[key]['per_task']={t:dict(completed=sum(r['task']==t for r in cases),successes=sum(r['task']==t and r['official_task_success'] for r in cases)) for t in sorted({r['task'] for r in cases})}
        lines.append(f'{key}：'+json.dumps(reports[key], ensure_ascii=False))
    blockers = status.get('blockers', [])
    if blockers:
        lines.append('待解决问题：'+'；'.join(blockers))
    try:
        gpu = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.free','--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = 'unavailable'
    lines += [f'GPU（利用率、已用、空闲）：{gpu}', 'smoke/VAL 指标是开发诊断；正式 TEST 未完成前不宣称超过基线。']
    snapshot = dict(created_utc=now.isoformat(), status=status, reports=reports, gpu=gpu)
    directory = BASE/'progress'
    directory.mkdir(parents=True, exist_ok=True)
    name = now.strftime('%Y%m%dT%H%M%SZ')
    (directory/(name+'.json')).write_text(json.dumps(snapshot, ensure_ascii=False, indent=2)+'\n')
    body = '\n'.join(lines)+'\n'
    (directory/(name+'.txt')).write_text(body)
    for path in (ROOT/'十分钟进度报告.txt', ROOT/'最新项目状态.txt'):
        tmp = path.with_suffix('.r05.tmp'); tmp.write_text(body); tmp.replace(path)
    print(body, flush=True)

if __name__ == '__main__':
    main()
