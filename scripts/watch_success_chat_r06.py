"""Read-only active-session watcher; emits timing/phase events for chat reports."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/framework/aligned_r06'

def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}

def snapshot():
    status=read(BASE/'status.json')
    reports=[read(p) for p in BASE.glob('collection_*/report.json')]
    return dict(utc=datetime.now(timezone.utc).isoformat(),stage=status.get('stage'),
        completed=sum(r.get('completed',0) for r in reports),expected=1440,
        trained_heads=[p for p in ('cv','context') if (BASE/f'{p}_success/report.json').exists()],
        acceptance=read(BASE/'success_acceptance.json').get('passed'),
        error=status.get('error'),stop=(ROOT/'runtime/framework/success_r06_STOP').exists())

def main():
    p=argparse.ArgumentParser();p.add_argument('--until-epoch',type=float,required=True);args=p.parse_args()
    previous=snapshot();print(json.dumps(previous),flush=True)
    while time.time()<args.until_epoch:
        time.sleep(min(45.,max(0.,args.until_epoch-time.time())))
        current=snapshot();print(json.dumps(current),flush=True)
        if current['stop'] or current['error'] or current['stage']!=previous['stage'] or current['trained_heads']!=previous['trained_heads']:
            print(json.dumps(dict(phase_event=True,**current)),flush=True);return
        previous=current
    print(json.dumps(dict(chat_report_due=True,**snapshot())),flush=True)

if __name__=='__main__':main()
