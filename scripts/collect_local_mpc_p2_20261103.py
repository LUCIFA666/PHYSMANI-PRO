"""Serial actual-transition collection with preassigned episode splits."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def main():
    plan_path, output = map(Path, sys.argv[1:])
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(plan_path.read_text())
    family = ROOT/'runtime/framework/local_mpc_20261102'
    results = []
    for job in plan['jobs']:
        if (family/'STOP').exists() or (output/'STOP').exists():
            write(output/'stopped.json', dict(completed=results, user_stop=True))
            return 2
        argv = [sys.executable, str(ROOT/'scripts/launch_local_mpc_20261102.py'),
                '--mode','p1','--scene',job['scene'],'--config',job['config'],'--output',job['path']]
        print(json.dumps(dict(event='start', episode=job['episode_id'], split=job['split'])),flush=True)
        process = subprocess.run(argv,cwd=ROOT)
        row = dict(episode_id=job['episode_id'], parent_id=job['parent_id'], split=job['split'],
                   actual_exit_code=process.returncode, finished_utc=datetime.now(timezone.utc).isoformat())
        case = ROOT/job['path']
        if process.returncode == 0:
            result = json.loads((case/'result.json').read_text())
            row.update(native_success=result['native_success_observed'], milestone_passed=result['milestone_passed'],
                       cycles=result['cycles'], p95=result['full_cycle_wall_seconds']['p95'])
        write(output/(job['episode_id']+'.json'),row)
        results.append(row)
        print(json.dumps(dict(event='finished', **row)),flush=True)
        if process.returncode != 0:
            write(output/'failed.json',dict(complete=False,results=results,reason='worker infrastructure or stop'))
            return process.returncode
    write(output/'result.json',dict(complete=True,episodes=len(results),results=results,
                                  formal_TEST_read=False, training_started=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
