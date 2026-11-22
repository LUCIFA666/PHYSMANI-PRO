"""Serial public-controller canaries; GT metrics run only after worker exit."""
from datetime import datetime,timezone
import hashlib,json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def write_new(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')

def main():
    plan_path,owner=map(Path,sys.argv[1:]);owner.mkdir(parents=True,exist_ok=False)
    identity=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plan_raw=plan_path.read_bytes();plan=json.loads(plan_raw);results=[]
    write_new(owner/'inputs.json',dict(plan=str(plan_path),plan_sha256=hashlib.sha256(plan_raw).hexdigest(),runner_sha256=identity))
    for job in plan['jobs']:
        if any(p.exists() for p in (owner/'STOP',ROOT/'runtime/framework/local_mpc_20261102/STOP')):
            write_new(owner/'stopped.json',dict(complete=False,user_stop=True,results=results));return 2
        print(json.dumps(dict(event='start',case=job['id'])),flush=True)
        argv=[sys.executable,str(ROOT/'scripts/launch_local_mpc_20261102.py'),'--mode','p1',
              '--scene',job['scene'],'--config',job['config'],'--output',job['path']]
        process=subprocess.run(argv,cwd=ROOT)
        if process.returncode:
            write_new(owner/'failed.json',dict(complete=False,case=job['id'],actual_returncode=process.returncode,results=results));return process.returncode
        argv=[str(ROOT/'.conda-env/bin/python'),'-m','our_mani.local_mpc_control_20261103.native_task_evaluation',
              '--case',job['path'],'--output',job['evaluation']]
        evaluation=subprocess.run(argv,cwd=ROOT,stdout=subprocess.DEVNULL)
        if evaluation.returncode:raise RuntimeError('offline evaluation failed after '+job['id'])
        metrics=json.loads((Path(job['evaluation'])/'report.json').read_text())['metrics']
        result=json.loads((Path(job['path'])/'result.json').read_text())
        row=dict(id=job['id'],Native_original_success=metrics['native_success_any'],
            original_goal_plus_strict_current_hold=metrics['strict_current_joint_condition'],
            actual_hold=metrics['p1_hold_metric'],public_visual_hold=result.get('public_lift_hold_observed'),
            finished_utc=datetime.now(timezone.utc).isoformat(),formal_TEST_used=False)
        write_new(owner/(job['id']+'.json'),row);results.append(row);print(json.dumps(dict(event='finished',**row)),flush=True)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=identity or plan_path.read_bytes()!=plan_raw:
        raise RuntimeError('canary runner or plan changed during suite')
    write_new(owner/'result.json',dict(complete=True,n=len(results),results=results,formal_TEST_used=False,training_started=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
