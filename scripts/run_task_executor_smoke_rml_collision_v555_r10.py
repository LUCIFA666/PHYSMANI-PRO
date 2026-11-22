"""One fixed TRAIN high-bin CV case with unchanged RML and read-only witnesses."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scripts import run_task_executor_smoke_physmani_skills_v535_r10 as source
from our_mani.multitask.all_phase_collision_diagnostic_v17 import CollisionObserver,binding
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.frozen_subset_batch_v132 import checked_subset


def main():
    args=source.button_source.parser().parse_args()
    if args.task!='put_rubbish_in_moving_bin_high_speed' or args.method!='cv_geometry':
        raise ValueError('single fixed original high-bin CV diagnostic only')
    p=source.preflight(args);expected=binding(ROOT,p['config']['baseline_repo'])
    if p['config'].get('RML_public_collision_diagnostics_v554')!=expected:
        raise ValueError('freeze read-only public collision observer')
    checked_subset(ROOT,p['files'],{str(ROOT/n):file_sha(ROOT/n) for n in (
        'our_mani/multitask/all_phase_collision_diagnostic_v17.py',
        'scripts/run_task_executor_smoke_rml_collision_v555_r10.py')})
    if args.check:
        print(json.dumps(dict(complete=True,native_started=False,GPU_started=False,identity=p['identity'],diagnostic=expected)))
        return
    original=source.components;observers=[]
    class Backend:
        def __init__(self,backend):self.backend=backend
        def __getattr__(self,name):return getattr(self.backend,name)
        def actor(self,*a,**kw):
            actor,identity=self.backend.actor(*a,**kw)
            observers.append(CollisionObserver(actor,a[1]));return actor,identity
    def components(task):
        driver,backend=original(task);return driver,Backend(backend)
    source.components=components
    try:r=source.run(p)
    finally:
        source.components=original
        if observers:
            atomic_json(p['out']/'public_RML_collision_diagnostics.json',dict(
                identity=p['identity'],observers=[o.receipt() for o in observers],
                native_policy_bytes_unchanged=True,formal_VAL_TEST=False))
    print(json.dumps(dict(complete=True,native_success=r['official_task_success'],case=str(p['out']/'case.json'))))


if __name__=='__main__':main()
