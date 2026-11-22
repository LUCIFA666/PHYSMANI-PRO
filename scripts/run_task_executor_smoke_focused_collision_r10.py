"""Replay one fixed TRAIN case with unchanged policy and focused collision logs."""
from pathlib import Path
from types import MethodType
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scripts.run_task_executor_smoke_joint16_v174_r10 import parser
from our_mani.multitask.task_executor_registry import driver,verify,SharedTaskBackend
from our_mani.multitask.focused_collision_observer import FocusedCollisionObserver
from our_mani.multitask.frozen_subset_batch_v132 import checked_subset
from our_mani.multitask.evaluation import atomic_json,file_sha
SOURCES=('our_mani/multitask/focused_collision_observer.py',
    'our_mani/multitask/public_collision_diagnostic_v6.py',
    'scripts/run_task_executor_smoke_focused_collision_r10.py')


def binding(config,task):
    return dict(schema='r10_focused_existing_collision_queries_v1',task=task,phases=['insert'],
        underlying_executor=config['shared_task_executor'],sources_sha256={p:file_sha(ROOT/p) for p in SOURCES},
        added_physics_camera_or_collision_pair_queries=0,policy_changed=False,formal_VAL_TEST=False)


class Backend(SharedTaskBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        observer=FocusedCollisionObserver(actor,env,phases=tuple(prepared['config']['focused_collision_diagnostic']['phases']))
        previous=actor.metrics_report
        def metrics(instance):return dict(previous(),focused_collision_diagnostic=observer.receipt())
        actor.metrics_report=MethodType(metrics,actor)
        return actor,identity


def main():
    args=parser().parse_args()
    if args.task!='insert_onto_rotating_peg' or args.method!='context_geometry':
        raise ValueError('one declared original TRAIN source and method; no outcome-based resampling')
    module=driver(args.task);prepared=module.preflight(args);verify(ROOT,prepared)
    expected=binding(prepared['config'],args.task)
    if prepared['config'].get('focused_collision_diagnostic')!=expected:raise ValueError('freeze focused observer')
    checked_subset(ROOT,prepared['files'],{str(ROOT/p):sha for p,sha in expected['sources_sha256'].items()})
    if args.check:
        print(json.dumps(dict(complete=True,native_started=False,GPU_started=False,identity=prepared['identity'],diagnostic=expected)))
        return
    record=module.run_prepared(prepared,backend=Backend(args.task))
    atomic_json(prepared['out']/'focused_diagnostic_entrypoint.json',dict(complete=True,diagnostic=expected,
        native_success=record['official_task_success'],formal_VAL_TEST=False))
    print(json.dumps(dict(complete=True,native_success=record['official_task_success'],case=str(prepared['out']/'case.json'))))


if __name__=='__main__':main()
