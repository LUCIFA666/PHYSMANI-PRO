"""Same fixed high-speed remove policy with public registration metadata logs."""
from pathlib import Path
from types import MethodType
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scripts.run_task_executor_smoke_joint16_v174_r10 import parser
from our_mani.multitask.task_executor_registry import driver
from our_mani.multitask.remove_continuous_executor import RemoveContinuousBackend,verify
from our_mani.multitask.public_registration_diagnostic import install
from our_mani.multitask.evaluation import atomic_json,file_sha
from our_mani.multitask.frozen_subset_batch_v132 import checked_subset
SOURCES=('scripts/run_task_executor_smoke_registration_diagnostic_r10.py',
    'our_mani/multitask/public_registration_diagnostic.py')


def binding():
    return dict(schema='r10_current_public_registration_metric_only_v1',
        sources_sha256={p:file_sha(ROOT/p) for p in SOURCES},policy_changed=False,
        additional_camera_physics_or_native_truth_queries=0,formal_VAL_TEST=False)


class Backend(RemoveContinuousBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions);install(actor)
        return actor,identity


def main():
    args=parser().parse_args()
    if args.task!='remove_cups_from_rotating_frame_high_speed' or args.method!='cv_geometry':
        raise ValueError('one original fixed first high-speed remove CV source')
    module=driver(args.task);prepared=module.preflight(args);verify(ROOT,prepared)
    expected=binding()
    if prepared['config'].get('public_registration_diagnostic')!=expected:raise ValueError('freeze observer')
    checked_subset(ROOT,prepared['files'],{str(ROOT/p):sha for p,sha in expected['sources_sha256'].items()})
    if args.check:
        print(json.dumps(dict(complete=True,native_started=False,GPU_started=False,diagnostic=expected)))
        return
    record=module.run_prepared(prepared,backend=Backend(args.task))
    atomic_json(prepared['out']/'registration_diagnostic_entrypoint.json',dict(complete=True,diagnostic=expected))
    print(json.dumps(dict(complete=True,native_success=record['official_task_success'])))


if __name__=='__main__':main()
