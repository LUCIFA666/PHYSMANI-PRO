"""Moving peg tracking after current measured-grasp calibration, native TRAIN."""
from pathlib import Path
from .task_executor_registry import SharedTaskBackend
from .moving_command_canonical import install as install_live
from .grasp_frame_compensation_native_compat import install as install_compensation,binding as compensation_binding
from .evaluation import file_sha
from .frozen_subset_batch_v132 import checked_subset
SOURCES=('our_mani/multitask/live_grasp_executor_native.py','our_mani/multitask/moving_command_canonical.py',
    'our_mani/multitask/grasp_frame_compensation_native_compat.py')


def binding(root,config,task):
    if task!='insert_onto_rotating_peg':raise ValueError('normal-ring live frame pilot first')
    return dict(schema='r10_measured_grasp_live_peg_control_original_canonical_completion_v1',task=task,
        compensation=compensation_binding(root,config,task),phases=['align','insert'],replan_seconds=.1,
        sources_sha256={p:file_sha(Path(root)/p) for p in SOURCES},
        old_committed_control_frame_explicitly_replaced_by_current_causal_forecast=True,
        actual_canonical_completion_after_grasp_compensation_retained=True,
        original_full_motion_collision_grasp_and_global10s_unchanged=True,formal_VAL_TEST=False)


def verify(root,prepared):
    expected=binding(root,prepared['config'],prepared['args'].task)
    if prepared['config'].get('moving_command_executor')!=expected:raise ValueError('freeze live command tracking')
    if prepared['config'].get('grasp_frame_compensation')!=expected['compensation']:raise ValueError('frozen current grasp calibration required')
    required={str(Path(root)/p):sha for p,sha in expected['sources_sha256'].items()}
    required.update({str(Path(root)/p):sha for p,sha in expected['compensation']['sources_sha256'].items()})
    prior=expected['compensation']['prior'];required[str(Path(root)/prior['path'])]=prior['sha256']
    checked_subset(root,prepared['files'],required)
    return expected


class LiveGraspBackend(SharedTaskBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        root=Path(__file__).resolve().parents[2];task=prepared['args'].task
        expected=binding(root,prepared['config'],task);holder={'events':[]}
        install_live(actor,phases=tuple(expected['phases']),replan_seconds=expected['replan_seconds'],
            ready=lambda:any(e['event']=='measured_grasp_frame_placement_compensation_admitted' for e in holder['events']))
        holder['events']=install_compensation(actor,prepared['config'],root,task)
        return actor,identity
