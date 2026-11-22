"""Calibrated live peg tracking with a canonical endpoint and zero local twist."""
from pathlib import Path
from .task_executor_registry import SharedTaskBackend
from .live_grasp_executor_native import binding as live_binding
from .moving_command_canonical import install as install_live
from .canonical_terminal_curve import install as install_terminal
from .grasp_frame_compensation_native_compat import install as install_compensation
from .evaluation import file_sha
from .frozen_subset_batch_v132 import checked_subset
SOURCES=('our_mani/multitask/terminal_live_grasp_executor.py','our_mani/multitask/canonical_terminal_curve.py')


def binding(root,config,task):
    return dict(schema='r10_canonical_zero_local_twist_live_grasp_executor_v1',
        live_control=live_binding(root,config,task),
        sources_sha256={p:file_sha(Path(root)/p) for p in SOURCES},
        curve_time_parameter='smoothstep geometry; zero local endpoint twist; original duration',
        original_current_canonical_completion_and_external_frame_twist_retained=True,formal_VAL_TEST=False)


def verify(root,prepared):
    from .live_grasp_executor_native import verify as verify_live
    verify_live(root,prepared);expected=binding(root,prepared['config'],prepared['args'].task)
    if prepared['config'].get('terminal_live_grasp_executor')!=expected:raise ValueError('freeze canonical endpoint control')
    checked_subset(root,prepared['files'],{str(Path(root)/p):sha for p,sha in expected['sources_sha256'].items()})
    return expected


class TerminalLiveGraspBackend(SharedTaskBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        root=Path(__file__).resolve().parents[2];task=prepared['args'].task
        expected=binding(root,prepared['config'],task)['live_control'];holder={'events':[]}
        ready=lambda:any(e['event']=='measured_grasp_frame_placement_compensation_admitted' for e in holder['events'])
        install_live(actor,phases=tuple(expected['phases']),replan_seconds=expected['replan_seconds'],ready=ready)
        install_terminal(actor,phases=tuple(expected['phases']),ready=ready)
        holder['events']=install_compensation(actor,prepared['config'],root,task)
        return actor,identity
