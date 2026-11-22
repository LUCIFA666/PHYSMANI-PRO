"""Button executor with moving-target contact corridor and consistent brakes."""
from pathlib import Path
from types import MethodType
import json
from .button_reference_native_v387 import ButtonIntentNativeBackend
from .receding_task_transit import install as install_transit
from .moving_reference import install_moving_reference
from .reference_constraint_authority import install as install_authority
from .button_contact_corridor import install as install_corridor
from .executor_trace import install_trace
from .evaluation import file_sha
from .frozen_subset_batch_v132 import checked_subset
TASKS=('push_moving_button','push_moving_button_high_speed')
SOURCES=('our_mani/multitask/button_corridor_executor_native.py',
    'our_mani/multitask/button_reference_native_v387.py',
    'our_mani/multitask/receding_task_transit.py','our_mani/multitask/continuous_transit.py',
    'our_mani/multitask/moving_reference.py','our_mani/multitask/reference_constraint_authority.py',
    'our_mani/multitask/button_contact_corridor.py','our_mani/multitask/executor_trace.py')


def binding(root,config,task):
    if task not in TASKS:raise ValueError('explicit original button task')
    return dict(schema='r10_button_complete_contact_corridor_executor_v1',task=task,
        replan_seconds=.2,geometry=config['learned_artifacts'][task]['geometry'],
        interaction=config['learned_artifacts'][task]['interaction'],
        sources_sha256={p:file_sha(Path(root)/p) for p in SOURCES},
        reference_acceleration_limit=4.,actual_acceleration_limit=6.,
        native_success_and_global10s_unchanged=True,formal_VAL_TEST=False)


def verify(root,prepared):
    expected=binding(root,prepared['config'],prepared['args'].task)
    if prepared['config'].get('button_corridor_executor',{}).get(prepared['args'].task)!=expected:
        raise ValueError('freeze full contact corridor policy')
    checked_subset(root,prepared['files'],{str(Path(root)/p):sha for p,sha in expected['sources_sha256'].items()})
    return expected


class ButtonCorridorBackend(ButtonIntentNativeBackend):
    def actor(self,prepared,env,descriptions):
        actor,identity=super().actor(prepared,env,descriptions)
        root=Path(__file__).resolve().parents[2];expected=binding(root,prepared['config'],prepared['args'].task)
        install_transit(actor,replan_seconds=expected['replan_seconds'])
        install_moving_reference(actor,phases=('press',),replan_seconds=expected['replan_seconds'])
        install_authority(actor)
        path=root/expected['geometry']['path']
        if file_sha(path)!=expected['geometry']['sha256']:raise ValueError('TRAIN plate geometry changed')
        install_corridor(actor,json.loads(path.read_text()),replan_seconds=expected['replan_seconds'])
        install_trace(actor)
        previous=actor.metrics_report
        def metrics(instance):return dict(previous(),button_corridor_executor=expected)
        actor.metrics_report=MethodType(metrics,actor)
        return actor,identity
