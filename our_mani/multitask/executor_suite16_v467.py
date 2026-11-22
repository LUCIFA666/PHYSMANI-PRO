"""Explicit 16-task, three-method routing for the current executor candidates.

Routing does not grant qualification. Every route names an immutable existing
setup, worker and consumer; the caller must separately audit its TRAIN cases.
The two original block baselines retain their V6 path and the other fourteen
retain their terminal-aware V7 path.
"""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import json
from .evaluation import file_sha
from .evaluation_roster16_v283 import TASKS,METHODS
from .joint16_tracks_v31 import BLOCKS

CHECKPOINT_SHA='6447ac755032b022d18884936e02a90d8ec2f2b219b5d42fc140cdd1618cd017'


@dataclass(frozen=True)
class ExecutorRoute:
    setup: str
    worker: str
    consumer_module: str
    consumer_class: str
    task_named_configuration: bool=False
    backend_takes_task: bool=True

    def configuration(self,task):
        return self.setup+'/'+(task+'_configuration.json' if self.task_named_configuration else 'configuration.json')


OURS=MappingProxyType({
    **{t:ExecutorRoute('block_next_sample_v459_setup','run_task_executor_smoke_block_next_sample_v458_r10',
        'joint16_block_next_sample_episode_v457','run_TRAIN_development_episode') for t in BLOCKS},
    **{t:ExecutorRoute('buzz_physical_v418_setup','run_task_executor_smoke_buzz_physical_r10',
        'buzz_physical_executor_native','BuzzPhysicalBackend',True) for t in TASKS if t.startswith('beat_the_rotating_buzz')},
    'insert_onto_rotating_peg':ExecutorRoute('terminal_live_grasp_v430_setup',
        'run_task_executor_smoke_terminal_live_grasp_r10','terminal_live_grasp_executor','TerminalLiveGraspBackend'),
    'insert_onto_rotating_peg_high_speed':ExecutorRoute('high_peg_reserve_v425_setup',
        'run_task_executor_smoke_high_peg_reserve_r10','high_peg_reserve_executor_native','HighPegReserveBackend',True),
    **{t:ExecutorRoute('basketball_executor_v421_setup','run_task_executor_smoke_basketball_r10',
        'basketball_executor_native','BasketballBackend',True) for t in TASKS if t.startswith('moving_basketball_in_hoop')},
    **{t:ExecutorRoute('button_corridor_v414_setup','run_task_executor_smoke_button_corridor_r10',
        'button_corridor_executor_native','ButtonCorridorBackend',False,False) for t in TASKS if t.startswith('push_moving_button')},
    **{t:ExecutorRoute('cup_next_sample_v462_setup','run_task_executor_smoke_cup_next_sample_v461_r10',
        'cup_next_sample_executor_v460','CupNextSampleBackend',True) for t in TASKS if t.startswith('place_cups_on_rotating_frame')},
    **{t:ExecutorRoute('remove_uncertain_translation_v445_setup','run_task_executor_smoke_remove_uncertain_translation_r10',
        'remove_uncertain_translation_executor','RemoveUncertainTranslationBackend',True) for t in TASKS if t.startswith('remove_cups_from_rotating_frame')},
    'put_rubbish_in_moving_bin':ExecutorRoute('bin_physical_smoke_v240_setup',
        'run_task_executor_smoke_bin_physical_v239_r10','bin_physical_native_v235','SharedJoint16NativeBackend',False,False),
    'put_rubbish_in_moving_bin_high_speed':ExecutorRoute('bin_capture_v429_setup',
        'run_task_executor_smoke_bin_capture_r10','bin_capture_executor_native','BinCaptureBackend'),
})


def route(task,method):
    if task not in TASKS or method not in METHODS:raise ValueError('exactly sixteen tasks and three authorized methods')
    if method!='physmani':return OURS[task]
    if task in BLOCKS:
        return ExecutorRoute('block_physmani_v348_setup','run_task_executor_smoke_block_physmani_v347_r10',
            'block_physmani_v346','run_TRAIN_baseline',False,False)
    return ExecutorRoute('physmani_windows_v194_setup','run_task_executor_smoke_physmani_v192_r10',
        'native_window_observer_v191','BaselineWindowBackend',False,False)


def snapshot(root):
    root=Path(root);base=root/'artifacts/framework/multitask_r10';rows=[]
    if set(OURS)!=set(TASKS):raise ValueError('missing full sixteen-task policy routes')
    for task in TASKS:
        for method in METHODS:
            r=route(task,method);config=base/r.configuration(task)
            lock=base/r.setup/'frozen_inputs.json'
            c=json.loads(config.read_text())
            if method!='physmani' and c['context_checkpoint']['sha256']!=CHECKPOINT_SHA:
                raise ValueError('all sixteen policies must share the same joint16 dynamics')
            consumer=root/'our_mani/multitask'/(r.consumer_module+'.py')
            worker=root/'scripts'/(r.worker+'.py')
            rows.append(dict(task=task,method=method,component='aligned_block' if task in BLOCKS else 'r10_multitask',
                configuration=dict(path=str(config.relative_to(root)),sha256=file_sha(config)),
                frozen_inputs=dict(path=str(lock.relative_to(root)),sha256=file_sha(lock)),
                worker=dict(path=str(worker.relative_to(root)),sha256=file_sha(worker)),
                consumer=dict(path=str(consumer.relative_to(root)),sha256=file_sha(consumer),
                    symbol=r.consumer_class,takes_task=r.backend_takes_task),GPU_required=method=='physmani'))
    return dict(schema='r10_current16_three_method_executor_candidate_routes_v1',routes=rows,
        methods=list(METHODS),tasks=list(TASKS),shared_context_checkpoint_sha256=CHECKPOINT_SHA,
        route_count=48,qualification_granted=False,formal_VAL_TEST_started=False,TEST_read=False,
        original_baseline_algorithms_unchanged=True,source_sha256=file_sha(__file__))
