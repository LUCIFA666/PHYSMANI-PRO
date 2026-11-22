"""Check the common60second physical budget on the real all16-task backend."""
from pathlib import Path
import json
from . import TaskBudget
from .feedback_backend import ExtendedTimeFeedbackBackend


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)


def run_case(spec,output,stop_requested):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    budget=TaskBudget.from_config(spec['config']);backend=None;closed=False;error=None
    try:
        backend=ExtendedTimeFeedbackBackend(spec['task'],spec['calibration_episode'],seed=spec['seed'],variation=0,
            budget_ticks=budget.ticks,execute_native=True,dataset_use='development')
        home=backend.get_joint_positions();ticks=0
        while backend.remaining_ticks()>0 and not backend.terminal():
            if stop_requested():raise InterruptedError('Time-budget validation STOP')
            backend.set_joint_target_positions(home)
            backend.tick('all16_time_budget_robot_initial_posture_hold')
            ticks+=1
            if ticks%200==0:print(f'All16 budget check: ticks={ticks}, Native_time={backend.clock():.3f}',flush=True)
        actual=backend.clock()
        if ticks!=budget.ticks or abs(actual-budget.seconds)>2e-5:
            raise AssertionError('Expected complete60s hold; inspect Native terminal outcome')
        result=dict(complete=True,task=spec['task'],seed=spec['seed'],actual_ticks=ticks,actual_physics_seconds=actual,
            task_budget=budget.receipt(),original200_tick_limit_exceeded=ticks>200,remaining_ticks=backend.remaining_ticks(),
            original_native_metric=backend.metric_receipt(),task_success_claimed=False,learned_task_policy_evaluated=False,
            physics_or_native_success_modified=False,formal_VAL_TEST=False,Native_contact_or_object_pose_used_by_policy=False)
        write(output/'result.json',result)
        return result
    except BaseException as exception:error=type(exception).__name__+': '+str(exception);raise
    finally:
        if backend is not None:backend.close();closed=backend._closed
        write(output/'worker_exit.json',dict(native_closed=closed,error=error))
