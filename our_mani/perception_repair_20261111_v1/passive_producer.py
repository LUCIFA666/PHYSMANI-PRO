"""Public-camera diagnostic control: keep the initial robot posture only."""
import numpy as np
from ..execution_interface_20261111_v1.commands import JointGoal
from ..local_mpc_20261102.records import append

def run(backend,executor,output,config,*,stop_requested=lambda:False):
    from pathlib import Path
    output=Path(output);home=np.asarray(backend.get_joint_positions());ticks=0
    while ticks<int(config['diagnostic_capture_cycles']) and not backend.terminal():
        if stop_requested():raise InterruptedError('Passive capture STOP')
        receipt=executor.execute_command(JointGoal(home,opening=1.,ticks=1,expected_time=backend.clock()),phase='passive_initial_robot_posture')
        append(output/'passive_commands.jsonl',receipt);ticks+=receipt.executed_ticks
    return dict(complete=True,episode_success=backend.native_task_success_latched,actual_ticks=ticks,
        task_evaluation_complete=False,stop_reason='passive_diagnostic_window_completed',
        native_task_judge=backend.native_task_judge_receipt(),robot_policy_used=False,Native_target_pose_used_by_policy=False)
