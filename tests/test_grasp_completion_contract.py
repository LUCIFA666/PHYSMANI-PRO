from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace as NS
import json
import unittest
from our_mani.multitask.interaction import InteractionLibrary,GripperTimingConfig
from our_mani.multitask.grasp_completion_contract import binding,install,TASKS
ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def test_actual_two_buzz_libraries_tighten_close_to_the_same_acquire_completion(self):
        for task in TASKS:
            config=json.loads((ROOT/'artifacts/framework/multitask_r10/buzz_command_v412_setup'/
                (task+'_configuration.json')).read_text())
            library=InteractionLibrary.load(ROOT/config['learned_artifacts'][task]['interaction']['path'])
            c=NS(goal=None,feedback_step_calls=0,library=library,
                gripper_timing_config=GripperTimingConfig(**config['gripper_timing_config']))
            actor=NS(controller=c,continuous_reference_manager=NS(confirmations=[]),metrics_report=lambda:{})
            config['grasp_completion_contract']={task:binding(ROOT,config,task)}
            actual=install(actor,config,ROOT,task)
            self.assertEqual(actual.close_position_tolerance_m,.005)
            self.assertEqual(actual.close_orientation_tolerance_rad,.03)
            self.assertEqual(asdict(c.gripper_timing_config),config['grasp_completion_contract'][task]['effective_close_gate'])
            self.assertIs(c.library,library)
            self.assertLess(actual.close_orientation_tolerance_rad,.033831992957180944)
            self.assertTrue(actor.metrics_report()['grasp_completion_contract']['public_contact_still_required'])

    def test_actual_runtime_gate_drift_is_rejected(self):
        task=TASKS[0];config=json.loads((ROOT/'artifacts/framework/multitask_r10/buzz_command_v412_setup'/
            (task+'_configuration.json')).read_text())
        config['grasp_completion_contract']={task:binding(ROOT,config,task)}
        library=InteractionLibrary.load(ROOT/config['learned_artifacts'][task]['interaction']['path'])
        c=NS(goal=None,feedback_step_calls=0,library=library,
            gripper_timing_config=GripperTimingConfig(.03,.2))
        actor=NS(controller=c,continuous_reference_manager=NS(confirmations=[]),metrics_report=lambda:{})
        with self.assertRaisesRegex(ValueError,'preexisting'):install(actor,config,ROOT,task)


if __name__=='__main__':unittest.main()
