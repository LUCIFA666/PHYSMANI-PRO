"""STOP, PID ownership and complete-grid stage gates without any simulator."""
import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from scripts import continue_resumed_VAL_TEST_v775_r10 as controller


class ResumeControllerTests(unittest.TestCase):
    def test_STOP_prevents_any_new_child(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(controller,'ROOT',Path(tmp)),patch.object(controller,'BASE',Path(tmp)/'controller'):
            p=Path(tmp)/'runtime/framework/multitask_r10_STOP';p.parent.mkdir(parents=True);p.touch()
            with patch.object(controller.subprocess,'Popen') as popen:
                with self.assertRaises(InterruptedError):controller.launch('no_launch','script.py')
                popen.assert_not_called()

    def test_PID_reuse_cannot_count_as_previous_owner_alive(self):
        pid=os.getpid();ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
        self.assertTrue(controller.alive({'pid':pid,'start_ticks':ticks}))
        self.assertFalse(controller.alive({'pid':pid,'start_ticks':ticks+1}))

    def test_new_infrastructure_failure_requests_DRAIN_without_signalling_child(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(controller,'ROOT',Path(tmp)):
            runtime=Path(tmp)/'native';runtime.mkdir();leaf=Path(tmp)/'stage';leaf.mkdir()
            (runtime/'status.json').write_text(json.dumps({'failed':{'bad':{'complete':False}}}))
            child=Mock();child.poll.side_effect=[None,0];child.wait.return_value=0
            with (patch.object(controller,'launch',return_value=(child,leaf)),patch.object(controller,'guard'),
                  patch.object(controller,'pulse'),patch.object(controller.time,'sleep'),
                  patch.object(controller,'interrupt_owned_tree') as kill):
                with self.assertRaises(RuntimeError):controller.run('stage','script',formal_runtime='native')
                self.assertTrue((runtime/'DRAIN').exists());kill.assert_not_called()

    def test_exit_zero_does_not_open_next_stage_without_complete_grid(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(controller,'ROOT',Path(tmp)):
            runtime=Path(tmp)/'native';runtime.mkdir();leaf=Path(tmp)/'stage';leaf.mkdir()
            (runtime/'terminal.json').write_text(json.dumps({'complete':False,'completed':95,'expected':945}))
            child=Mock();child.poll.return_value=0;child.wait.return_value=0
            with patch.object(controller,'launch',return_value=(child,leaf)):
                with self.assertRaisesRegex(RuntimeError,'complete full grid'):
                    controller.run('stage','script',formal_runtime='native')


if __name__=='__main__':unittest.main()
