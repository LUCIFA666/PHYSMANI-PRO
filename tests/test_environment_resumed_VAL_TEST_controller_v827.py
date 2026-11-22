"""CPU mocks: STOP, namespace inheritance, startup race, and noninterrupting DRAIN."""
from pathlib import Path
from unittest.mock import Mock,patch
import json,os,tempfile,unittest
from scripts import continue_environment_resumed_VAL_TEST_v827_r10 as ctl


class ControllerTests(unittest.TestCase):
    def test_STOP_has_priority_over_missing_namespace_and_any_new_launch(self):
        for name in ('runtime/framework/multitask_r10_STOP','runtime/framework/ordered_resume_VAL_TEST_v775/STOP',
                     ctl.VAL_RUNTIME+'/STOP',ctl.TEST_RUNTIME+'/STOP'):
            with tempfile.TemporaryDirectory() as tmp,patch.object(ctl,'ROOT',Path(tmp)),patch.object(ctl,'BASE',Path(tmp)/'new'):
                path=Path(tmp)/name;path.parent.mkdir(parents=True);path.touch()
                with patch.object(ctl.subprocess,'Popen') as popen:
                    with self.assertRaises(InterruptedError):ctl.launch('fake','fake.py')
                    popen.assert_not_called()

    def test_no_namespace_capture_means_no_child(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(ctl,'ROOT',Path(tmp)),patch.object(ctl,'BASE',Path(tmp)/'new'),patch.object(ctl,'ENVIRONMENT_RECEIPT',None):
            with patch.object(ctl.subprocess,'Popen') as popen:
                with self.assertRaisesRegex(ValueError,'namespace'):ctl.launch('fake','fake.py')
                popen.assert_not_called()

    def test_changed_mount_namespace_cannot_inherit_a_stale_receipt(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(ctl,'ROOT',Path(tmp)),patch.object(ctl,'BASE',Path(tmp)/'new'),patch.object(ctl,'ENVIRONMENT_RECEIPT',dict(current_mount_namespace='not-current')):
            with patch.object(ctl.subprocess,'Popen') as popen:
                with self.assertRaises(ValueError):ctl.launch('fake','fake.py')
                popen.assert_not_called()

    def test_infrastructure_failure_requests_DRAIN_without_signalling_active_case(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(ctl,'ROOT',Path(tmp)):
            runtime=Path(tmp)/'native';runtime.mkdir();leaf=Path(tmp)/'leaf';leaf.mkdir()
            (runtime/'status.json').write_text(json.dumps(dict(failed={'bad':dict(complete=False)})))
            child=Mock();child.poll.side_effect=[None,0];child.wait.return_value=0
            with (patch.object(ctl,'launch',return_value=(child,leaf)),patch.object(ctl,'guard'),
                  patch.object(ctl,'pulse'),patch.object(ctl.time,'sleep'),patch.object(ctl,'interrupt_owned_tree') as kill):
                with self.assertRaises(RuntimeError):ctl.run('fake','fake.py',formal_runtime='native')
                self.assertTrue((runtime/'DRAIN').exists());kill.assert_not_called()

    def test_owner_exit_zero_without_full945_terminal_cannot_release_TEST(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(ctl,'ROOT',Path(tmp)):
            runtime=Path(tmp)/'native';runtime.mkdir();leaf=Path(tmp)/'leaf';leaf.mkdir()
            (runtime/'terminal.json').write_text(json.dumps(dict(complete=False,completed=591,expected=945)))
            child=Mock();child.poll.return_value=0;child.wait.return_value=0
            with patch.object(ctl,'launch',return_value=(child,leaf)):
                with self.assertRaisesRegex(RuntimeError,'complete full grid'):ctl.run('fake','fake.py',formal_runtime='native')

    def test_wait_for_initial_status_does_not_read_a_missing_file_or_start_another_owner(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(ctl,'ROOT',Path(tmp)):
            child=Mock();child.poll.return_value=None;child.pid=123
            def ready(*args,**kwargs):
                path=Path(tmp)/'native/status.json';path.parent.mkdir();path.write_text('{}')
            with patch.object(ctl,'guard'),patch.object(ctl,'pulse',side_effect=ready),patch.object(ctl.time,'sleep'),patch.object(ctl.subprocess,'Popen') as popen:
                ctl.await_initial_status(child,'native');popen.assert_not_called()

    def test_owner_exit_before_status_is_a_concrete_failure(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(ctl,'ROOT',Path(tmp)),patch.object(ctl,'guard'):
            child=Mock();child.poll.return_value=1
            with self.assertRaisesRegex(RuntimeError,'before first status'):ctl.await_initial_status(child,'native')

    def test_PID_reuse_is_not_a_live_previous_owner(self):
        pid=os.getpid();start=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
        self.assertTrue(ctl.alive(dict(pid=pid,start_ticks=start)));self.assertFalse(ctl.alive(dict(pid=pid,start_ticks=start+1)))


if __name__=='__main__':unittest.main()
