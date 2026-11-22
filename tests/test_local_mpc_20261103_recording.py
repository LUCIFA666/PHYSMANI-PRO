import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from our_mani.local_mpc_20261102.recording import FrameRecorder
from our_mani.local_mpc_20261102 import records


class RecordingTests(unittest.TestCase):
    def test_background_frames_copy_inputs_and_drain_before_success(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = FrameRecorder(max_pending=2)
            value = np.arange(20)
            for index in range(6):
                recorder.submit(Path(directory) / f'{index}.npz', {'rgb': value})
            value[:] = -1
            recorder.close()
            for path in Path(directory).glob('*.npz'):
                np.testing.assert_array_equal(np.load(path)['rgb'], np.arange(20))
            self.assertEqual(len(list(Path(directory).glob('*.npz'))), 6)

    def test_writer_failure_is_reported_and_second_close_is_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'frame.npz'
            path.write_bytes(b'prior evidence')
            recorder = FrameRecorder()
            recorder.submit(path, {'x': np.zeros(1)})
            with self.assertRaises(FileExistsError):
                recorder.close()
            recorder.close()
            self.assertEqual(path.read_bytes(), b'prior evidence')

    def test_archive_retains_exact_source_and_reports_later_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'source.py').write_text('value = 1\n')
            frozen = {'source.py': records.sha(root/'source.py')}
            with patch.object(records, 'ROOT', root):
                records.archive_sources(root/'case', frozen)
                self.assertEqual((root/'case/source_snapshot/source.py').read_text(), 'value = 1\n')
                (root/'source.py').write_text('value = 2\n')
                with self.assertRaises(RuntimeError):
                    records.verify_sources(frozen)

    def test_hold_phase_numpy_fields_and_infeasible_cost_are_json_serializable(self):
        value = records.plain({'phase': {'lift_goal_position': np.zeros(3)}, 'cost': np.float64(np.inf)})
        self.assertIn('INFEASIBLE', json.dumps(value, allow_nan=False))

