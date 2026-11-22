"""Prospective archival repair only; never modify original payload or case."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from our_mani.local_mpc_20261102 import records

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'config/framework/P4_student_closedloop_devVAL_v56_v1/00_val_static.json'
RUN = ROOT/'runtime/framework/local_mpc_20261102/P4_student_closedloop_devVAL_v56_v1/00_val_static/run_inputs.json'
PAYLOADS = {
    'artifacts/framework/local_mpc_20261102/P2_servo_training_v2/best.pt':
        '25a0e0cbd49e0e0c9e8a4671649f96a1bb565013011d728e88b2c2f3eaca0073',
    'runtime/framework/local_mpc_20261102/p2_fk_canary_v1/result.json':
        '2a3d301dbd0926760b74ba6b68fc1de85f448c402c215d0f6877e7059278600a',
    'artifacts/framework/local_mpc_20261102/P2_data_v2/manifest.json':
        'bf7187fcb47b7e1392d762915bc3a577b0d472be124c8a6ecc7dbe65bec2af1d',
    'artifacts/framework/local_mpc_20261102/P2_servo_training_v2/source_archive.json':
        '6e004e03e3c3829a4504b5fb43b3200839721e3c190000ce7ba00f9a7bcdf4ff'}


class ProspectivePayloadArchiveTests(unittest.TestCase):
    def test_same_closed_VAL_identity_adds_exact_four_without_rewriting_old_map(self):
        old_bytes = RUN.read_bytes(); old = json.loads(old_bytes)['source_sha256']
        new = records.freeze_sources(CONFIG, 'p1')
        self.assertEqual(len(old), 120); self.assertEqual(len(new), 124)
        self.assertEqual(set(new)-set(old), set(PAYLOADS))
        self.assertEqual(set(old)-set(new), set())
        # Historical120 stays immutable. The optional compute-only modifier
        # subsequently changed these four sources, with False on this config.
        self.assertEqual({name for name in old if old[name] != new[name]}, {
            'our_mani/local_mpc_20261102/records.py',
            'our_mani/local_mpc_20261102/worker.py',
            'our_mani/local_mpc_control_20261103/public_grasp_candidates_v5.py',
            'our_mani/local_mpc_control_20261103/public_candidate_goal_maintenance_v56.py'})
        for name, expected in PAYLOADS.items():
            self.assertEqual(new[name], expected)
            self.assertEqual(records.sha(ROOT/name), expected)
        self.assertEqual(RUN.read_bytes(), old_bytes)

    def test_each_hard_pin_fails_closed_on_wrong_hash_without_touching_payloads(self):
        actual_sha = records.sha
        for name, expected in PAYLOADS.items():
            with self.subTest(payload=name):
                target = (ROOT/name).resolve()
                def digest(path):
                    return '0'*64 if Path(path).resolve() == target else actual_sha(path)
                with patch.object(records, 'sha', side_effect=digest):
                    with self.assertRaisesRegex(ValueError, 'factory runtime/training evidence payload changed') as failure:
                        records.freeze_sources(CONFIG, 'p1')
                self.assertIn(name, str(failure.exception))
                self.assertEqual(actual_sha(target), expected)

    def test_four_exact_archived_payloads_and_corrupted_temp_archive_rejected(self):
        with TemporaryDirectory(prefix='v57_payload_archive_CPU_') as directory:
            records.archive_sources(directory, PAYLOADS)
            for name, expected in PAYLOADS.items():
                archived = Path(directory)/'source_snapshot'/name
                self.assertEqual(records.sha(archived), expected)
                self.assertEqual(archived.read_bytes(), (ROOT/name).read_bytes())
        # Corrupt only a disposable output stream after copying. The real model
        # bytes and historical120 source inventories remain completely intact.
        original_copy = records.shutil.copyfileobj
        def corrupt_output(source, output, *args, **kwargs):
            original_copy(source, output, *args, **kwargs)
            output.write(b'explicit_CPU_temp_archive_corruption')
        one = next(iter(PAYLOADS))
        with TemporaryDirectory(prefix='v57_corrupt_archive_CPU_') as directory, \
             patch.object(records.shutil, 'copyfileobj', side_effect=corrupt_output):
            with self.assertRaisesRegex(RuntimeError, 'source raced during archival'):
                records.archive_sources(directory, {one:PAYLOADS[one]})
        for name, expected in PAYLOADS.items():
            self.assertEqual(records.sha(ROOT/name), expected)


if __name__ == '__main__':
    unittest.main()
