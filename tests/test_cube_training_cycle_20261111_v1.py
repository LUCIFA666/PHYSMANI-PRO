import json
import tempfile
import unittest
from pathlib import Path
from our_mani.cube_training_cycle_20261111_v1.native_worker import TrainingLabelRecorder

class Backend:
    def __init__(self): self.time=0.
    def clock(self): return self.time
    def read_state(self):
        return dict(time=self.time,physics_time=1+self.time,task_time=2+self.time,
                    object_position=[.2,0,.775],tcp_position=[.1,0,1],tcp_quaternion=[0,0,0,1],
                    held=False,native_success=False,requested_opening=1.,object_native_future='forbidden')
    def tick(self,phase): self.time+=.05;return {'public_proprio':'unchanged'}

class TrainingSinkContracts(unittest.TestCase):
    def test_private_sink_records_initial_and_actual_ticks_only(self):
        with tempfile.TemporaryDirectory() as d:
            raw=Backend();p=Path(d)/'private.jsonl';sink=TrainingLabelRecorder(raw,p,'fit0')
            self.assertEqual(sink.tick('test'),{'public_proprio':'unchanged'})
            rows=[json.loads(x)for x in p.read_text().splitlines()]
            self.assertEqual([r['time']for r in rows],[0.,.05])
            self.assertEqual(set(rows[0]),set(TrainingLabelRecorder.KEYS)|{'episode_id'})
            self.assertEqual(raw.clock(),.05)
            with self.assertRaisesRegex(RuntimeError,'one row'): sink.record()
    def test_private_sink_rejects_missing_physical_tick(self):
        with tempfile.TemporaryDirectory() as d:
            raw=Backend();sink=TrainingLabelRecorder(raw,Path(d)/'private.jsonl','fit0')
            raw.time=.1
            with self.assertRaisesRegex(RuntimeError,'one row'): sink.record()

if __name__=='__main__':unittest.main()
