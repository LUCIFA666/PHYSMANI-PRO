import importlib.util
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('p4teacher', ROOT/'scripts/collect_p4_teacher_train_20261103.py')
teacher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(teacher)


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def state(k, *, held=False, success=False, lift=0.):
    return dict(time=k*.05, counted_tick=k, held=held, native_success=success,
        requested_opening=1. if k<5 else 0., tcp_position=[0.,0.,.8], tcp_quaternion=[0.,0.,0.,1.],
        object_position=[0.,0.,.75+lift], object_quaternion=[0.,0.,0.,1.])


def transitions(case, count):
    ticks=[dict(clock_before=k*.05,clock_after=(k+1)*.05,
        gripper_target=1. if k<5 else 0.,gripper_done=k>=8,tick_completed=True) for k in range(count)]
    (case/'transitions.jsonl').write_text(json.dumps(dict(executor=dict(ticks=ticks)))+'\n')


def capture(case,count=6):
    directory=case/'public_shadow';directory.mkdir(parents=True)
    write(directory/'scope.json',dict(controller_uses_public_forecast=False,actual_sensor_dt=.05,
        object_GT_held_success_allowed_in_public_bridge=False))
    write(directory/'camera_calibration.json',{})
    rows=[]
    for k in range(count):
        now=k*.05
        data=dict(timestamp=np.array(now))
        for view in teacher.CAMERAS:
            data['images_'+view]=np.zeros((2,2,3),dtype=np.uint8)
            data['depths_'+view]=np.ones((2,2),dtype=np.float32)
        relative=f'rgbd/{k:04d}.npz';path=directory/relative;path.parent.mkdir(exist_ok=True)
        with path.open('xb') as stream:np.savez(stream,**data)
        robot=dict(timestamp=now,tcp_position=[0,0,1],tcp_quaternion=[0,0,0,1],q=[0]*7,dq=[0]*7,finger_open_amounts=[1,1])
        result=dict(timestamp=now,Native_inputs_used=False,future_observations_used=False,
            observations_fabricated=False,free_forecast_available=k>=4,history_frames=k+1,
            frozen_checkpoint_sha256='frozen',forecast=dict(times=(now+np.arange(17)*.05).tolist()))
        rows.append(dict(result=result,public_robot=robot,rgbd=relative))
    (directory/'public_outputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (directory/'evaluation_labels.jsonl').write_text('{}\n'*count)
    return rows


def gate_rows(positives=2):
    return [dict(parent_id=f'p{k}',positive_contact_labels=int(k<positives),causal_macro_anchors=int(k<positives),
        original_Native_sources_passed=True,public_capture=dict(exactly_50ms=True)) for k in range(4)]


class TeacherCollectionTests(unittest.TestCase):
    def test_default_preflight_never_launches_child(self):
        with patch.object(teacher,'validate_plan',return_value=(dict(jobs=[{}]*16,source_sha256={'x':'y'}),b'',[])), \
             patch.object(teacher.subprocess,'Popen',side_effect=AssertionError('Native forbidden')):
            r=teacher.run('unused',None)
        self.assertFalse(r['Native_started'])
        self.assertTrue(r['execution_requires_explicit_execute'])

    def test_pilot_counts_independent_positive_parents_and_fidelity(self):
        self.assertTrue(teacher.pilot_gate(gate_rows())['passed'])
        self.assertFalse(teacher.pilot_gate(gate_rows(1))['passed'])
        rows=gate_rows();rows[1]['parent_id']=rows[0]['parent_id']
        self.assertFalse(teacher.pilot_gate(rows)['passed'])
        rows=gate_rows();rows[3]['public_capture']['exactly_50ms']=False
        self.assertFalse(teacher.pilot_gate(rows)['passed'])

    def test_source_verification_uses_archived_bytes_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            case=Path(tmp);path=case/'source_snapshot/our_mani/x.py';path.parent.mkdir(parents=True)
            path.write_bytes(b'frozen original')
            expected={'our_mani/x.py':hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(teacher.validate_archived_sources(case,expected)['archived_source_files_verified'],1)
            path.write_bytes(b'corruption')
            with self.assertRaises(RuntimeError):teacher.validate_archived_sources(case,expected)
            with self.assertRaises(ValueError):teacher.validate_archived_sources(case,{'../../outside':'x'})

    def test_real50ms_capture_and_forecast_metadata_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case=root/'case';rows=capture(case)
            with patch.object(teacher,'ROOT',root):
                result,forecasts=teacher.audit_public_capture(case,[dict(time=k*.05) for k in range(6)],'frozen')
                self.assertEqual(result['real_public_samples'],6)
                self.assertEqual(set(forecasts),{4,5})
                rows[2]['result']['timestamp']=.15
                (case/'public_shadow/public_outputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.assertRaises(ValueError):teacher.audit_public_capture(case,[dict(time=k*.05) for k in range(6)],'frozen')

    def test_GT_robot_channel_and_fake_future_flag_are_rejected(self):
        for mutation in ('held','future'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);case=root/'case';rows=capture(case)
                if mutation=='held':rows[0]['public_robot']['held']=True
                else:rows[0]['result']['future_observations_used']=True
                (case/'public_shadow/public_outputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with patch.object(teacher,'ROOT',root),self.assertRaises(ValueError):
                    teacher.audit_public_capture(case,[dict(time=k*.05) for k in range(6)],'frozen')

    def test_dispatch_held_and_API_done_remain_different(self):
        with tempfile.TemporaryDirectory() as tmp:
            case=Path(tmp);states=[state(k,held=k>=6,success=k==12,lift=.05 if k>=6 else 0.) for k in range(21)]
            transitions(case,len(states)-1)
            rows=teacher.contact_labels(case,states,{k:{} for k in range(4,6)},'parent')
            self.assertEqual(len(rows),1)
            row=rows[0];self.assertTrue(row['accepted_positive'])
            self.assertAlmostEqual(row['dispatch_tBefore'],.25)
            self.assertAlmostEqual(row['first_actual_Native_held_time'],.30)
            self.assertAlmostEqual(row['API_completion_call_time'],.40)
            self.assertAlmostEqual(row['API_completion_feedback_time'],.45)
            self.assertEqual(len(row['causal_50ms_anchors']),2)
            self.assertTrue(all(a['t0']<row['label']['tau'] and not a['features_emitted'] for a in row['causal_50ms_anchors']))

    def test_goal_and_hold_cannot_be_borrowed_across_held_segments(self):
        from our_mani.local_mpc_p4_20261103.data import successful_contacts
        with tempfile.TemporaryDirectory() as tmp:
            case=Path(tmp)
            states=[state(k,held=(6<=k<=10 or 12<=k<=26),success=k==8,
                lift=.05 if k>=6 else 0.) for k in range(28)]
            transitions(case,len(states)-1)
            labels=teacher.contact_labels(case,states,{k:{} for k in range(4,6)},'parent')
            self.assertEqual(len(labels),2)
            self.assertFalse(any(r['accepted_positive'] for r in labels))
            self.assertEqual(successful_contacts(states[0],states[1:]),[])
            states[22]['native_success']=True
            labels=teacher.contact_labels(case,states,{k:{} for k in range(4,6)},'parent')
            ours=[r['label']['tau'] for r in labels if r['accepted_positive']]
            canonical=[r['tau'] for r in successful_contacts(states[0],states[1:])]
            self.assertEqual(ours,canonical)


if __name__=='__main__':unittest.main()
