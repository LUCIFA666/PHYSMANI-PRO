"""Synthetic hardware/dispatch contracts; no real-data fit, Native or TEST."""
from copy import deepcopy
import importlib.util,io,json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

from our_mani.local_mpc_20261102.executor import JointLimits
from our_mani.local_mpc_p2_20261103.reference import reference_step
from our_mani.local_mpc_onpolicy_response_20261103 import data,ridge,evaluation


def fixture():
    limits=JointLimits([-3.]*7,[3.]*7,[2.]*7)
    q=np.zeros(7);dq=np.zeros(7);goal=np.full(7,.02)
    h=dict(goal_q=q[None].copy(),q_ref=q[None].copy(),dq_ref=dq[None].copy(),ddq_ref=q[None].copy(),
        valid=np.array([False]),initialized=np.array([False]),gripper_target=np.array([0.]))
    def robot(t,q,dq):return dict(timestamp=t,q=q.copy(),dq=dq.copy(),
        tcp_position=np.r_[q[:2],.8+q[2]],tcp_quaternion=[0.,0.,0.,1.],finger_open_amounts=[.3,.9])
    frames=[dict(public_robot=robot(0.,q,dq),result={'unread_private_object':object()})];controls=[]
    for macro in range(2):
        old=deepcopy(h);now=macro*.1
        if old['valid'][0]:
            actual_h=dict(reference_valid=True,reference_positions=old['q_ref'][0],reference_velocities=old['dq_ref'][0],
                reference_accelerations=old['ddq_ref'][0],target_joints=old['goal_q'][0],reference_clock=now,
                gripper_target=0.,reference_is_measured_state=False)
        else:actual_h=dict(reference_valid=False,reference_positions=None,reference_velocities=None,
            reference_accelerations=None,target_joints=None,reference_clock=None,gripper_target=0.,reference_is_measured_state=False)
        ticks=[]
        for micro in range(2):
            before=(macro*2+micro)*.05;ref=reference_step(q[None],dq[None],goal[None],h,limits)
            nq=ref['q_ref'][0].copy();ndq=(nq-q)/.05
            ticks.append(dict(clock_before=before,clock_after=before+.05,tick_completed=True,
                measured_positions_before=q.copy(),measured_velocities_before=dq.copy(),
                measured_positions_after=nq.copy(),measured_velocities_after=ndq.copy(),
                reference_positions=ref['q_ref'][0].copy(),reference_velocities=ref['dq_ref'][0].copy(),
                reference_accelerations=ref['ddq_ref'][0].copy(),gripper_target=0.))
            q,dq=nq,ndq;h=ref['next_h']
            frames.append(dict(public_robot=robot((macro*2+micro+1)*.05,q,dq),result={'Native_held':object()}))
        controls.append(dict(time=now,public_robot=deepcopy(frames[macro*2]['public_robot']),
            actuator_state_before=actual_h,decision=dict(mode=data.PROBE if macro else 'FREE_GRU'),
            executor=dict(ticks=ticks,executed_ticks=2,unexpected_physics_ticks=0,target_joints=goal.copy(),limits=limits.receipt())))
    return controls,frames


def episodes():
    e=data.extract_transitions(*fixture());out=[]
    for i in range(8):out.append(dict(deepcopy(e),episode_id=f'fake{i}',parent_id=f'parent{i}',role='fit' if i<6 else 'internal_val'))
    protocol=dict(fit_parent_ids=[f'parent{i}' for i in range(6)],internal_val_parent_ids=['parent6','parent7'],ridge=.001,native_dt=.05)
    return out,protocol


class OnPolicyResponseTests(unittest.TestCase):
    def test_actual_ref_labels_midcycle_h_and_oracle_reconstruction(self):
        controls,frames=fixture();e=data.extract_transitions(controls,frames)
        self.assertEqual(e['q'].shape,(5,7));self.assertEqual(e['refs'].shape,(4,21))
        np.testing.assert_array_equal(e['probe_mask'],[False,False,True,True])
        np.testing.assert_array_equal(e['h']['valid'],[False,True,True,True])
        np.testing.assert_array_equal(e['h']['q_ref'][1],e['refs'][0,:7])
        self.assertEqual(data.oracle_reference_audit(e)['maximum_absolute'],0.)
        self.assertFalse(e['audit']['Native_object_held_success_accessed'])
        self.assertFalse(e['audit']['reconstructed_reference_labels'])

    def test_missing_duplicate_stale_or_unfinished_tick_is_not_imputed(self):
        for fault in ('missing_frame','duplicate_frame','timestamp','dq_mismatch','unfinished','extra_tick'):
            with self.subTest(fault=fault):
                controls,frames=fixture()
                if fault=='missing_frame':frames.pop(2)
                elif fault=='duplicate_frame':frames.append(deepcopy(frames[-1]))
                elif fault=='timestamp':controls[0]['executor']['ticks'][0]['clock_after']=.1
                elif fault=='dq_mismatch':controls[1]['public_robot']['dq'][0]+=.01
                elif fault=='unfinished':controls[0]['executor']['ticks'][0]['tick_completed']=False
                else:controls[0]['executor']['unexpected_physics_ticks']=1
                with self.assertRaises((ValueError,KeyError)):data.extract_transitions(controls,frames)

    def test_oracle_detects_reference_corruption_not_a_model_error(self):
        e=data.extract_transitions(*fixture());e['refs'][1,4]+=.001
        with self.assertRaisesRegex(ValueError,'reconstruction mismatch'):data.oracle_reference_audit(e)

    def test_equal_parent_phase_mass_and_fit_scaler_ignore_internal_arrays(self):
        values,protocol=episodes();values[0]['probe_mask'][:]=False
        fit,w,ledger=ridge.balanced_rows(values,protocol['fit_parent_ids'])
        self.assertAlmostEqual(w.sum(),1.)
        for row in ledger:
            self.assertAlmostEqual(row['total_weight'],1/6)
            if row['parent_id']!='parent0':
                self.assertAlmostEqual(row['probe_weight'],1/12);self.assertAlmostEqual(row['nonprobe_weight'],1/12)
        a=ridge.fit_parameters(values,protocol,source_identity={'synthetic_only':True})
        for e in values:
            if e['role']=='internal_val':e['q'][:]=1e100;e['refs'][:]=1e200
        b=ridge.fit_parameters(values,protocol,source_identity={'synthetic_only':True})
        self.assertEqual(a['coefficient_scaled'],b['coefficient_scaled']);self.assertEqual(a['normalization'],b['normalization'])
        model=ridge.OnPolicyCoupledResponse(a);e=fit[0];r=e['refs']
        out=model.predict_joint(e['q'][:-1],e['dq'][:-1],r[:,:7],r[:,7:14],r[:,14:])
        np.testing.assert_allclose(out['dq'],(out['q']-e['q'][:-1])/.05,rtol=0,atol=1e-12)

    def test_evaluation_uses_predicted_future_q_and_reports_short_tail_coverage(self):
        values,_=episodes();e=values[0]
        class ExactReference:
            def predict_joint(self,q,dq,qref,vref,aref,dt=.05):return dict(q=qref.copy(),dq=(qref-q)/dt)
        def fk(q):
            matrices=np.broadcast_to(np.eye(4),q.shape[:-1]+(4,4)).copy()
            matrices[...,:3,3]=q[...,:3]+[0.,0.,.8];return matrices
        report=evaluation.evaluate(ExactReference(),[e],fk)
        role=report['roles']['fit']
        self.assertEqual(role['conditional_H1']['all']['joint_model_FK_position_l2_m']['maximum'],0.)
        for mode in ('recorded_reference','regenerated_reference'):
            h=role['rollouts'][mode]['by_horizon']
            self.assertEqual(h['1']['cohorts']['all']['labelled'],4)
            self.assertEqual(h['4']['cohorts']['all']['labelled'],1)
            self.assertEqual(h['5']['cohorts']['all']['labelled'],0)
            self.assertEqual(h['4']['cohorts']['all']['predictable'],1)
            self.assertEqual(h['4']['cohorts']['all']['errors']['joint_model_FK_position_l2_m']['maximum'],0.)

    def test_real_preflight_opens_only_preassigned_plan_config_not_training_states(self):
        x=data.preflight_plan(data.ROOT/'config/framework/P2_onpolicy_probe_TRAIN8_20261103_v1/plan.json',
            data.ROOT/'config/framework/P2_onpolicy_probe_TRAIN8_20261103_v1/response_fit_protocol.json')
        self.assertEqual(sum(j['role']=='fit' for j in x['jobs']),6)
        self.assertEqual(sum(j['role']=='internal_val' for j in x['jobs']),2)
        self.assertFalse(x['actual_state_files_opened']);self.assertFalse(x['fit_started'])

    def test_official_or_dev_VAL_source_is_rejected_before_any_robot_record_open(self):
        with TemporaryDirectory() as directory:
            p=Path(directory)
            (p/'run_inputs.json').write_text(json.dumps(dict(TEST_read=False,config=dict(dataset_use='val',
                P2_response_collection_metadata=dict(parent_id='pretend',split='val',official_VAL_or_TEST_source=True)))))
            # No other file exists. The source role check must precede any
            # attempt to read owner/worker, robot frames or labels.
            with self.assertRaisesRegex(ValueError,'only preassigned new TRAIN'):
                data.extract_case(p,'pretend','fit')

    def test_free_rollout_does_not_reseed_from_future_actual_state(self):
        values,_=episodes();e=values[0]
        class OffsetIntegrator:
            def predict_joint(self,q,dq,qref,vref,aref,dt=.05):
                return dict(q=q+.001,dq=np.ones_like(q)*(.001/dt))
        def fk(q):
            m=np.broadcast_to(np.eye(4),q.shape[:-1]+(4,4)).copy()
            m[...,:3,3]=q[...,:3]+[0.,0.,.8];return m
        report=evaluation.evaluate(OffsetIntegrator(),[e],fk)
        expected=np.linalg.norm((e['q'][:-2,:3]+.002)-e['q'][2:,:3],axis=-1)
        for mode in ('recorded_reference','regenerated_reference'):
            metric=report['roles']['fit']['rollouts'][mode]['by_horizon']['2']['cohorts']['all']['errors']['joint_model_FK_position_l2_m']
            self.assertAlmostEqual(metric['maximum'],float(expected.max()),places=12)

    def test_fit_CLI_default_check_never_calls_fit_or_evaluate(self):
        path=data.ROOT/'scripts/fit_onpolicy_response_20261103.py'
        spec=importlib.util.spec_from_file_location('independent_onpolicy_cli',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with patch.object(module,'load_dataset',return_value=([],{})), \
             patch.object(module,'fit_parameters',side_effect=AssertionError('implicit fit')), \
             patch.object(module,'evaluate',side_effect=AssertionError('implicit evaluate')), \
             patch('sys.argv',[str(path),'--dataset','/tmp/CPU_STUB_NO_SOURCE_READ']), \
             patch('sys.stdout',new_callable=io.StringIO) as output:
            module.main()
        self.assertEqual(json.loads(output.getvalue())['mode'],'check_only_no_fit')


if __name__=='__main__':unittest.main()
