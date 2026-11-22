from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS,MethodType
import json
import unittest
import numpy as np
from our_mani.contracts import Pose,RobotState
from our_mani.multitask.normal_cup_primary_library_v109 import PrimarySequenceLibrary
from our_mani.multitask.interaction import _world_target
from our_mani.multitask.cup_author_action_guard_v251 import (
    binding,install_completion,install_release,prefix_guard,install_prefix_transition,install_prefix_confirmation)

ROOT=Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def setUp(self):
        self.protocol=json.loads((ROOT/'artifacts/framework/multitask_r10/cup_slack_v208_setup/configuration.json').read_text())
        self.library=PrimarySequenceLibrary.load(ROOT/self.protocol['learned_artifacts']['place_cups_on_rotating_frame']['interaction']['path'])
        self.template=self.library.for_phase('approach')[0];self.reference=Pose([.2,.1,.8])
        self.target=_world_target(self.reference,np.zeros(3),np.zeros(3),self.template,1.)[0]
        now=2.+self.template.duration_seconds+.05
        robot=RobotState(now,Pose(self.target.position+[.08,0.,0.],self.target.rotation),
            np.zeros(3),np.zeros(3),np.zeros(7),np.zeros(7),gripper_open=True,contact=False)
        self.state=NS(robot=robot,role_bindings={'cup0':('cup0',)},
            perception_metadata={'bodies':{'cup0':{'motion_mode':'external'}}})
        goal=NS(template=self.template,phase='approach',selected_at=2.,phase_budget=4.,
            reference_id='cup0',manipulated_ids=('cup0',),candidate_id='actual-task-candidate')
        self.c=NS(goal=goal,phase_index=0,origin=None,library=self.library,trajectory=None,committed_roles={},
            generator=NS(observed_reference=lambda *args:NS(pose=self.reference)),
            _phase_done=lambda *args,**kwargs:False,_hold=lambda reason:{'command':'hold','reason':reason},
            step=lambda state,forecast,remaining:{'phase_index':self.c.phase_index},
            _gripper_opening=lambda *args:1.)
        self.invalidations=[]
        self.m=NS(transport_phase_clock=NS(epochs={}),qualification_failed=False,
            last_confirmed={'time':now,'opening':0.},capture_stopped=lambda robot:True,
            braking_active=lambda now:False,invalidate=lambda reason,now:self.invalidations.append(reason),
            cancel_unissued_retarget=lambda:None,pending=None)
        self.events=[]
        self.witness=dict(phase='approach',phase_index=0,reference_id='cup0',manipulated_ids=['cup0'],
            issued_at=2.,confirmed_after=now,deadline=6.,actual_prefix_plan_id='actual-prefix')

    def test_original_author_sources_and_thresholds_bound_for_both_tasks(self):
        for task in ('place_cups_on_rotating_frame','place_cups_on_rotating_frame_high_speed'):
            proof=binding(ROOT,self.protocol,task)
            self.assertEqual(proof['actual_position_guards_m'],{'approach':.1,'transport':.1,'align':.02,'hang':.03})
            self.assertTrue(proof['phase_orientation_completion_guard_replaced_by_author_position_guard'])

    def test_current_public_geometry_used_instead_of_future_target_argument(self):
        install_completion(self.c,self.m,self.events)
        self.assertTrue(self.c._phase_done(self.state,Pose([9.,9.,9.])))
        self.assertEqual(self.template.position_tolerance_m,.005)
        self.reference=Pose(self.reference.position+[.4,0.,0.])
        self.assertFalse(self.c._phase_done(self.state,self.state.robot.pose))

    def test_fitted_elapsed_and_original_deadline_remain_required(self):
        install_completion(self.c,self.m,self.events)
        self.state.robot=replace(self.state.robot,timestamp=2.01)
        self.assertFalse(self.c._phase_done(self.state,self.target))
        self.state.robot=replace(self.state.robot,timestamp=6.1)
        self.assertFalse(self.c._phase_done(self.state,self.target))

    def test_acquire_delegates_to_original_physical_capture_verifier(self):
        called=[];self.c._phase_done=lambda *args,**kwargs:called.append('original physical acquire') or False
        self.c.goal.template=self.library.for_phase('acquire')[0]
        install_completion(self.c,self.m,self.events)
        self.assertFalse(self.c._phase_done(self.state,self.target))
        self.assertEqual(called,['original physical acquire'])

    def test_unissued_expired_or_changed_reference_prefix_cannot_advance(self):
        self.c.goal=None
        self.assertIsNone(prefix_guard(self.c,self.m,self.state,None))
        bad=dict(self.witness,deadline=2.1)
        self.assertIsNone(prefix_guard(self.c,self.m,self.state,bad))
        bad=dict(self.witness,reference_id='cup1')
        self.assertIsNone(prefix_guard(self.c,self.m,self.state,bad))

    def test_qualified_prefix_waits_for_original_verified_brake(self):
        self.c.goal=None;self.m.capture_stopped=lambda robot:False
        install_prefix_transition(self.c,self.m,{0:self.witness},self.events)
        command=self.c.step(self.state,NS(metadata={'observed_until':self.state.robot.timestamp}),6.)
        self.assertEqual(command['command'],'hold');self.assertEqual(self.c.phase_index,0)
        self.assertIn('original_verified_brake',command['reason'])

    def test_stopped_qualified_prefix_advances_without_candidate_or_training_label(self):
        self.c.goal=None
        def refuse(*args):raise AssertionError('unselected fitted candidate must not get a training label')
        self.c.attempts=NS(skill_completed=refuse)
        install_prefix_transition(self.c,self.m,{0:self.witness},self.events)
        command=self.c.step(self.state,NS(metadata={'observed_until':self.state.robot.timestamp}),6.)
        self.assertEqual(command['phase_index'],1);self.assertEqual(self.c.committed_roles['cup0'],('cup0',))
        self.assertFalse(self.events[-1]['task_candidate_fabricated'])
        self.assertFalse(self.events[-1]['fitted_skill_training_label_created'])

    def test_prefix_witness_requires_actual_qualified_dispatch(self):
        self.m.approach_transit=NS(active=dict(self.witness));self.m.plan_kind='approach_transit'
        self.m.plan_candidate_id='actual-prefix';self.m.confirm_actual_issued=lambda **kwargs:None
        witnesses={};install_prefix_confirmation(self.m,witnesses,self.events)
        kwargs=dict(time=self.state.robot.timestamp,command={'feedforward_execution':{'candidate_id':'actual-prefix'}},
            execution={'actual_issued_at':self.state.robot.timestamp-.05,
                'actual_window_velocity_within_limits':True,'actual_window_acceleration_within_limits':False})
        self.m.confirm_actual_issued(**kwargs);self.assertFalse(witnesses)
        kwargs['execution']['actual_window_acceleration_within_limits']=True
        self.m.confirm_actual_issued(**kwargs);self.assertIn(0,witnesses)

    def test_hang_release_deferred_until_current_geometry_and_stop(self):
        template=self.library.for_phase('hang')[0];self.c.goal.template=template;self.c.goal.phase='hang'
        self.c.goal.reference_id='spoke0'
        self.target=_world_target(self.reference,np.zeros(3),np.zeros(3),template,1.)[0]
        install_release(self.c,self.m,self.events)
        self.state.robot=replace(self.state.robot,pose=Pose(self.target.position+[.04,0.,0.]))
        self.assertEqual(self.c._gripper_opening(self.state,1.,None),0.)
        self.state.robot=replace(self.state.robot,pose=self.target)
        self.m.capture_stopped=lambda robot:False
        self.assertEqual(self.c._gripper_opening(self.state,1.,None),0.)
        self.m.capture_stopped=lambda robot:True
        self.assertEqual(self.c._gripper_opening(self.state,1.,None),1.)

    def test_transport_prefix_requires_unique_public_held_identity_and_original_clock(self):
        template=self.library.for_phase('transport')[0]
        self.c.goal=None;self.c.phase_index=3
        target=_world_target(self.reference,np.zeros(3),np.zeros(3),template,1.)[0]
        self.state.robot=replace(self.state.robot,timestamp=4.,pose=target,contact=True)
        self.state.role_bindings={'selected_spoke':('spoke0',),'cup0':('cup0',)}
        self.state.perception_metadata['bodies']={'spoke0':{'motion_mode':'external'},
            'cup0':{'motion_mode':'attached','mode_provenance':'caller_controller_touch_action_proprioception'}}
        self.m.last_confirmed['time']=4.
        self.m.transport_phase_clock.epochs[3]={'issued_at':2.,'deadline':6.}
        witness=dict(self.witness,phase='transport',phase_index=3,reference_id='spoke0')
        self.assertIsNotNone(prefix_guard(self.c,self.m,self.state,witness))
        self.state.perception_metadata['bodies']['cup1']={'motion_mode':'attached'}
        self.assertIsNone(prefix_guard(self.c,self.m,self.state,witness))
        self.state.perception_metadata['bodies'].pop('cup1');self.m.transport_phase_clock.epochs.clear()
        self.assertIsNone(prefix_guard(self.c,self.m,self.state,witness))


if __name__=='__main__':unittest.main()
