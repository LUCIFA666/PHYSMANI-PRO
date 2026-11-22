"""Probe-only warm ablation, no Native/fit/action override."""
import ast,inspect,json,hashlib,dis,builtins,types
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from our_mani.contracts import ObjectState,Pose
from our_mani.local_mpc_control_20261103 import public_grasp_v41 as baseline
from our_mani.local_mpc_control_20261103 import public_grasp_cold_probe_v43 as cold
from our_mani.local_mpc_control_20261103.active_visual_attachment import ActiveVisualAttachment,ActiveVisualConfig
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicRobotSensors
from tests.test_local_mpc_control_20261103_joint_goal_cem import fixture
from tests.test_local_mpc_active_visual_attachment_20261103 import h


class ColdProbeTests(unittest.TestCase):
    def controller(self):return cold.ColdProbeGraspV43Coordinator(SimpleNamespace(),object(),object(),ActiveVisualAttachment())
    def test_existing_probe_warm_removed_both_sides_and_selected_action_identical(self):
        control=self.controller();control.mode='ACTIVE_VISUAL_VERIFICATION_PROBE';control.warm={'old_probe':True}
        state_h={'fresh_actual_marker':123};before=deepcopy(state_h);action={'joint_goal':np.arange(7.),'opening':0.}
        def parent(self,*args):
            self.asserted_warm_input=self.warm;self.warm={'output_for_audit_only':True}
            return dict(mode='ACTIVE_VISUAL_VERIFICATION_PROBE',action=action,plan=dict(diagnostics={},budget={'sequences':256,'horizon':8}))
        with patch.object(baseline.ActiveVisualGraspV41Coordinator,'decide',parent):result=control.decide(None,None,None,state_h)
        self.assertIsNone(control.asserted_warm_input);self.assertIsNone(control.warm)
        self.assertIs(result['action'],action);self.assertEqual(state_h,before)
        self.assertFalse(result['probe_warm_policy']['nominal_action_forced'])
        self.assertFalse(result['probe_warm_policy']['action_postprocessed'])
        self.assertEqual(result['plan']['budget'],{'sequences':256,'horizon':8})

    def test_FREE_ATTACHED_and_other_nonprobe_memory_stays_under_original_policy(self):
        for mode in ('FREE_GRU','ATTACHED_PUBLIC','PENDING_CLOSE_UNKNOWN_CONTACT','POST_CONFIRMED_UNKNOWN_HOLD'):
            with self.subTest(mode=mode):
                control=self.controller();control.mode=mode;old={'nonprobe':True};new={'fresh_parent_output':True};control.warm=old
                def parent(self,*args):
                    self.seen=self.warm;self.warm=new;return dict(mode=mode,action=None)
                with patch.object(baseline.ActiveVisualGraspV41Coordinator,'decide',parent):result=control.decide(None,None,None,{})
                self.assertIs(control.seen,old);self.assertIs(control.warm,new)
                self.assertNotIn('probe_warm_policy',result)

    def test_probe_to_attached_keeps_new_attached_warm_after_parent_transition(self):
        control=self.controller();control.mode='ACTIVE_VISUAL_VERIFICATION_PROBE';control.warm={'probe':True};new={'attached':True}
        def parent(self,*args):
            assert self.warm is None;self.warm=new;return dict(mode='ATTACHED_PUBLIC',action=None)
        with patch.object(baseline.ActiveVisualGraspV41Coordinator,'decide',parent):control.decide(None,None,None,{})
        self.assertIs(control.warm,new)

    def test_first_probe_entry_parent_mode_transition_clears_pending_warm_before_planner(self):
        base,state,candidates=fixture();state.update(opening=.6,finger_open_amounts=np.array([.7,.3]),requested_opening=0.)
        active=ActiveVisualAttachment();obj=ObjectState('target',0.,Pose(state['object_position']),np.zeros((3,3)),np.zeros(3),np.zeros(3))
        robot=PublicRobotSensors(0.,state['tcp_position'],state['tcp_quaternion'],state['q'],state['dq'],state['finger_open_amounts'])
        a=active.observe(obj,robot,close_requested=True)
        class Assembler:
            robot_twist=(np.zeros(3),np.zeros(3),True)
            binding=SimpleNamespace(latest_visual_target=obj)
            def build(self,*args,**kwargs):return None
            def build_pending(self,*args,**kwargs):
                value=deepcopy(state);value['phase_state']=kwargs.get('phase_state');return value
        class Oracle:
            def probe_joint_collision(self,q):return False
        seen=[]
        class ProbePlanner:
            def __init__(self,*args,**kwargs):pass
            def plan(self,state,candidates,*,warm_start):
                seen.append(warm_start)
                return dict(action=dict(joint_goal=np.zeros(7),opening=0.),selected_candidate_id=candidates[0]['id'],
                    phase_state=state['phase_state'],warm_start={'would_reuse':True},diagnostics={},budget={'sequences':256,'horizon':8})
        core=SimpleNamespace(config=base.config,execution_model=base.execution_model,planner=SimpleNamespace(constraint_cost=None),terminal_config=base.terminal_config)
        control=cold.ColdProbeGraspV43Coordinator(core,base.goal_helper,Assembler(),active)
        control.mode='PENDING_CLOSE_UNKNOWN_CONTACT';control.pending_started=-.5;control.candidates=candidates;control.warm={'pending':True}
        result=dict(timestamp=0.,mode='UNRESOLVED',available=False,attachment_estimate=a)
        with patch('our_mani.local_mpc_control_20261103.public_grasp_v4.VerificationProbePlanner',ProbePlanner):decision=control.decide(Oracle(),result,robot,h(0.))
        self.assertEqual(seen,[None]);self.assertIsNone(control.warm)
        self.assertEqual(decision['mode'],'ACTIVE_VISUAL_VERIFICATION_PROBE')

    def test_runner_AST_diff_only_declared_names_flag_and_coordinator_and_exact_old_observer(self):
        source=inspect.getsource(cold.run_public_grasp_cold_probe_v43)
        for original,new in sorted(cold.RUNNER_APPROVED_RENAMES.items(),key=lambda pair:len(pair[1]),reverse=True):source=source.replace(new,original)
        self.assertEqual(ast.dump(ast.parse(source)),ast.dump(ast.parse(inspect.getsource(baseline.run_public_grasp_v41))))
        self.assertEqual(hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest(),cold.RUNNER_BASE_SOURCE_SHA256)
        self.assertIs(cold.ObservedPublicGraspV41Backend,baseline.ObservedPublicGraspV41Backend)
        self.assertIs(cold.build_response_planner,baseline.build_response_planner)

    def test_runner_all_global_loads_including_nested_code_resolve(self):
        function=cold.run_public_grasp_cold_probe_v43;codes=[function.__code__];missing=[]
        while codes:
            code=codes.pop()
            codes.extend(value for value in code.co_consts if isinstance(value,types.CodeType))
            for instruction in dis.get_instructions(code):
                if instruction.opname=='LOAD_GLOBAL':
                    name=instruction.argval
                    if name not in function.__globals__ and not hasattr(builtins,name):missing.append((code.co_name,name))
        self.assertEqual(missing,[])
        self.assertIs(cold.PublicRobotSensors,baseline.PublicRobotSensors)

    def test_new_configs_keep_point8_and_original_physics_cost_and_evidence_settings(self):
        root=Path(__file__).resolve().parents[1]
        for suffix in ('','_high'):
            old=json.loads((root/('config/framework/local_mpc_20261103_public_grasp_active_visual'+suffix+'_v41.json')).read_text())
            new=json.loads((root/('config/framework/local_mpc_20261103_public_grasp_cold_probe'+suffix+'_v43.json')).read_text())
            for key in ('active_visual_settings','public_grasp_settings','physics_dt','control_dt','horizon_steps','horizon_seconds','cem_sequences_total','cem_iteration_sequences','terminal_goal_cost','response_controller','uniform_velocity','joint_reference_acceleration','joint_reference_jerk'):
                self.assertEqual(old[key],new[key],key)
            self.assertEqual(new['active_visual_settings']['probe_seconds'],.8)
            ActiveVisualConfig(**new['active_visual_settings'])
            self.assertTrue(new['public_grasp_cold_probe_v43']);self.assertFalse(new['probe_warm_start_enabled'])
            self.assertFalse(new['public_grasp_active_visual_v41']);self.assertTrue(new['cold_policy_not_nominal_only'])


if __name__=='__main__':unittest.main()
