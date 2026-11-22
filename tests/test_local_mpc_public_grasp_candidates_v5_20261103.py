"""Independent v5 wiring contracts, real frozen CPU weights, no Native/fit."""
import ast
from copy import deepcopy
from dataclasses import replace
import dis
import builtins
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace,CodeType,FunctionType
import unittest
from unittest.mock import patch,PropertyMock

import numpy as np
from scipy.spatial.transform import Rotation

from our_mani.contracts import Pose,Forecast
import our_mani.local_mpc_control_20261103.public_grasp_candidates_v5 as v5
import our_mani.local_mpc_control_20261103.public_grasp_v41 as v41
from our_mani.local_mpc_control_20261103.public_grasp_v4 import ActiveVisualPublicBridge
from our_mani.local_mpc_p4_20261103.runtime import ProposalUnavailable
from our_mani.local_mpc_p4_20261103.runtime import LearnedCandidates
from our_mani.local_mpc_p4_20261103.symmetry_runtime import choose_pi_representative,SymmetricLearnedCandidates
from our_mani.local_mpc_control_20261103.forecast_control import BoundForecastExecutionModel
from our_mani.local_mpc_control_20261103.public_grasp import qualify_public_grasp_candidates
from our_mani.local_mpc_control_20261103.public_grasp_probe import qualify_probe_candidates
from our_mani.local_mpc_20261102.mpc import MPCConfig
from our_mani.local_mpc_20261102.terminal_mpc import TerminalCostConfig
from tests.test_local_mpc_active_visual_attachment_20261103 import sample,h
from tests.test_local_mpc_public_grasp_20261103 import RawBackend,GoalHelper
from tests.test_local_mpc_public_pregrasp_20261103 import NumericModel,Executor

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=ROOT/'artifacts/framework/local_mpc_20261102/P4_candidate_training_v1/final.pt'
SHA='61e5827b4838649b8e477048bea56fcace18697f906fb610a3f963bba2437f60'


class RecordingPlanner:
    calls=[]
    def __init__(self,*args,**kwargs):self.model=kwargs['execution_model']
    def plan(self,state,candidates,*,warm_start=None):
        self.calls.append(dict(warm=deepcopy(warm_start),candidates=deepcopy(candidates)))
        return dict(action=dict(joint_goal=np.zeros(7),opening=1.),selected_candidate_id=candidates[0]['id'],
            phase_state=dict(phase='approach',dwell=0.,lost_hold=0.,updated_at=state['time']),
            warm_start=dict(proposal_only=True),budget=dict(sequences=256,horizon=8))


class V5WiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.bridge=ActiveVisualPublicBridge()
    def setUp(self):self.bridge.reset();RecordingPlanner.calls=[]
    def observer(self,raw,directory,**kwargs):
        return v5.ObservedPublicCandidateV5Backend(raw,Path(directory)/'public',public_bridge=self.bridge,
            save_frames=False,checkpoint=CHECKPOINT,checkpoint_sha256=SHA,**kwargs)
    def tracker(self,observation):
        target,_=sample(observation['timestamp'])
        cube=np.array([[x,y,z] for x in (-.025,.025) for y in (-.025,.025) for z in (-.025,.025)])
        target=replace(target,points_world=target.pose.position+cube)
        return target,dict(backend='explicit_CPU_sensor_fixture_known_3D_cube')
    def patches(self):
        return (patch.object(self.bridge.tracker,'observe',side_effect=self.tracker),
                patch.object(self.bridge.bridge.goal_marker,'observe',return_value=(Pose([.3,.1,1.]),{})))
    def core(self):
        return SimpleNamespace(config=MPCConfig(),execution_model=NumericModel(),
            planner=SimpleNamespace(constraint_cost=None),terminal_config=TerminalCostConfig())
    def synthetic_output(self,times):
        return dict(positions=np.zeros((8,3)),rotation_matrices=np.broadcast_to(np.diag([1.,-1.,-1.]),(8,3,3)).copy(),
                    times=np.asarray(times,float)),dict(explicit_CPU_output_fixture=True)

    def test_every_new_function_recursive_LOAD_GLOBAL_resolves(self):
        missing=[]
        def scan(code):
            for op in dis.get_instructions(code):
                if op.opname=='LOAD_GLOBAL' and op.argval not in vars(v5) and not hasattr(builtins,op.argval):
                    missing.append((code.co_name,op.argval))
            for constant in code.co_consts:
                if isinstance(constant,CodeType):scan(constant)
        for value in vars(v5).values():
            if isinstance(value,FunctionType) and value.__module__==v5.__name__:scan(value.__code__)
            if isinstance(value,type) and value.__module__==v5.__name__:
                for member in vars(value).values():
                    if isinstance(member,FunctionType):scan(member.__code__)
        self.assertEqual(missing,[])

    def test_runner_AST_changes_only_version_names_candidate_constructor_and_receipts(self):
        original=ast.parse(inspect.getsource(v41.run_public_grasp_v41))
        current=ast.parse(inspect.getsource(v5.run_public_grasp_candidates_v5))
        replacements={
            'ObservedPublicCandidateV5Backend':'ObservedPublicGraspV41Backend',
            'CandidateGraspV5Coordinator':'ActiveVisualGraspV41Coordinator',
            'run_public_grasp_candidates_v5':'run_public_grasp_v41'}
        strings={'public_grasp_candidate_transformer_v5':'public_grasp_active_visual_v41',
            'public_grasp_candidates_v5_provenance.json':'public_grasp_v41_provenance.json',
            'public_grasp_candidates_v5_decisions.jsonl':'public_grasp_v41_decisions.jsonl',
            'P4_candidate_transformer_v5_':'P3_active_visual_v41_',
            'P4_public_candidate_transformer_grasp_v5_development':'P3_active_visual_grasp_v41_development'}
        extras={'Transformer_used','candidate_transformer','candidate_events','candidate_public_cache','geometry_fallback_used',
                'require_public_goal_completion','public_goal_completion','full_cycle_wall_seconds_samples'}
        class Normalize(ast.NodeTransformer):
            def visit_If(self,node):
                # These exact construction-only branches disappear with the
                # default False modifier; the original controller stays intact.
                skipped=("type(config.get('finger_response_fastpath', False)) is not bool",
                         "config.get('finger_response_fastpath', False)")
                if ast.unparse(node.test) in skipped:return None
                return self.generic_visit(node)
            def visit_FunctionDef(self,node):
                node.name=replacements.get(node.name,node.name);return self.generic_visit(node)
            def visit_Name(self,node):node.id=replacements.get(node.id,node.id);return node
            def visit_Constant(self,node):
                if isinstance(node.value,str):node.value=strings.get(node.value,node.value)
                return node
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='update' and any(k.arg=='candidate_transformer' for k in node.value.keywords):return None
                return self.generic_visit(node)
            def visit_Call(self,node):
                if isinstance(node.func,ast.Name) and node.func.id=='_public_completion_ready':
                    return ast.parse('backend.public_hold_metric.passed',mode='eval').body
                if isinstance(node.func,ast.Name) and node.func.id=='_public_completion_reason':
                    return ast.Constant(value='visible_public_lift_hold_observed')
                if isinstance(node.func,ast.Name) and node.func.id=='CandidateGraspV5Coordinator':
                    node.keywords=[k for k in node.keywords if k.arg not in ('inference','cache','pi_symmetry_representative','cache_fixed_probe_nominal_ik','probe_terminal_at_fixed_deadline','probe_onpolicy_joint_response')]
                return self.generic_visit(node)
            def visit_Return(self,node):
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='dict':
                    node.value.keywords=[k for k in node.value.keywords if k.arg not in extras]
                return self.generic_visit(node)
        self.assertEqual(ast.dump(original),ast.dump(Normalize().visit(current)))

    def test_default_and_explicit_false_keep_the_original_raw_candidate_adapter(self):
        raw=RawBackend(budget=8)
        with TemporaryDirectory() as directory:
            observer=self.observer(raw,directory);tracker,goal=self.patches()
            with tracker,goal:
                observer.sample()
                for _ in range(4):observer.tick('warmup')
                options=dict(config=observer.public_hold_metric.config,recovery=observer.recovery,
                    inference=observer.pilot_inference,cache=observer.candidate_cache)
                default=v5.CandidateGraspV5Coordinator(self.core(),GoalHelper(),observer.assembler,observer.bridge.active,**options)
                explicit=v5.CandidateGraspV5Coordinator(self.core(),GoalHelper(),observer.assembler,observer.bridge.active,
                    pi_symmetry_representative=False,**options)
                self.assertIs(type(default.geometry),LearnedCandidates)
                self.assertIs(type(explicit.geometry),LearnedCandidates)
                self.assertFalse(default.pi_symmetry_representative)
                state=observer.assembler.build(observer.latest,observer.latest_robot,h(.2,target=1.))
                self.assertEqual(default.geometry.update(state),explicit.geometry.update(state))
                with self.assertRaises(ValueError):
                    v5.CandidateGraspV5Coordinator(self.core(),GoalHelper(),observer.assembler,observer.bridge.active,
                        pi_symmetry_representative=1,**options)

    def test_true_helper_right_local_pi_uses_original_deadline_and_SO3_interpolation(self):
        times=np.array([.2,.6,1.]);rot=Rotation.from_euler('xyz',[[.1,.2,.3],[.4,-.3,.7],[-.2,.5,-.4]]).as_matrix()
        forecast=Forecast(('target',),times,np.zeros((3,1,3)),rot[:,None],np.zeros((3,1,3)),np.zeros((3,1,3)))
        raw=Rotation.from_euler('xyz',[.7,-.45,.2]);deadline=.47
        object_at=forecast.at('target',deadline)[0].rotation
        zpi=np.diag([-1.,-1.,1.]);near=object_at@raw.as_matrix()@zpi@Rotation.from_rotvec([.02,0.,0.]).as_matrix()
        choice=choose_pi_representative(raw.as_quat(),forecast,deadline,Rotation.from_matrix(near).as_quat())
        self.assertTrue(choice['available']);self.assertEqual(choice['selection_index'],1)
        chosen=Rotation.from_quat(choice['selected_quaternion_xyzw']).as_matrix()
        np.testing.assert_allclose(chosen,raw.as_matrix()@zpi,atol=1e-12)
        self.assertGreater(np.linalg.norm(chosen-zpi@raw.as_matrix()),.5)
        np.testing.assert_allclose(choice['object_world_rotation_at_deadline'],object_at,atol=1e-12)
        np.testing.assert_allclose(chosen[:,2],raw.as_matrix()[:,2],atol=1e-12)
        np.testing.assert_allclose(chosen[:,1],-raw.as_matrix()[:,1],atol=1e-12)
        self.assertEqual(choice['sample_time'],deadline);self.assertFalse(choice['deadline_or_sample_clamped'])
        self.assertLess(choice['angles_rad'][1],choice['angles_rad'][0])

    def test_true_pool_representation_is_fixed_raw_ledger_unchanged_then_new_generation_can_flip(self):
        class RotatingRobot(RawBackend):
            tcp_q=Rotation.from_matrix(np.diag([-1.,1.,-1.])).as_quat()
            def read_state(self):
                state=super().read_state();state['tcp_quaternion']=self.tcp_q.copy();return state
        raw=RotatingRobot(budget=30)
        with TemporaryDirectory() as directory:
            observer=self.observer(raw,directory);tracker,goal=self.patches()
            with tracker,goal:
                observer.sample()
                for _ in range(4):observer.tick('warmup')
                control=v5.CandidateGraspV5Coordinator(self.core(),GoalHelper(),observer.assembler,observer.bridge.active,
                    config=observer.public_hold_metric.config,recovery=observer.recovery,
                    inference=observer.pilot_inference,cache=observer.candidate_cache,pi_symmetry_representative=True)
                self.assertIs(type(control.geometry),SymmetricLearnedCandidates)
                with patch.object(observer.pilot_inference,'predict',return_value=self.synthetic_output([.3]*8)):
                    state=observer.assembler.build(observer.latest,observer.latest_robot,h(.2,target=1.))
                    first=control.geometry.update(state);raw_first=control.geometry.raw_candidates
                    self.assertTrue(all(c['parallel_gripper_symmetry_choice']==1 for c in first))
                    for selected,original in zip(first,raw_first):
                        self.assertEqual(selected['object_tcp_position'],original['object_tcp_position'])
                        self.assertEqual(selected['deadline'],original['deadline']);self.assertEqual(selected['id'],original['id'])
                        np.testing.assert_allclose(Rotation.from_quat(original['object_tcp_quaternion']).as_matrix(),np.diag([1.,-1.,-1.]),atol=1e-12)
                    raw.tcp_q=Rotation.from_matrix(np.diag([1.,-1.,-1.])).as_quat();observer.tick('new_robot_orientation')
                    state=observer.assembler.build(observer.latest,observer.latest_robot,h(.25,target=1.))
                    same=control.geometry.update(state)
                    self.assertEqual([(c['id'],c['deadline'],c['object_tcp_quaternion']) for c in first],
                                     [(c['id'],c['deadline'],c['object_tcp_quaternion']) for c in same])
                    while observer.clock()<=max(c['deadline'] for c in first)+1e-8:observer.tick('expire')
                    state=observer.assembler.build(observer.latest,observer.latest_robot,h(observer.clock(),target=1.))
                    fresh=control.geometry.update(state)
                    self.assertTrue(all(c['parallel_gripper_symmetry_choice']==0 for c in fresh))
                    self.assertFalse({c['id'] for c in first}&{c['id'] for c in fresh})

    def test_creation_representation_mask_cannot_be_revived_by_FREE_pending_hold_or_probe(self):
        raw=RawBackend(budget=8)
        with TemporaryDirectory() as directory:
            observer=self.observer(raw,directory);tracker,goal=self.patches()
            with tracker,goal:
                observer.sample()
                for _ in range(4):observer.tick('warmup')
                control=v5.CandidateGraspV5Coordinator(self.core(),GoalHelper(),observer.assembler,observer.bridge.active,
                    config=observer.public_hold_metric.config,recovery=observer.recovery,
                    inference=observer.pilot_inference,cache=observer.candidate_cache,pi_symmetry_representative=True)
                with patch.object(observer.pilot_inference,'predict',return_value=self.synthetic_output([.3]*7+[np.nextafter(.8,np.inf)])):
                    state=observer.assembler.build(observer.latest,observer.latest_robot,h(.2,target=1.))
                    candidates=control.geometry.update(state)
                self.assertIsNone(candidates[-1]['parallel_gripper_symmetry_choice'])
                self.assertFalse(candidates[-1]['proposal_available'])
                bound=BoundForecastExecutionModel.from_public_bridge(self.core().execution_model,observer.latest)
                with patch.object(bound,'sample',wraps=bound.sample) as samples:
                    raw.probes=[];qualified,_=qualify_public_grasp_candidates(raw,state,candidates,GoalHelper(),mode='FREE_GRU',bound=bound)
                self.assertFalse(qualified[-1]['native_pose_qualified']);self.assertEqual(len(raw.probes),7)
                self.assertNotIn(candidates[-1]['deadline'],[call.args[0] for call in samples.call_args_list])
                # Merely expired FREE masks must not kill the original later
                # probe/hold pool. The permanent undefined representative does.
                frozen=deepcopy(candidates)
                for c in frozen:c['proposal_available']=False
                for mode in ('PENDING_CLOSE_UNKNOWN_CONTACT','REOPENING_UNKNOWN_CONTACT','ATTACHED_PUBLIC'):
                    raw.probes=[];qualified,_=qualify_public_grasp_candidates(raw,state,frozen,GoalHelper(),mode=mode)
                    self.assertEqual([c['native_pose_qualified'] for c in qualified],[True]*7+[False])
                    self.assertEqual(len(raw.probes),7)
                raw.probes=[]
                spec=dict(target_position=state['tcp_position'].copy(),target_quaternion=state['tcp_quaternion'].copy())
                qualified,_=qualify_probe_candidates(raw,state,frozen,GoalHelper(),spec)
                self.assertEqual([c['native_pose_qualified'] for c in qualified],[True]*7+[False])
                self.assertEqual(len(raw.probes),7)

    def test_real50ms_hook_uses_one_sensor_frame_and_never_packs_Native_poison(self):
        raw=RawBackend(budget=8)
        with TemporaryDirectory() as directory:
            observer=self.observer(raw,directory);tracker,goal=self.patches()
            with tracker,goal:
                observer.sample()
                for _ in range(4):observer.tick('CPU_fixture')
                self.assertEqual(raw.ticks,4);self.assertEqual(raw.captures,5)
                self.assertEqual(observer.candidate_cache.receipt()['stored_history_frames'],5)
                self.assertEqual(observer.candidate_cache.latest['public_robot']['timestamp'],.2)
                self.assertNotEqual(observer.candidate_cache.latest['result']['object_observation']['pose']['position'],[999,888,777])
                output,provenance=observer.pilot_inference.predict(observer.candidate_cache,current_time=.2)
                self.assertEqual(output['positions'].shape,(1,8,3));self.assertEqual(provenance['model_checkpoint_sha256'],SHA)
                raw.requested=0.;observer.tick('closed_intent')
            self.assertTrue(observer.candidate_cache.latest['close_requested'])
            self.assertEqual(observer.candidate_cache.receipt()['stored_history_frames'],0)
            with self.assertRaises(ProposalUnavailable):observer.candidate_cache.features(current_time=.25)
            self.assertFalse(observer.candidate_cache.latest['result']['Native_inputs_used'])

    def test_fixed_pool_deadlines_and_generation_change_clear_warm_before_same_planner(self):
        raw=RawBackend(budget=30)
        with TemporaryDirectory() as directory:
            observer=self.observer(raw,directory);tracker,goal=self.patches()
            with tracker,goal,patch('our_mani.local_mpc_control_20261103.public_grasp_v4.ForecastJointGoalCEMPlanner',RecordingPlanner):
                observer.sample()
                for _ in range(4):observer.tick('warmup')
                control=v5.CandidateGraspV5Coordinator(self.core(),GoalHelper(),observer.assembler,observer.bridge.active,
                    config=observer.public_hold_metric.config,recovery=observer.recovery,inference=observer.pilot_inference,cache=observer.candidate_cache)
                control.decide(raw,observer.latest,observer.latest_robot,h(.2,target=1.))
                first=deepcopy(control.candidates);generation=control.geometry.generation
                observer.tick('same_generation');control.warm={'same_generation_proposal':True}
                control.decide(raw,observer.latest,observer.latest_robot,h(.25,target=1.))
                self.assertEqual([(c['id'],c['deadline']) for c in first],[(c['id'],c['deadline']) for c in control.candidates])
                self.assertEqual(RecordingPlanner.calls[-1]['warm'],{'same_generation_proposal':True})
                while observer.clock()<=max(c['deadline'] for c in first)+1e-8:observer.tick('expire')
                control.warm={'must_not_cross_generation':True}
                control.decide(raw,observer.latest,observer.latest_robot,h(observer.clock(),target=1.))
                self.assertGreater(control.geometry.generation,generation)
                self.assertIsNone(RecordingPlanner.calls[-1]['warm'])
                self.assertFalse({c['id'] for c in first}&{c['id'] for c in control.candidates})

    def test_unavailable_exception_is_explicit_hold_without_geometry_fallback(self):
        raw=RawBackend(budget=4)
        with TemporaryDirectory() as directory:
            observer=self.observer(raw,directory);tracker,goal=self.patches()
            with tracker,goal:observer.sample()
            control=v5.CandidateGraspV5Coordinator(self.core(),GoalHelper(),observer.assembler,observer.bridge.active,
                config=observer.public_hold_metric.config,recovery=observer.recovery,inference=observer.pilot_inference,cache=observer.candidate_cache)
            control.warm={'old':True}
            with patch.object(v5.ActiveVisualGraspV41Coordinator,'decide',side_effect=ProposalUnavailable('not ready')):
                decision=control.decide(raw,observer.latest,observer.latest_robot,h(0.,target=1.))
            self.assertEqual(decision['mode'],'CANDIDATE_PUBLIC_INPUT_UNAVAILABLE')
            self.assertIsNone(decision['state']);self.assertIsNone(decision['action']);self.assertIsNone(control.warm)
            self.assertFalse(decision['geometry_fallback_used']);self.assertFalse(decision['current_held_asserted'])

    def test_runner_dispatches_exact_selected_q_opening_and_two_real_cached_ticks(self):
        for enabled,cached in ((False,False),(True,False),(False,True),(True,True)):
            with self.subTest(pi_symmetry_representative=enabled,cache_fixed_probe_nominal_ik=cached):
                self.bridge.reset();raw=RawBackend(budget=2);selected=np.arange(7)*.01
                helper=SimpleNamespace(qualify_action=lambda backend,state,action,**kwargs:(np.asarray(action['joint_goal']).copy(),None,{'CPU_fixture':True}))
                with TemporaryDirectory() as directory:
                    observer=self.observer(raw,directory);executor=Executor(observer);executor.h=h(0.,target=1.)
                    tracker,goal=self.patches()
                    with tracker,goal,patch.object(v5,'build_response_planner',return_value=(SimpleNamespace(),helper,{})), \
                         patch.object(v5.CandidateGraspV5Coordinator,'decide',return_value=dict(mode='FREE_GRU',state={},action=dict(joint_goal=selected,opening=1.))):
                        observer.sample();report=v5.run_public_grasp_candidates_v5(observer,executor,directory,
                            dict(public_grasp_candidate_transformer_v5=True,simulation_seconds=.1,
                                 candidate_pi_symmetry_representative=enabled,cache_fixed_probe_nominal_ik=cached))
                    self.assertEqual(raw.ticks,2);self.assertEqual(raw.captures,3);self.assertEqual(report['cycles'],1)
                    np.testing.assert_array_equal(executor.commands[0]['q'],selected)
                    self.assertEqual(executor.commands[0]['opening'],1.)
                    self.assertEqual(observer.candidate_cache.receipt()['history_timestamps'],[0.,.05,.1])
                    self.assertFalse(report['Native_success_exit_used'])

    def test_runner_returns_every_existing_cycle_duration_and_matching_quantiles(self):
        raw=RawBackend(budget=6);selected=np.arange(7)*.01
        helper=SimpleNamespace(qualify_action=lambda backend,state,action,**kwargs:
            (np.asarray(action['joint_goal']).copy(),None,{}))
        readings=[10.,10.12,20.,20.34,30.,30.56]
        expected=[readings[i+1]-readings[i] for i in range(0,len(readings),2)]
        with TemporaryDirectory() as directory:
            observer=self.observer(raw,directory);executor=Executor(observer);executor.h=h(0.,target=1.)
            tracker,goal=self.patches()
            with tracker,goal,patch.object(v5,'build_response_planner',return_value=(SimpleNamespace(),helper,{})), \
                 patch.object(v5.CandidateGraspV5Coordinator,'decide',return_value=
                    dict(mode='FREE_GRU',state={},action=dict(joint_goal=selected,opening=1.))), \
                 patch.object(v5,'time',SimpleNamespace(perf_counter=unittest.mock.Mock(side_effect=readings))) as clock:
                observer.sample()
                report=v5.run_public_grasp_candidates_v5(observer,executor,directory,
                    dict(public_grasp_candidate_transformer_v5=True,public_grasp_max_decisions=3))
        self.assertEqual(report['cycles'],3)
        self.assertEqual(report['full_cycle_wall_seconds_samples'],expected)
        self.assertEqual(len(report['full_cycle_wall_seconds_samples']),report['cycles'])
        self.assertEqual(report['full_cycle_wall_seconds'],dict(count=3,p50=float(np.percentile(expected,50)),
            p95=float(np.percentile(expected,95)),max=max(expected)))
        self.assertEqual(clock.perf_counter.call_count,2*report['cycles'])
        self.assertEqual(raw.ticks,6)

    def test_probe_cache_flag_false_delegates_old_hook_without_changing_warm_or_action(self):
        from tests.test_local_mpc_probe_nominal_cache_v53_20261103 import fixture as probe_fixture
        base,state,candidates=probe_fixture()
        raw=RawBackend(budget=2)
        with TemporaryDirectory() as directory:
            observer=self.observer(raw,directory)
            core=SimpleNamespace(config=base.config,execution_model=base.execution_model,
                planner=SimpleNamespace(constraint_cost=base.constraint_cost),terminal_config=base.terminal_config)
            options=dict(config=observer.public_hold_metric.config,recovery=observer.recovery,
                inference=observer.pilot_inference,cache=observer.candidate_cache)
            for explicit in ({},{'cache_fixed_probe_nominal_ik':False}):
                control=v5.CandidateGraspV5Coordinator(core,base.goal_helper,observer.assembler,observer.bridge.active,
                    **options,**explicit)
                self.assertIs(control.cache_fixed_probe_nominal_ik,False)
                self.assertIsNone(control.probe_nominal_cache)
                self.assertIsNone(control.probe_cache_sources)
                warm={'original_optimizer_proposal':np.arange(7)};control.warm=warm
                original_action={'joint_goal':np.arange(7)*.01,'opening':0.}
                result=(candidates,{'old_qualification':True},SimpleNamespace(action=original_action))
                with patch.object(v5.ActiveVisualGraspV41Coordinator,'_verification_components',return_value=result) as old:
                    returned=control._verification_components(raw,state,candidates,base.execution_model,base.probe_spec)
                self.assertIs(returned,result);self.assertIs(control.warm,warm)
                self.assertIs(returned[2].action,original_action)
                old.assert_called_once_with(raw,state,candidates,base.execution_model,base.probe_spec)
            for invalid in (0,1,None,np.bool_(True)):
                with self.subTest(flag=invalid),self.assertRaisesRegex(ValueError,'boolean fixed probe'):
                    v5.CandidateGraspV5Coordinator(core,base.goal_helper,observer.assembler,observer.bridge.active,
                        **options,cache_fixed_probe_nominal_ik=invalid)

    def test_true_probe_hook_shares_destination_keeps_deadline_fresh_h_and_original_path_check(self):
        from tests.test_local_mpc_probe_nominal_cache_v53_20261103 import fixture as probe_fixture,SyntheticNoStepCollision
        from our_mani.local_mpc_control_20261103.public_grasp_probe import VerificationProbePlanner
        from our_mani.local_mpc_p2_20261103.action_adapter import matrix_quaternion
        base,state,candidates=probe_fixture();spec=deepcopy(base.probe_spec)
        spec.update(attempt_id=1,deadline=spec['trigger_time']+.8)
        raw=RawBackend(budget=2)
        with TemporaryDirectory() as directory:
            observer=self.observer(raw,directory)
            core=SimpleNamespace(config=base.config,execution_model=base.execution_model,
                planner=SimpleNamespace(constraint_cost=base.constraint_cost),terminal_config=base.terminal_config)
            control=v5.CandidateGraspV5Coordinator(core,base.goal_helper,observer.assembler,observer.bridge.active,
                config=observer.public_hold_metric.config,recovery=observer.recovery,
                inference=observer.pilot_inference,cache=observer.candidate_cache,cache_fixed_probe_nominal_ik=True)
            rows,qualification,planner=control._verification_components(raw,state,candidates,base.execution_model,spec)
            first=planner.plan(state,rows)
            legacy=VerificationProbePlanner(base.config,execution_model=base.execution_model,goal_helper=base.goal_helper,
                constraint_cost=base.constraint_cost,terminal_config=base.terminal_config,probe_spec=spec).plan(state,rows)
            self.assertEqual(first['budget'],legacy['budget'])
            np.testing.assert_array_equal(first['action']['joint_goal'],legacy['action']['joint_goal'])
            self.assertEqual(first['action']['opening'],legacy['action']['opening'])
            self.assertEqual(len(raw.probes),8)
            saved=control.probe_nominal_cache._entry['joint_goal'].copy()
            fresh=deepcopy(state);fresh['time']+=.1;fresh['q'][1]+=1e-5;fresh['dq'][1]=.001
            fresh['actuator_state'].update(reference_clock=fresh['time'])
            pose=base.goal_helper.kinematics(fresh['q']);fresh['tcp_position']=pose[:3,3]
            fresh['tcp_quaternion']=matrix_quaternion(pose[:3,:3])
            next_rows,receipt,next_planner=control._verification_components(raw,fresh,candidates,base.execution_model,spec)
            second=next_planner.plan(fresh,next_rows,warm_start=first['warm_start'])
            self.assertIs(planner.nominal_cache,next_planner.nominal_cache)
            self.assertEqual(control.probe_nominal_cache.solve_count,1)
            self.assertEqual(control.probe_nominal_cache.reuse_count,1)
            self.assertEqual(receipt['batch_DLS_calls'],0);self.assertEqual(len(raw.probes),16)
            np.testing.assert_array_equal(control.probe_nominal_cache._entry['joint_goal'],saved)
            np.testing.assert_array_equal(second['predicted_states'][0]['q'],fresh['q'])
            self.assertEqual(second['predicted_states'][0]['time'],fresh['time'])
            self.assertEqual(next_planner.probe_spec['deadline'],.8)
            self.assertEqual(spec['deadline'],.8)
            self.assertEqual([(c['id'],c['deadline']) for c in rows],[(c['id'],c['deadline']) for c in next_rows])
            oracle=SyntheticNoStepCollision()
            q,reason,action_receipt=base.goal_helper.qualify_action(oracle,fresh,second['action'],require_predicted_joint_goal=True)
            self.assertIsNone(reason);np.testing.assert_array_equal(q,second['action']['joint_goal'])
            self.assertEqual(second['action']['opening'],0.)
            self.assertEqual(action_receipt['DLS_solves'],0)
            self.assertEqual([r['alpha'] for r in action_receipt['collision_probes']],[1.,.125,.25,.375,.5,.625,.75,.875])
            self.assertEqual(len(oracle.calls),8)

    def test_probe_hook_is_called_only_in_actual_probe_phase_not_FREE_or_ATTACHED(self):
        # This is the sole production call site. Check its owning conditional,
        # rather than mocking the caller and assuming FREE/ATTACHED behavior.
        from our_mani.local_mpc_control_20261103.public_grasp_v4 import ActiveVisualGraspCoordinator
        import textwrap
        tree=ast.parse(textwrap.dedent(inspect.getsource(ActiveVisualGraspCoordinator.decide)))
        owners=[]
        def walk(node,conditions=()):
            if isinstance(node,ast.If):
                for child in node.body:walk(child,conditions+(ast.unparse(node.test),))
                for child in node.orelse:walk(child,conditions+('NOT '+ast.unparse(node.test),))
                return
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='_verification_components':
                owners.append(conditions)
            for child in ast.iter_child_nodes(node):walk(child,conditions)
        walk(tree)
        self.assertEqual(owners,[("mode == 'ACTIVE_VISUAL_VERIFICATION_PROBE'",)])

    def test_goal_completion_flag_defaults_false_and_requires_explicit_bool(self):
        raw=RawBackend(budget=2)
        with TemporaryDirectory() as directory:
            observer=self.observer(raw,directory)
            self.assertIs(observer.require_public_goal_completion,False)
            self.assertIsNone(observer.public_goal_completion)
            for bad in (0,1,None,np.bool_(True)):
                with self.subTest(flag=bad),self.assertRaisesRegex(ValueError,'boolean'):
                    self.observer(raw,directory,require_public_goal_completion=bad)

    def test_goal_completion_only_advances_on_actual50ms_observer_not_candidate_rollout_or_receipt(self):
        raw=RawBackend(budget=8)
        with TemporaryDirectory() as directory:
            observer=self.observer(raw,directory,require_public_goal_completion=True)
            tracker,goal=self.patches()
            with tracker,goal:
                self.assertIsNone(observer.public_goal_completion)
                observer.sample()
                metric=observer.public_goal_completion
                self.assertIsNotNone(metric)
                np.testing.assert_array_equal(metric.goal_pose.position,[.3,.1,1.])
                # The raw Native poison point was never used as the public goal.
                self.assertNotEqual(metric.goal_pose.position.tolist(),[999,888,777])
                with patch.object(metric,'observe',wraps=metric.observe) as samples:
                    for _ in range(4):observer.tick('actual_sensor_tick')
                    self.assertEqual([c.args[1].timestamp for c in samples.call_args_list],[i*.05 for i in range(1,5)])
                    for call in samples.call_args_list:
                        self.assertEqual(call.args[0]['timestamp'],call.args[1].timestamp)
                        self.assertEqual(call.args[2]['latest']['timestamp'],call.args[1].timestamp)
                        self.assertIs(call.kwargs['close_requested'],False)
                    observer.receipt();v5._public_completion_ready(observer)
                    observer.pilot_inference.predict(observer.candidate_cache,current_time=.2)
                    self.assertEqual(samples.call_count,4)
                self.assertEqual(raw.ticks,4);self.assertEqual(raw.captures,5)
                self.assertEqual(observer.public_goal_completion.last_timestamp,.2)
                self.assertIs(observer.public_goal_completion,metric)

    def test_false_keeps_old_hold_stop_true_continues_when_goal_far_and_dispatch_is_exact(self):
        from tests.test_local_mpc_public_grasp_goal_completion_v52_20261103 import frame as completion_frame
        selected=np.arange(7)*.013
        for enabled in (False,True):
            with self.subTest(require_public_goal_completion=enabled):
                self.bridge.reset();raw=RawBackend(budget=2)
                helper=SimpleNamespace(qualify_action=lambda backend,state,action,**kwargs:
                    (np.asarray(action['joint_goal']).copy(),None,{'CPU_fixture':True}))
                with TemporaryDirectory() as directory:
                    observer=self.observer(raw,directory,require_public_goal_completion=enabled)
                    executor=Executor(observer);executor.h=h(0.,target=1.)
                    tracker,goal=self.patches()
                    with tracker,goal,patch.object(v5,'build_response_planner',return_value=(SimpleNamespace(),helper,{})), \
                         patch.object(v5.CandidateGraspV5Coordinator,'decide',return_value=
                            dict(mode='FREE_GRU',state={},action=dict(joint_goal=selected,opening=1.))), \
                         patch.object(type(observer.public_hold_metric),'passed',new_callable=PropertyMock,return_value=True), \
                         patch.object(observer.public_hold_metric,'receipt',side_effect=lambda:completion_frame(observer.clock())[2]):
                        observer.sample()
                        report=v5.run_public_grasp_candidates_v5(observer,executor,directory,
                            dict(public_grasp_candidate_transformer_v5=True,simulation_seconds=.1))
                    self.assertTrue(report['public_lift_hold_observed'])
                    self.assertEqual(report['cycles'],int(enabled))
                    self.assertEqual(len(report['full_cycle_wall_seconds_samples']),report['cycles'])
                    self.assertEqual(raw.ticks,2*int(enabled))
                    self.assertEqual(report['require_public_goal_completion'],enabled)
                    if enabled:
                        self.assertEqual(report['canary_stop_reason'],'budget_exhausted')
                        latest=report['public_goal_completion']['latest']
                        self.assertTrue(latest['public_lift_hold_passed'])
                        self.assertGreater(latest['goal_distance_m'],.02)
                        self.assertFalse(latest['public_grasp_goal_complete'])
                        np.testing.assert_array_equal(executor.commands[0]['q'],selected)
                        self.assertEqual(executor.commands[0]['opening'],1.)
                        self.assertEqual(observer.candidate_cache.receipt()['history_timestamps'],[0.,.05,.1])
                    else:
                        self.assertEqual(report['canary_stop_reason'],'visible_public_lift_hold_observed')
                        self.assertIsNone(report['public_goal_completion'])
                        self.assertEqual(executor.commands,[])
                        self.assertEqual(report['full_cycle_wall_seconds_samples'],[])
                        self.assertEqual(report['full_cycle_wall_seconds'],{'count':0})


if __name__=='__main__':unittest.main()
