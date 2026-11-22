"""V59 exact public producer contracts; no Native, fit or GT/TEST reads."""
import ast,builtins,dis,hashlib,inspect,json,textwrap
from copy import deepcopy
from dataclasses import asdict,replace
from pathlib import Path
from types import CodeType,FunctionType,SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from our_mani.contracts import Pose
from our_mani.benchmark_perception import BoxPrior,RedBoxTracker
from our_mani.perception import CameraCalibration
from our_mani.local_mpc_control_20261103 import public_candidate_darkred_v59 as v59
from our_mani.local_mpc_control_20261103 import public_candidate_goal_maintenance_v56 as v56
from our_mani.local_mpc_control_20261103 import goal_aware_binding_v59 as binding59
from our_mani.local_mpc_control_20261103.darkred_goal_perception_v59 import DarkRedGoalAwareTrackerV59
from our_mani.local_mpc_control_20261103.goal_aware_perception import GoalAwareRedBoxTracker
from our_mani.local_mpc_control_20261103.goal_aware_binding_v42 import ApprovedGoalAwareTrackerCacheBinding
from our_mani.local_mpc_control_20261103.public_attachment import PublicTargetFrameCache
from our_mani.local_mpc_control_20261103.public_motion_bridge import PublicMotionBridge
from our_mani.local_mpc_control_20261103.active_visual_attachment import ActiveVisualConfig
from tests.test_local_mpc_active_visual_attachment_phase_scope_20261103 import sensor_frame,actual_h
from tests.test_local_mpc_goal_aware_binding_v42_20261103 import raw_metadata
from tests.test_local_mpc_goal_aware_perception_20261103 import metadata
from tests.test_local_mpc_public_motion_bridge_20261103 import observation

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'runtime/framework/local_mpc_20261102/public_candidate_finger_fastpath_development_v58_v1/03_val_uniform/public_shadow'


def real_observation(index):
    calibration=json.loads((PUBLIC/'camera_calibration.json').read_text())
    cameras={name:CameraCalibration(value['intrinsics'],Pose.from_xyzw(value['camera_to_world']))
             for name,value in calibration.items()}
    with np.load(PUBLIC/'rgbd'/f'{index:04d}.npz',allow_pickle=False) as source:
        return dict(timestamp=float(source['timestamp']),
            **{'observation/images':{name:source['images_'+name].copy() for name in cameras},
               'observation/depths':{name:source['depths_'+name].copy() for name in cameras},
               'observation/calibrations':cameras})


class V59ProducerTests(unittest.TestCase):
    def wrapper(self):
        return v59.PublicCandidateDarkRedBridgeV59(active_config=ActiveVisualConfig(
            enforce_probe_anchor_orientation_after_confirmation=False))

    def feed(self,wrapper,index,*,close=False,height=0.,shift=(0.,0.,0.)):
        target,robot=sensor_frame(index,height=height,relative_shift=shift)
        with patch.object(wrapper.tracker.marker,'observe',return_value=(Pose([.3,.1,1.]),metadata())),\
             patch.object(wrapper.tracker.tracker,'observe',return_value=(target,raw_metadata())),\
             patch.object(wrapper.bridge.goal_marker,'observe',return_value=(Pose([.3,.1,1.]),metadata())):
            result=wrapper.observe(observation(robot.timestamp),robot,close_requested=close)
        return result,robot,target

    def test_only_fixed_color_prior_changes_and_old_capability_stays_closed(self):
        tracker=DarkRedGoalAwareTrackerV59()
        old=asdict(BoxPrior());new=asdict(tracker.prior)
        self.assertEqual({key for key in old if old[key]!=new[key]},{'minimum_red'})
        self.assertEqual(new['minimum_red'],100);self.assertIs(type(tracker.tracker),RedBoxTracker)
        plain=GoalAwareRedBoxTracker(tracker=RedBoxTracker(prior=BoxPrior(minimum_red=100)))
        cache=PublicTargetFrameCache(plain);bridge=PublicMotionBridge(tracker=cache)
        with self.assertRaises(ValueError):ApprovedGoalAwareTrackerCacheBinding(bridge,plain,cache)
        wrapper=self.wrapper();assembler=v59.PublicCandidateDarkRedStateAssemblerV59(wrapper.binding)
        self.assertFalse(assembler.allow_test_doubles)
        with self.assertRaises(TypeError):v59.PublicCandidateDarkRedStateAssemblerV59(SimpleNamespace())
        prior=wrapper.tracker.tracker.prior
        wrapper.tracker.tracker.prior=BoxPrior(minimum_red=99)
        with self.assertRaises(ValueError):wrapper.binding.validate_dependencies()
        wrapper.tracker.tracker.prior=prior
        class Unapproved(DarkRedGoalAwareTrackerV59):pass
        with self.assertRaises(ValueError):
            newtracker=Unapproved();newcache=PublicTargetFrameCache(newtracker)
            binding59.ApprovedDarkRedGoalAwareTrackerCacheBindingV59(PublicMotionBridge(tracker=newcache),newtracker,newcache)

    def test_flags_types_and_authenticated_aliases_cannot_change(self):
        for bad in (False,0,1,None,'yes',np.bool_(True)):
            with self.assertRaises(ValueError):v59.v59_enabled(goal_sphere_filter=bad)
            with self.assertRaises(ValueError):v59.v59_enabled(postconfirmed_maintenance=bad)
        wrapper=self.wrapper()
        wrapper.tracker=DarkRedGoalAwareTrackerV59()
        with self.assertRaisesRegex(ValueError,'authenticated'):wrapper.receipt()
        wrapper=self.wrapper();wrapper.postconfirmed_maintenance=False
        with self.assertRaisesRegex(ValueError,'modifiers'):wrapper.receipt()
        for flag in (None,False,1,'true'):
            with self.assertRaises(ValueError):v59._validate_v59_runner(SimpleNamespace(),dict(candidate_darkred_target_v59=flag))

    def test_real_four_view_RGBD_matches_exact_R100_math_without_rewriting_pixels(self):
        wrapper=self.wrapper();reference=GoalAwareRedBoxTracker(tracker=RedBoxTracker(prior=BoxPrior(minimum_red=100)))
        standalone=DarkRedGoalAwareTrackerV59()
        for index in (0,1):
            obs=real_observation(index);before=binding59.public_RGBD_digest(obs)
            expected,expected_metadata=reference.observe(obs)
            measured,new_metadata=standalone.observe(obs)
            _,robot=sensor_frame(index,timestamp=obs['timestamp'])
            result=wrapper.observe(obs,robot,close_requested=False)
            actual=result['object_observation']
            np.testing.assert_array_equal(expected.pose.position,actual.pose.position)
            np.testing.assert_array_equal(expected.pose.position,measured.pose.position)
            self.assertEqual(binding59.public_RGBD_digest(obs),before)
            self.assertEqual(set(obs['observation/images']),{'front','8','16','36'})
            self.assertTrue(actual.visible);self.assertEqual(actual.confidence,1.)
            self.assertFalse(new_metadata['box_prior_unchanged']);self.assertTrue(new_metadata['box_geometry_prior_unchanged'])
            self.assertTrue(new_metadata['color_prior_changed']);self.assertFalse(new_metadata['raw_RGB_brightness_synthesized'])
            self.assertEqual(result['goal_aware_binding_version'],'v59')
            self.assertTrue(result['tracker_dependency_injected']);self.assertFalse(result['Native_inputs_used'])
            self.assertFalse(result['forecaster_is_injected_test_double'])
            self.assertIs(wrapper.binding.latest_result,result)
        receipt=wrapper.binding.receipt()
        self.assertEqual(receipt['fixed_minimum_red'],100)
        self.assertFalse(receipt['preclose_bitexact_with_original_R170'])
        self.assertIsNotNone(receipt['scene_identity']['initial_raw_RGBD_sha256'])
        self.assertFalse(wrapper.receipt()['box_prior_unchanged'])

    def test_first_sphere_failure_stays_unknown_and_cache_receives_only_raw7(self):
        wrapper=self.wrapper();target,robot=sensor_frame(0)
        with patch.object(wrapper.tracker.marker,'observe',side_effect=ValueError('no first sphere')),\
             patch.object(wrapper.tracker.tracker,'observe',return_value=(target,raw_metadata())):
            result=wrapper.observe(observation(0.),robot,close_requested=False)
        self.assertIsNone(result['object_observation']);self.assertFalse(result['available'])
        self.assertFalse(result['free_forecast_available']);self.assertFalse(wrapper.maintenance.started)
        self.assertTrue(result['goal_aware_filter_trace']['strict_initial_fit_unavailable_output_rejected_for_policy'])
        wrapper=self.wrapper();seen=[];stage=wrapper.cache.stage_public_target
        def record(*args,**kwargs):seen.append(kwargs['diagnostic']);return stage(*args,**kwargs)
        with patch.object(wrapper.cache,'stage_public_target',side_effect=record):self.feed(wrapper,0)
        self.assertEqual(set(seen[0]),binding59.RAW_TRACKER_KEYS)

    def test_strict_input_identity_clock_sphere_calibration_and_sources(self):
        wrapper=self.wrapper();result,_,_=self.feed(wrapper,0)
        with self.assertRaises(ValueError):wrapper.binding.approve(deepcopy(result))
        bad=observation(.05);bad['Native_held']=True
        with self.assertRaises(ValueError):wrapper.binding.begin_frame(bad)
        with self.assertRaises(ValueError):wrapper.binding.begin_frame(observation(float('nan')))
        with self.assertRaises(ValueError):wrapper.binding.begin_frame(observation(0.))
        original=wrapper.tracker.sphere;wrapper.tracker._sphere=replace(original)
        with self.assertRaisesRegex(ValueError,'sphere identity'):wrapper.binding.validate_dependencies()
        wrapper.tracker._sphere=original
        changed=observation(.05);camera=changed['observation/calibrations']['front'];K=camera.intrinsics.copy();K[0,0]*=1.01
        changed['observation/calibrations']['front']=CameraCalibration(K,camera.camera_to_world)
        with self.assertRaisesRegex(ValueError,'calibration identity'):wrapper.binding.begin_frame(changed)
        for attr in ('_mock_forecaster','_goal_marker_injected'):
            setattr(wrapper.bridge,attr,True)
            with self.assertRaises(ValueError):wrapper.binding.validate_dependencies()
            setattr(wrapper.bridge,attr,False)
        with patch.dict(binding59.APPROVED_COMPONENT_SHA256,{'our_mani/benchmark_perception.py':'0'*64}):
            with self.assertRaisesRegex(ValueError,'SHA'):wrapper.binding.validate_dependencies()
        old_id=wrapper.binding.episode_id;wrapper.reset();self.assertNotEqual(wrapper.binding.episode_id,old_id)

    def test_original_probe_then_10mm_position_maintenance_remains_current_visual(self):
        wrapper=self.wrapper();self.feed(wrapper,0)
        _,robot,target=self.feed(wrapper,1,close=True)
        wrapper.active.begin_probe(target,robot,actual_h(robot))
        for index in range(2,7):result,robot,_=self.feed(wrapper,index,close=True,height=.012*(index-1)/5)
        self.assertEqual(result['attachment_estimate']['protocol'],'active_vertical_visual_comotion_v4')
        self.assertTrue(wrapper.maintenance.started)
        result,robot,target=self.feed(wrapper,7,close=True,height=.05,shift=(.009,0.,0.))
        self.assertEqual(result['mode'],'ATTACHED_PUBLIC');self.assertIs(result['object_observation'],target)
        self.assertEqual(result['attachment_estimate']['protocol'],v56.MAINTENANCE_PROTOCOL)
        self.assertTrue(wrapper.active.failed)
        result,_,_=self.feed(wrapper,8,close=True,height=.05,shift=(.011,0.,0.))
        self.assertEqual(result['mode'],'UNRESOLVED');self.assertFalse(result['free_forecast_available'])

    def test_runner_and_control_math_AST_are_mechanical_names_only(self):
        source=inspect.getsource(v59.run_public_candidate_darkred_v59)
        for new,old in [('run_public_candidate_darkred_v59','run_public_candidate_goal_maintenance_v56'),
            ('ObservedPublicCandidateDarkRedV59Backend','ObservedPublicCandidateV56Backend'),
            ('CandidateGraspDarkRedV59Coordinator','CandidateGraspV56Coordinator'),
            ('_validate_v59_runner','_validate_v56_runner'),
            ('public_candidate_darkred_v59_provenance.json','public_candidate_goal_maintenance_v56_provenance.json'),
            ('public_candidate_darkred_v59_decisions.jsonl','public_candidate_goal_maintenance_v56_decisions.jsonl'),
            ('P4_candidate_transformer_v59_','P4_candidate_transformer_v56_'),
            ('P4_public_candidate_darkred_v59_development','P4_public_candidate_goal_maintenance_v56_development')]:source=source.replace(new,old)
        self.assertEqual(ast.dump(ast.parse(source)),ast.dump(ast.parse(inspect.getsource(v56.run_public_candidate_goal_maintenance_v56))))
        method=inspect.getsource(v59.PublicCandidateDarkRedBridgeV59.observe).replace("goal_aware_binding_version='v59'","goal_aware_binding_version='v42'")
        self.assertEqual(ast.dump(ast.parse(textwrap.dedent(method))),ast.dump(ast.parse(textwrap.dedent(inspect.getsource(v56.PublicCandidateBridgeV56.observe)))))
        self.assertEqual(hashlib.sha256(Path(v56.__file__).read_bytes()).hexdigest(),v59.RUNNER_BASE_SOURCE_SHA256)
        self.assertIn('finger_response_fastpath',source);self.assertIn('execute_joint_target',source)
        self.assertIs(v59.build_response_planner,v56.build_response_planner)
        self.assertIs(v59.CandidateGraspDarkRedV59Coordinator.__bases__[0],v56.CandidateGraspV56Coordinator.__bases__[0])

    def test_added_functions_global_names_all_resolve(self):
        missing=[]
        def scan(code):
            for op in dis.get_instructions(code):
                if op.opname=='LOAD_GLOBAL' and op.argval not in vars(v59) and not hasattr(builtins,op.argval):missing.append(op.argval)
            for const in code.co_consts:
                if isinstance(const,CodeType):scan(const)
        for value in vars(v59).values():
            if isinstance(value,FunctionType) and value.__module__==v59.__name__:scan(value.__code__)
            if isinstance(value,type) and value.__module__==v59.__name__:
                for member in vars(value).values():
                    if isinstance(member,FunctionType):scan(member.__code__)
        self.assertEqual(missing,[])


if __name__=='__main__':unittest.main()
