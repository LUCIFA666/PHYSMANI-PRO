"""CPU contracts: no simulation, worker, Native import or source-overlay write."""
import ast
import hashlib
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest

import numpy as np

from our_mani.contracts import Forecast, Pose, RobotState
from our_mani.multitask.interaction import PolicyState
from our_mani.multitask import physmani_ring_release_v590 as ring
from our_mani.multitask import physmani_skill_policy_v548 as skill
from our_mani.multitask.native_outcome_sampling_v741 import ActionOutcomeSampling, NONTERMINAL
from our_mani.multitask.passive_execution_diagnostic_v1162 import (
    PassiveExecutionDiagnostic, ProfileSite, frozen_source_sites,
)


def public_state(*, opened=True, contact=False):
    robot = RobotState(4.2, Pose(np.zeros(3)), np.zeros(3), np.zeros(3),
                       np.zeros(7), np.zeros(7), gripper_open=opened, contact=contact)
    return PolicyState(robot, {}, {}, {})


def returns(tap, label):
    return [e for e in tap.events if e['site']==label and e['event']=='original_python_return_event']


class PassiveExecutionTests(unittest.TestCase):
    def make_ring(self, target, *, opened=True, contact=False):
        source = ring.AuthorRingReleaseIntents.__new__(ring.AuthorRingReleaseIntents)
        source.frontend=SimpleNamespace(state=public_state(opened=opened,contact=contact))
        source.selected={'phase':'release','reference_id':'peg0','manipulated_ids':('ring',)}
        source.phase_index=5;source.phase_order=('approach','acquire','lift','align','insert','release')
        source.finished=False; calls=[]
        def completion(owned, goal):
            calls.append(goal)
            return target
        source._completion_target=MethodType(completion,source)
        sites=(ProfileSite.from_callable('release.done',source._done,kind='release_done'),
               ProfileSite.from_callable('completion',source._completion_target,kind='completion'),
               ProfileSite.from_callable('geometry.errors',skill.errors,kind='errors'))
        return source,calls,sites

    def test_real590_original_bool_and_actual_errors_tuple_once(self):
        source,calls,sites=self.make_ring(Pose(np.array([.02,0.,0.])))
        with PassiveExecutionDiagnostic(sites) as tap:
            result=source._done(source.selected)
        self.assertIs(result,True); self.assertEqual(len(calls),1)
        e=returns(tap,'release.done')[0]
        self.assertEqual(e['direct_done_errors_return'],[.02,0.])
        self.assertEqual(e['release_leaf_evidence'],dict(target_present=True,gripper_open=True,
                         not_contact=True,distance_m=.02,distance_le_3cm=True))
        self.assertEqual(tap.counts['geometry.errors'],dict(calls=1,return_events=1))
        self.assertEqual(returns(tap,'geometry.errors')[0]['parent_call_id'],e['call_id'])
        self.assertTrue(tap.receipt()['diagnostic_available']); self.assertIsNone(sys.getprofile())

    def test_real590_shortcircuits_are_not_false_distance_measurements(self):
        for target,opened,contact,leaf in ((None,True,False,'target_present'),
            (Pose(np.zeros(3)),False,False,'gripper_open'),(Pose(np.zeros(3)),True,True,'not_contact')):
            with self.subTest(leaf=leaf):
                source,calls,sites=self.make_ring(target,opened=opened,contact=contact)
                with PassiveExecutionDiagnostic(sites) as tap:result=source._done(source.selected)
                self.assertIs(result,False); self.assertEqual(len(calls),1)
                self.assertEqual(tap.counts['geometry.errors']['calls'],0)
                leaves=returns(tap,'release.done')[0]['release_leaf_evidence']
                self.assertIs(leaves[leaf],False)
                self.assertEqual(leaves['distance_le_3cm']['status'],'not_evaluated')

    def test_missing_error_probe_does_not_recompute_or_invent_distance(self):
        source,calls,sites=self.make_ring(Pose(np.array([.04,0.,0.])))
        with PassiveExecutionDiagnostic(sites[:2]) as tap: result=source._done(source.selected)
        self.assertIs(result,False); self.assertEqual(len(calls),1)
        leaf=returns(tap,'release.done')[0]['release_leaf_evidence']['distance_m']
        self.assertEqual(leaf['status'],'unavailable')

    def test_missing_locals_never_become_false_or_claimed_shortcircuit(self):
        def original():return True
        with PassiveExecutionDiagnostic((ProfileSite.from_callable('missing',original,kind='release_done'),)) as tap:
            self.assertIs(original(),True)
        leaves=returns(tap,'missing')[0]['release_leaf_evidence']
        self.assertTrue(all(v['status']=='unavailable' for v in leaves.values()))
        self.assertEqual(returns(tap,'missing')[0]['context']['action_index']['status'],'unavailable')
        def with_target():
            target=Pose(np.zeros(3))
            return False
        with PassiveExecutionDiagnostic((ProfileSite.from_callable('no_state',with_target,kind='release_done'),)) as tap:
            self.assertIs(with_target(),False)
        leaves=returns(tap,'no_state')[0]['release_leaf_evidence']
        self.assertIs(leaves['target_present'],True)
        for key in ('gripper_open','not_contact','distance_m','distance_le_3cm'):
            self.assertEqual(leaves[key]['status'],'unavailable')

    def test_same_original_exception_object_is_transparent_and_none_is_ambiguous(self):
        error=RuntimeError('original error');calls=[]
        def original():
            calls.append(1)
            raise error
        tap=PassiveExecutionDiagnostic((ProfileSite.from_callable('bad',original,kind='completion'),))
        try:
            with tap:original()
        except RuntimeError as actual:self.assertIs(actual,error)
        else:self.fail('original exception lost')
        self.assertEqual(calls,[1]);self.assertTrue(tap.restored)
        self.assertEqual(tap.scope_exception_type,'RuntimeError')
        e=returns(tap,'bad')[0]
        self.assertEqual(e['return_status'],'none_return_or_python_exception_unwind_unknown')
        self.assertEqual(e['original_return']['status'],'unavailable')
        self.assertNotIn('original_exception_type',e)

    def test_normal_none_and_caught_python_exception_have_same_profile_boundary(self):
        def normal():return None
        def caught():raise ValueError('caught by original caller')
        sites=(ProfileSite.from_callable('normal',normal,kind='completion'),
               ProfileSite.from_callable('caught',caught,kind='completion'))
        with PassiveExecutionDiagnostic(sites) as tap:
            self.assertIsNone(normal())
            try:caught()
            except ValueError:pass
        self.assertEqual(tap.ambiguous_none_returns,2)
        self.assertIsNone(tap.scope_exception_type);self.assertEqual(tap.recording_errors,0)
        self.assertEqual(returns(tap,'normal')[0]['return_status'],returns(tap,'caught')[0]['return_status'])

    def test_existing_profile_is_refused_and_exact_old_object_is_preserved(self):
        def original():return 1
        def old(frame,event,arg):pass
        before=sys.getprofile();self.assertIsNone(before)
        sys.setprofile(old)
        try:
            tap=PassiveExecutionDiagnostic((ProfileSite.from_callable('f',original,kind='completion'),))
            with self.assertRaises(RuntimeError):tap.__enter__()
            self.assertIs(sys.getprofile(),old)
        finally:sys.setprofile(before)

    def test_foreign_profile_installed_mid_scope_is_not_erased(self):
        def original():return 1
        def foreign(frame,event,arg):pass
        tap=PassiveExecutionDiagnostic((ProfileSite.from_callable('f',original,kind='completion'),))
        try:
            with tap:
                original();sys.setprofile(foreign)
            self.assertIs(sys.getprofile(),foreign)
            self.assertFalse(tap.restored);self.assertFalse(tap.receipt()['diagnostic_available'])
        finally:sys.setprofile(None)

    def test_event_and_byte_limits_mark_diagnostic_unavailable_without_changing_calls(self):
        calls=[]
        def original():calls.append(1);return True
        for limits in ({'max_events':1},{'max_serialized_bytes':1}):
            with self.subTest(limits=limits):
                calls.clear()
                with PassiveExecutionDiagnostic((ProfileSite.from_callable('f',original,kind='completion'),),**limits) as tap:
                    self.assertIs(original(),True);self.assertIs(original(),True)
                self.assertEqual(len(calls),2);self.assertGreater(tap.dropped_events,0)
                self.assertLessEqual(len(tap.events),tap.max_events)
                self.assertLessEqual(tap.serialized_bytes,tap.max_serialized_bytes)
                self.assertFalse(tap.receipt()['diagnostic_available']);self.assertEqual(tap._active,{})

    def test_active_frame_and_value_limits_do_not_affect_original_recursion(self):
        def original(n):return original(n-1)+1 if n else 0
        site=ProfileSite.from_callable('f',original,kind='completion')
        with PassiveExecutionDiagnostic((site,),max_active_frames=1) as tap:self.assertEqual(original(4),4)
        self.assertGreater(tap.active_frame_overflows,0);self.assertFalse(tap.receipt()['diagnostic_available'])
        def values():return [1,2,3]
        with PassiveExecutionDiagnostic((ProfileSite.from_callable('values',values,kind='completion'),),
                                        max_values_per_event=1) as tap:self.assertEqual(values(),[1,2,3])
        self.assertGreater(tap.value_truncations,0);self.assertFalse(tap.receipt()['diagnostic_available'])

    def test_owner_filter_and_no_property_or_getter_reads(self):
        hits=[]
        class Opaque:
            @property
            def __dict__(self):hits.append('__dict__');raise AssertionError('descriptor invoked')
            def original(self):return True
            def get_pose(self):hits.append('get_pose');raise AssertionError('Native getter invoked')
        owned,other=Opaque(),Opaque()
        site=ProfileSite.from_callable('owned',owned.original,kind='ready')
        with PassiveExecutionDiagnostic((site,)) as tap:
            self.assertTrue(owned.original());self.assertTrue(other.original())
        self.assertEqual(hits,[]);self.assertEqual(tap.counts['owned']['calls'],1)

    def test_post_step_helper_observes_updated_target_then_repeated_grip_target(self):
        calls=[]
        class PathFixture:
            def __init__(self):
                self._path_done=False;self._joint_position_action=None;self._rml_handle=17;self._num_joints=7
            def step(self):
                calls.append('step');self._joint_position_action=np.arange(7.)+3;self._path_done=True
                return True
            def get_joint_positions(self):raise AssertionError('new getter')
        def helper(*args,**kwargs):
            calls.append(('command',args[1].copy(),kwargs['opening']))
            return None
        class Terminal:
            def execute(self,env,action,path):
                done=False;last_target=np.zeros(7);ticks=2;executed=0;error=arm_error=None
                for tick in range(ticks):
                    if tick<ticks-1:
                        done=path.step()
                        last_target=np.asarray(path._joint_position_action)
                    helper(env,last_target,opening=float(tick==ticks-1),phase='fixture')
                    executed+=1
                return action
        terminal=Terminal();path=PathFixture();action=np.arange(8.)
        sampling=ActionOutcomeSampling();sampling.initialize(NONTERMINAL)
        sampling.begin_action(tick=0,label='geometry_execute',max_ticks=10)
        env=SimpleNamespace(ledger=SimpleNamespace(ticks=0,dt=.05),_requested_opening=0.,
                            _native_action_sampling=sampling)
        sites=(ProfileSite.from_callable('execute',terminal.execute,kind='terminal'),
               ProfileSite.from_callable('step',path.step,kind='path_step'),
               ProfileSite.from_callable('command',helper,kind='command'))
        with PassiveExecutionDiagnostic(sites) as tap:result=terminal.execute(env,action,path)
        self.assertIs(result,action);self.assertEqual(len(calls),3);self.assertEqual(calls[0],'step')
        rows=[e for e in tap.events if e['site']=='command' and e['event']=='original_python_call']
        for tick,row in enumerate(rows):
            saved=row['snapshot']['terminal_after_step_assignment_before_dispatch']
            self.assertIs(saved['done'],True);self.assertEqual(saved['tick'],tick)
            self.assertEqual(saved['last_target'],list(np.arange(7.)+3))
            self.assertEqual(row['snapshot']['forwarded_joints'],saved['last_target'])
            self.assertEqual(row['context']['ledger_ticks'],0)
            self.assertEqual(row['context']['action_index'],0)
            self.assertEqual(row['context']['action_label'],'geometry_execute')
            self.assertEqual(row['context']['action_start_tick'],0)
        self.assertTrue(returns(tap,'step')[0]['original_return'])
        path._joint_position_action[:]=999
        self.assertEqual(rows[0]['snapshot']['forwarded_joints'],list(np.arange(7.)+3))
        self.assertEqual(len([e for e in tap.events if e['site']=='command']),4)

    def test_apply_pre_q_post_q_and_original_return_identity(self):
        counts=[]
        def apply(env,joints,*,opening,phase):
            current=env.actual.copy();requested=np.asarray(joints);bounded=requested.copy()
            env.actual=bounded.copy();counts.append('apply');return bounded
        def outer(current_env,joints,*,opening,phase):
            before=current_env.actual.copy();requested=np.asarray(joints)
            applied=apply(current_env,joints,opening=opening,phase=phase)
            after=current_env.actual.copy();native=np.ones(7);counts.append('outer')
        env=SimpleNamespace(actual=np.zeros(7));joints=np.arange(7.)
        sites=(ProfileSite.from_callable('apply',apply,kind='apply'),
               ProfileSite.from_callable('outer',outer,kind='apply_outer'))
        with PassiveExecutionDiagnostic(sites) as tap:self.assertIsNone(outer(env,joints,opening=0.,phase='x'))
        self.assertEqual(counts,['apply','outer'])
        self.assertEqual(returns(tap,'apply')[0]['snapshot']['current'],[0.]*7)
        outer_event=returns(tap,'outer')[0]
        self.assertEqual(outer_event['snapshot']['after'],list(joints))
        self.assertEqual(outer_event['snapshot']['native'],[1.]*7)
        self.assertEqual(outer_event['return_status'],'none_return_or_python_exception_unwind_unknown')
        self.assertEqual(tap.receipt()['extra_native_getter_observe_predict_predicate_calls'],0)

    def test_real_ready_shortcircuit_never_calls_port_to_fill_missing_value(self):
        from our_mani.multitask.physmani_geometry_policy_v518 import PublicModelFrontend
        calls=[]
        class Port:
            def readiness(self,state,forecast,phase):calls.append(phase);return {'ready':True}
        frontend=PublicModelFrontend.__new__(PublicModelFrontend)
        frontend.state=public_state();frontend.forecast=None;frontend.role_ready=True
        frontend.actor=SimpleNamespace(collision_port=Port())
        sites=(ProfileSite.from_callable('ready',frontend.ready,kind='ready'),
               ProfileSite.from_callable('port',frontend.actor.collision_port.readiness,kind='port'))
        with PassiveExecutionDiagnostic(sites) as tap:self.assertIs(frontend.ready('release'),False)
        self.assertEqual(calls,[]);self.assertEqual(tap.counts['port']['calls'],0)
        self.assertEqual(returns(tap,'ready')[0]['direct_ready_port_return']['status'],'unavailable')

    def test_real_ready_consumes_original_port_return_once_with_exact_reason(self):
        from our_mani.multitask.physmani_geometry_policy_v518 import PublicModelFrontend
        calls=[];port_result={'ready':False,'reason':'existing original unavailable forecast'}
        class Port:
            def readiness(self,state,forecast,phase):calls.append(phase);return port_result
        frontend=PublicModelFrontend.__new__(PublicModelFrontend)
        frontend.state=public_state();frontend.forecast=object();frontend.role_ready=True
        frontend.actor=SimpleNamespace(collision_port=Port(),last_diagnostics={
            'invalid_mode_ids':[],'mode_estimates':{'ring':'released'}},body_ids=('ring',))
        sites=(ProfileSite.from_callable('ready',frontend.ready,kind='ready'),
               ProfileSite.from_callable('port',frontend.actor.collision_port.readiness,kind='port'))
        with PassiveExecutionDiagnostic(sites) as tap:self.assertIs(frontend.ready('release'),False)
        self.assertEqual(calls,['release'])
        saved=returns(tap,'ready')[0]['direct_ready_port_return']
        self.assertEqual(saved,port_result);self.assertIsNot(saved,port_result)
        self.assertEqual(returns(tap,'port')[0]['parent_call_id'],returns(tap,'ready')[0]['call_id'])

    def test_frozen_pyrep_method_body_with_CPU_API_stubs_has_no_extra_calls(self):
        root=Path(__file__).resolve().parents[1]
        path=root.parent/'phsimani/PhysMani/third_party/PyRep/pyrep/robots/configuration_paths/arm_configuration_path.py'
        if not path.exists():self.skipTest('local frozen author source is not installed')
        tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)
                                             and n.name=='ArmConfigurationPath')
        nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('step','_step_motion')]
        calls=[]
        class SimCPUStub:
            def simGetSimulationTimeStep(self):calls.append('dt');return .05
            def simRMLStep(self,handle,dt,count):calls.append('RMLstep');return 1,[1.,0.,0.]
            def simRMLRemove(self,handle):calls.append('remove')
        namespace={'sim':SimCPUStub()}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
        class ArmCPUStub:
            joints=tuple(range(7));_num_joints=7
            def set_joint_target_positions(self,q):calls.append('set_target')
        class PathCPUFixture:
            def __init__(self):
                self._path_done=False;self._joint_position_action=None;self._rml_handle=17
                self._path_points=np.r_[np.zeros(7),np.ones(7)];self._num_joints=7;self._arm=ArmCPUStub()
            def _get_path_point_lengths(self):calls.append('lengths');return [0.,1.]
        fixture=PathCPUFixture()
        fixture.step=MethodType(namespace['step'],fixture)
        fixture._step_motion=MethodType(namespace['_step_motion'],fixture)
        sites=(ProfileSite.from_callable('step',fixture.step,kind='path_step'),
               ProfileSite.from_callable('motion',fixture._step_motion,kind='path_motion'))
        with PassiveExecutionDiagnostic(sites) as tap:self.assertIs(fixture.step(),True)
        self.assertEqual(calls,['dt','lengths','RMLstep','set_target','remove'])
        motion=returns(tap,'motion')[0]
        self.assertEqual(motion['original_return'],1)
        self.assertEqual(motion['snapshot']['qs'],[1.]*7)
        self.assertEqual(motion['snapshot']['dt'],.05)
        self.assertTrue(returns(tap,'step')[0]['snapshot']['path']['_path_done'])
        self.assertEqual(tap.recording_errors,0)

    def test_actual_forecast_at_return_is_copied_without_a_second_query(self):
        f=Forecast(('peg0',),np.array([0.,.05]),np.zeros((2,1,3)),
                   np.broadcast_to(np.eye(3),(2,1,3,3)).copy(),np.zeros((2,1,3)),np.zeros((2,1,3)))
        site=ProfileSite.from_callable('forecast',f.at,kind='forecast_at')
        with PassiveExecutionDiagnostic((site,)) as tap:result=f.at('peg0',.025)
        self.assertIs(type(result[0]),Pose);self.assertEqual(tap.counts['forecast']['calls'],1)
        self.assertEqual(returns(tap,'forecast')[0]['original_return'][0]['position'],[0.,0.,0.])

    def test_large_original_FK_arrays_copy_only_existing_endpoints_without_extra_FK(self):
        calls=[]
        def extract():
            calls.append(1);poses=np.arange(343.,dtype=float).reshape(49,7);joints=poses+1000
            return poses,joints
        def modify():
            poses,joints=extract();selected=[10,15,20,25,30,35,40,45]
            return {'selected_extracted_indices':selected}
        sites=(ProfileSite.from_callable('extract',extract,kind='extract'),
               ProfileSite.from_callable('filter',modify,kind='filter'))
        with PassiveExecutionDiagnostic(sites) as tap:result=modify()
        self.assertEqual(calls,[1]);self.assertEqual(result['selected_extracted_indices'][-1],45)
        row=returns(tap,'filter')[0]['snapshot']['poses_existing_endpoints']
        self.assertEqual(row['last_selected_row'],list(np.arange(315.,322.)))
        self.assertEqual(row['last'],list(np.arange(336.,343.)))
        self.assertEqual(tap.value_truncations,0);self.assertTrue(tap.receipt()['diagnostic_available'])

    def test_default_source_declarations_are_exact_and_include529_and_real_pyrep(self):
        root=Path(__file__).resolve().parents[1]
        pyrep=root.parent/'phsimani/PhysMani/third_party/PyRep/pyrep/robots/configuration_paths/arm_configuration_path.py'
        if not pyrep.exists():self.skipTest('local frozen author source is not installed')
        sites=frozen_source_sites(root,pyrep_path_file=pyrep)
        labels={s.label:s for s in sites}
        self.assertEqual(len(sites),19)
        self.assertEqual(labels['completion.current529'].firstlineno,57)
        self.assertEqual(labels['release.done'].firstlineno,69)
        self.assertEqual(labels['path.step'].firstlineno,38)
        self.assertEqual(labels['path.motion'].firstlineno,177)
        for site in sites:self.assertEqual(hashlib.sha256(Path(site.filename).read_bytes()).hexdigest(),site.source_sha256)
        # Merely read/compile declarations; no profiler or PyRep app is created.
        self.assertIsNone(sys.getprofile())

    def test_source_pin_mismatch_refuses_installation(self):
        def original():return True
        s=ProfileSite.from_callable('f',original,kind='completion')
        pinned=ProfileSite(s.label,s.filename,s.qualname,s.firstlineno,s.kind,'0'*64)
        with self.assertRaises(ValueError):PassiveExecutionDiagnostic((pinned,)).__enter__()
        self.assertIsNone(sys.getprofile())


if __name__=='__main__':unittest.main()
