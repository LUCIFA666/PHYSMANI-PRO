"""CPU fakes and pinned-source compilation only; no native/preflight/main/data."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
from our_mani.multitask import train_remove_capture_ast_v892 as subject


def errors(robot,target):
    robot.calls.append('errors');return robot.distance,robot.angle


class Robot:
    def __init__(self):self.calls=[];self.distance=.01;self.angle=.05;self.now=1.;self.touch=True
    @property
    def timestamp(self):self.calls.append('timestamp');return self.now
    @property
    def contact(self):self.calls.append('contact');return self.touch


class Template:
    def __init__(self,calls):self.calls=calls;self.duration=.5
    @property
    def duration_seconds(self):self.calls.append('duration');return self.duration


class Base:
    def __init__(self):self.robot=Robot();self.phase_epoch=0.;self.target=object();self.capture=True
    def _completion_target(self,goal):self.robot.calls.append('target');return self.target
    def _public_capture(self,goal):
        if not self.robot.contact or goal.get('witness') is None:return False
        return self.capture
    def _done(self,goal):
        target=self._completion_target(goal)
        if target is None:return False
        distance,angle=errors(self.robot,target)
        if self.robot.timestamp-self.phase_epoch<goal['template'].duration_seconds-1e-8:return False
        return self._public_capture(goal) and distance<=.02 and angle<=.1
    def terminal(self,value):self.robot.calls.append('base');return value
    def explode(self,error):raise error
    def predict_once(self,predict):
        target=predict()
        return target
    def truth(self,value):
        if value and value:return value
        return None
    def defaulted(self,value=object()):
        return value
    def nested(self,a,b,c):
        if (a or b) and c:return self.target
        return None
    def not_compound(self,a,b):
        if not (a and b):return self.target
        return None
    def chain(self,a,b,c):
        if a<b<c:return self.target
        return None
    def loop(self,value):
        while value:return self.target
        return None


class Middle(Base):
    def terminal(self,value):
        self.robot.calls.append('middle')
        return super().terminal(value)


class Leaf(Middle):
    def terminal(self,value):
        self.robot.calls.append('leaf')
        return super().terminal(value)


class Scope:
    """Private installer exercise only; never an authentic native capability."""
    def __init__(self,target,entries):self.target=target;self.entries=entries
    def __enter__(self):
        import threading
        self.trace=subject.TrainRemoveCaptureValueObserver(self.target,enabled=True)
        self.trace._thread=threading.get_ident();self.trace._entered=True
        path=Path(__file__);checksum=hashlib.sha256(path.read_bytes()).hexdigest()
        try:self.trace._install([(self.target,owner,name,path,checksum) for owner,name in self.entries])
        except BaseException:self.trace._restore();raise
        return self.trace
    def __exit__(self,*exc):return self.trace.__exit__(*exc)


def goal(policy):return dict(template=Template(policy.robot.calls),witness=True)


class ValueTapTests(unittest.TestCase):
    def test_disabled_entry_is_inert(self):
        class Untouchable:
            def __getattribute__(self,name):raise AssertionError('disabled observer touched object')
        with subject.TrainRemoveCaptureValueObserver(Untouchable()) as trace:
            self.assertFalse(trace.bind_context(tick=0))
        self.assertTrue(trace.receipt()['restored'])

    def test_fake_domain_cannot_activate(self):
        p=SimpleNamespace(task=subject.TASKS[0])
        fake=SimpleNamespace(split='train',method='cv_geometry',require=lambda *a:'train_qualification')
        with self.assertRaisesRegex(ValueError,'authentic'):
            with subject.TrainRemoveCaptureValueObserver(p,domain=fake,enabled=True,development_only=True):pass

    def test_done_same_result_getter_geometry_and_capture_call_counts(self):
        original=Base();expected=original._done(goal(original));calls=original.robot.calls[:]
        observed=Base()
        with Scope(observed,[(Base,'_done'),(Base,'_public_capture')]) as trace:
            actual=observed._done(goal(observed))
        self.assertIs(actual,expected);self.assertEqual(observed.robot.calls,calls)
        self.assertEqual(observed.robot.calls.count('errors'),1)
        self.assertEqual(observed.robot.calls.count('duration'),1)
        self.assertEqual(observed.robot.calls.count('contact'),1)
        done=next(c for c in trace.calls if c['source']['method']=='_done')
        durations=[v for label,values in done['evaluations'].items() for v in values if done['expressions'][label]['kind']=='duration_arithmetic']
        self.assertTrue(any(v.get('value')==1. for v in durations))

    def test_short_circuit_keeps_later_original_expressions_unknown(self):
        p=Base();p.robot.touch=False;p.robot.distance=.03
        with Scope(p,[(Base,'_done'),(Base,'_public_capture')]) as trace:self.assertFalse(p._done(goal(p)))
        event=next(c for c in trace.calls if c['source']['method']=='_done')
        unvisited=[event['expressions'][key]['expression'] for key in event['not_evaluated']]
        self.assertTrue(any('distance' in text and '.02' in text for text in unvisited))
        self.assertTrue(all(v['status']=='unknown_not_evaluated_on_original_path' for v in event['not_evaluated'].values()))

    def test_duration_short_circuit_does_not_call_capture(self):
        p=Base();p.robot.now=.1
        before=Base();before.robot.now=.1;expected=before._done(goal(before));expected_calls=before.robot.calls[:]
        with Scope(p,[(Base,'_done'),(Base,'_public_capture')]) as trace:self.assertIs(p._done(goal(p)),expected)
        self.assertEqual(p.robot.calls,expected_calls);self.assertFalse(any(c['source']['method']=='_public_capture' for c in trace.calls))

    def test_zero_argument_super_preserves_original_MRO_and_return_identity(self):
        value=object();p=Leaf();original=Base.__dict__['terminal'];middle=Middle.__dict__['terminal'];leaf=Leaf.__dict__['terminal']
        with Scope(p,[(Base,'terminal'),(Middle,'terminal'),(Leaf,'terminal')]) as trace:
            self.assertIs(p.terminal(value),value)
            self.assertEqual(p.robot.calls,['leaf','middle','base'])
        self.assertIs(Base.__dict__['terminal'],original);self.assertIs(Middle.__dict__['terminal'],middle);self.assertIs(Leaf.__dict__['terminal'],leaf)
        self.assertEqual(len(trace.calls),3)
        self.assertEqual(len({call['call_id'] for call in trace.calls}),3)

    def test_public_pose_only_and_unknown_scalar_identity_are_not_serialized(self):
        from our_mani.contracts import Pose
        pose=Pose([.1,.2,.3]);p=Base()
        with Scope(p,[(Base,'predict_once')]) as trace:self.assertIs(p.predict_once(lambda:pose),pose)
        self.assertEqual(trace.calls[0]['return_value']['status'],'recorded_public_pose')
        self.assertEqual(trace.calls[0]['return_value']['position'],[.1,.2,.3])
        self.assertEqual(trace._encode(12345,'assigned_value')['status'],'unknown_value_not_on_public_allowlist')

    def test_default_object_identity_not_re_evaluated(self):
        p=Base();saved=p.defaulted()
        with Scope(p,[(Base,'defaulted')]):self.assertIs(p.defaulted(),saved)

    def test_context_rejects_arbitrary_object(self):
        p=Base()
        with Scope(p,[(Base,'terminal')]) as trace:
            with self.assertRaises(ValueError):trace.bind_context(tick=0,phase=object())
            with self.assertRaises(ValueError):trace.bind_context(tick=0,action_id=object())
            self.assertTrue(trace.bind_context(tick=2,action_id=0,phase='acquire'))

    def test_recorder_failure_is_unknown_evaluated_not_unknown_unvisited(self):
        p=Base();value=object()
        with Scope(p,[(Base,'predict_once')]) as trace:
            def bad_encode(*args):raise ValueError('fake encoder failure')
            trace._encode=bad_encode
            self.assertIs(p.predict_once(lambda:value),value)
        event=trace.calls[0]
        self.assertTrue(event['evaluations'])
        self.assertTrue(all(v['status']=='unknown_recording_failed' for values in event['evaluations'].values() for v in values))
        self.assertEqual(event['return_value']['status'],'unknown_recording_failed')
        self.assertFalse(set(event['evaluations']) & set(event['not_evaluated']))

    def test_original_exception_object_propagates_and_scope_restores(self):
        p=Base();error=RuntimeError('original');saved=Base.__dict__['explode']
        with self.assertRaises(RuntimeError) as caught:
            with Scope(p,[(Base,'explode')]) as trace:p.explode(error)
        self.assertIs(caught.exception,error);self.assertIs(Base.__dict__['explode'],saved)
        self.assertNotIn(subject._HELPER,globals());self.assertTrue(trace.receipt()['restored'])

    def test_predictor_is_called_once_and_unknown_return_is_not_inspected(self):
        class Hidden:
            def __getattribute__(self,name):raise AssertionError('hidden object was inspected')
        value=Hidden();calls=[];p=Base()
        def predict():calls.append(1);return value
        with Scope(p,[(Base,'predict_once')]) as trace:self.assertIs(p.predict_once(predict),value)
        self.assertEqual(calls,[1]);self.assertEqual(trace.calls[0]['return_value']['status'],'unknown_value_not_on_public_allowlist')

    def test_custom_truth_getter_is_not_evaluated_for_logging(self):
        class Truth:
            def __init__(self):self.calls=0
            def __bool__(self):self.calls+=1;return True
        a=Truth();p=Base();self.assertIs(p.truth(a),a);expected=a.calls
        b=Truth()
        with Scope(p,[(Base,'truth')]):self.assertIs(p.truth(b),b)
        self.assertEqual(b.calls,expected)

    def test_false_and_changing_truth_keep_exact_calls_and_return_identity(self):
        class Truth:
            def __init__(self,sequence,events,name):self.sequence=sequence;self.events=events;self.name=name;self.index=0
            def __bool__(self):
                self.events.append(self.name);value=self.sequence[min(self.index,len(self.sequence)-1)];self.index+=1
                if isinstance(value,BaseException):raise value
                return value
        for name,patterns in (
            ('truth',([False,True],)),('truth',([True,False],)),
            ('nested',([False,True],[True,False],[False,True])),
            ('nested',([True,False],[False,True],[True,False])),
            ('not_compound',([False,True],[True,False])),
            ('not_compound',([True,False],[False,True])),('loop',([False,True],))):
            with self.subTest(method=name,patterns=patterns):
                signal=object();plain=Base();plain.target=signal;before=[]
                operands=[Truth(seq,before,str(i)) for i,seq in enumerate(patterns)]
                expected=getattr(plain,name)(*operands)
                tapped=Base();tapped.target=signal;after=[]
                others=[Truth(seq,after,str(i)) for i,seq in enumerate(patterns)]
                with Scope(tapped,[(Base,name)]) as trace:actual=getattr(tapped,name)(*others)
                # truth() returns its original operand; all other return values
                # are a shared public sentinel or None.
                if expected is operands[0]:self.assertIs(actual,others[0])
                else:self.assertIs(actual,expected)
                self.assertEqual(after,before)
                self.assertTrue(trace.calls[0]['unobserved_compound_conditions'])

    def test_chained_comparison_truth_and_exception_calls_are_unchanged(self):
        class Truth:
            def __init__(self,answer,events):self.answer=answer;self.events=events
            def __bool__(self):
                self.events.append('bool')
                if isinstance(self.answer,BaseException):raise self.answer
                return self.answer
        class Item:
            def __init__(self,index,answers,events):self.index=index;self.answers=answers;self.events=events
            def __lt__(self,other):
                self.events.append(('compare',self.index,other.index));return Truth(self.answers[self.index],self.events)
        problem=RuntimeError('original comparison truth')
        for answers in ((False,True),(True,False),(True,True),(problem,True),(True,problem)):
            with self.subTest(answers=answers):
                before=[];after=[];signal=object();plain=Base();plain.target=signal;tapped=Base();tapped.target=signal
                def invoke(policy,events):
                    try:return ('return',policy.chain(*(Item(i,answers,events) for i in range(3))))
                    except BaseException as error:return ('exception',error)
                expected=invoke(plain,before)
                with Scope(tapped,[(Base,'chain')]) as trace:actual=invoke(tapped,after)
                self.assertEqual(actual[0],expected[0]);self.assertIs(actual[1],expected[1]);self.assertEqual(after,before)
                self.assertTrue(trace.calls[0]['unobserved_compound_conditions'])

    def test_other_instance_is_forwarded_without_observer_events(self):
        p=Base();other=Base();value=object()
        with Scope(p,[(Base,'terminal')]) as trace:self.assertIs(other.terminal(value),value)
        self.assertEqual(other.robot.calls,['base']);self.assertEqual(trace.calls,[])

    def test_source_sha_mismatch_and_overlap_refused(self):
        with self.assertRaisesRegex(ValueError,'source SHA'):
            subject.compile_tapped(Base,'_done',source_path=__file__,expected_sha='0'*64)
        p=Base()
        with Scope(p,[(Base,'terminal')]):
            with self.assertRaises(RuntimeError):
                with Scope(p,[(Base,'terminal')]):pass

    def test_all_pinned_actual_method_bodies_compile_with_strip_parity(self):
        #Compile AST class fragments only: no policy/native imports or constructor.
        count=0
        for basename,checksum in subject.PINS.items():
            path=ROOT/'our_mani/multitask'/basename;raw=path.read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),checksum)
            tree=ast.parse(raw)
            for cls in [n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in subject.METHODS]:
                methods=[deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in subject.METHODS[cls.name]]
                if not methods:continue
                synthetic=ast.ClassDef(name=cls.name,bases=[],keywords=[],body=methods,decorator_list=[])
                namespace={'__name__':'CPU_pinned_source_fragment'}
                exec(compile(ast.fix_missing_locations(ast.Module(body=[synthetic],type_ignores=[])),str(path),'exec'),namespace)
                owner=namespace[cls.name]
                for method in methods:
                    _,labels,receipt=subject.compile_tapped(owner,method.name,source_path=path,expected_sha=checksum)
                    self.assertTrue(receipt['stripped_AST_equal']);self.assertTrue(receipt['original_MRO_super_closure_preserved']);count+=1
        self.assertEqual(count,17)

    def test_actual_548_public_capture_short_circuit_matrix(self):
        # Execute only the pinned method AST with public CPU fake state. No
        # policy imports, constructors, simulator or stored episode is opened.
        path=ROOT/'our_mani/multitask/physmani_skill_policy_v548.py'
        cls=next(n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.ClassDef) and n.name=='SkillGeometryIntents')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_public_capture')
        tree=ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name=cls.name,bases=[],keywords=[],body=[deepcopy(method)],decorator_list=[])],type_ignores=[]))
        namespace={'__name__':'CPU_actual_capture_only'};exec(compile(tree,str(path),'exec'),namespace)
        owner=namespace[cls.name]
        def public_fixture(kind):
            robot=Robot();w=dict(phase_index=0,manipulated_ids=['public-cup'],confirmed_after=.5)
            metadata=dict(bodies={'public-cup':dict(motion_mode='attached',mode_provenance='caller_controller_touch_action_proprioception')})
            state=SimpleNamespace(robot=robot,perception_metadata=metadata,role_bindings={'cup':('public-cup',)})
            policy=owner();policy.frontend=SimpleNamespace(state=state,actor=SimpleNamespace(controller=SimpleNamespace(measured_attachments={'public-cup':object()})))
            policy.phase_index=0;policy.close_witness=w;g=dict(manipulated_ids=('public-cup',),template=SimpleNamespace(manipulated_role='cup'))
            if kind=='no-contact':robot.touch=False
            if kind=='no-witness':policy.close_witness=None
            if kind=='wrong-phase':w['phase_index']=1
            if kind=='future-witness':w['confirmed_after']=2.
            if kind=='unknown-association':metadata['bodies'].clear()
            if kind=='missing-transform':policy.frontend.actor.controller.measured_attachments.clear()
            return policy,g,robot
        for kind in ('valid','no-contact','no-witness','wrong-phase','future-witness','unknown-association','missing-transform'):
            with self.subTest(kind=kind):
                plain,g,robot=public_fixture(kind);expected=plain._public_capture(g);calls=robot.calls[:]
                tapped,g,robot=public_fixture(kind)
                import threading
                trace=subject.TrainRemoveCaptureValueObserver(tapped,enabled=True);trace._thread=threading.get_ident();trace._entered=True
                try:
                    trace._install([(tapped,owner,'_public_capture',path,subject.PINS[path.name])])
                    self.assertIs(tapped._public_capture(g),expected)
                    self.assertEqual(robot.calls,calls)
                    if kind=='no-contact':self.assertTrue(trace.calls[0]['not_evaluated'])
                finally:trace._restore()


if __name__=='__main__':unittest.main()
