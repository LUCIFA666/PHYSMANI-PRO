"""Pure mocked-native probes; no simulator, robot writes or physical ticks."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

from our_mani.multitask.rml_path_repair import (NativeRMLHandleFactory,create_repaired_rml_handle,
    RMLVirtualProbeExhausted,RMLProbeBackendFailure,RMLProbeCleanupFailure)


class Arm:
    max_velocity=3.;max_acceleration=6.;max_jerk=12.
    def get_joint_upper_velocity_limits(self):return [1.]
    def __getattr__(self,name):
        if name.startswith('set'):raise AssertionError('virtual probe attempted arm setter '+name)
        raise AttributeError(name)


def path():
    return SimpleNamespace(_arm=Arm(),_num_joints=1,_path_points=np.array([0.,1.]),
        _get_path_point_lengths=lambda:[0.,1.])


class Backend:
    def __init__(self):
        self.handles={};self.allocations=[];self.removed=[];self.step_failure=None;self.remove_failure=False
    def simGetSimulationTimeStep(self):return .1
    def simRMLPos(self,dofs,smallest,flags,current,limits,selection,target):
        handle=len(self.allocations)+1
        self.allocations.append(dict(handle=handle,dofs=dofs,smallest=smallest,flags=flags,current=current,
            limits=list(limits),selection=selection,target=target))
        self.handles[handle]=dict(position=0.,velocity=limits[0],target=target[0],steps=0);return handle
    def simRMLStep(self,handle,dt,dofs):
        if self.step_failure is not None:raise self.step_failure
        h=self.handles[handle];h['steps']+=1;h['position']=min(h['target'],h['position']+h['velocity']*dt)
        return int(h['position']>=h['target']),[h['position'],h['velocity'],0.]
    def simRMLRemove(self,handle):
        if self.remove_failure:raise ValueError('mock native delete failure')
        self.removed.append(handle);del self.handles[handle]
    def step(self):raise AssertionError('virtual probe called physics step')
    def simStep(self):raise AssertionError('virtual probe called physics step')


class RMLProbeRepairChecks(unittest.TestCase):
    def test_old_cross_probe_previous_q_creates_false_end_to_start_spike(self):
        backend=Backend();previous=0.;correction=1.;ratios=[]
        for _ in range(2):
            handle=backend.simRMLPos(1,.0001,-1,[0,0,0],[3.*correction,6.,12.],[1],[1.,0])
            state=0;rmax=0.
            while state==0:
                state,pva=backend.simRMLStep(handle,.1,1);q=pva[0]
                rmax=max(rmax,abs(q-previous)/.1);previous=q
            backend.simRMLRemove(handle);ratios.append(rmax);correction/=rmax
        self.assertAlmostEqual(ratios[0],3.)
        self.assertAlmostEqual(ratios[1],9.) # Genuine second-probe per-step ratio is <=1.
        factory=NativeRMLHandleFactory(sim_backend=Backend());factory(path());receipt=factory.last_receipt
        self.assertEqual(len(receipt['probes']),2)
        self.assertAlmostEqual(receipt['probes'][0]['maximum_joint_velocity_ratio'],3.)
        self.assertLessEqual(receipt['probes'][1]['maximum_joint_velocity_ratio'],1.001)
        self.assertEqual([p['previous_q_initial'] for p in receipt['probes']],[[0.],[0.]])

    def test_returned_unstepped_execution_handle_uses_correct_joint_velocity_and_unchanged_parameters(self):
        backend=Backend();receipt={};p=path();original=p._path_points.copy()
        handle=create_repaired_rml_handle(p,sim_backend=backend,receipt=receipt)
        self.assertEqual(backend.handles[handle]['steps'],0)
        self.assertEqual(set(backend.handles),{handle});self.assertEqual(backend.removed,[1,2])
        for request in backend.allocations:
            self.assertEqual((request['dofs'],request['smallest'],request['flags']),(1,.0001,-1))
            self.assertEqual(request['limits'][1:],[6.,12.]);self.assertEqual(request['selection'],[1])
        previous=0.;state=0
        while state==0:
            state,pva=backend.simRMLStep(handle,.1,1)
            self.assertLessEqual(abs(pva[0]-previous)/.1,1.001);previous=pva[0]
        np.testing.assert_array_equal(p._path_points,original)
        self.assertFalse(receipt['physics_step_called']);self.assertFalse(receipt['arm_joint_or_target_setter_called'])

    def test_virtual_budget_failure_deletes_handle_and_does_not_transfer_execution(self):
        backend=Backend();factory=NativeRMLHandleFactory(sim_backend=backend,maximum_virtual_steps=1)
        with self.assertRaises(RMLVirtualProbeExhausted):factory(path())
        self.assertEqual(backend.handles,{});self.assertEqual(backend.removed,[1])
        self.assertFalse(factory.last_receipt['complete']);self.assertFalse(factory.last_receipt['live_execution_handle_transferred'])

    def test_native_step_exception_still_deletes_probe(self):
        backend=Backend();backend.step_failure=ValueError('mock native step failure')
        with self.assertRaisesRegex(ValueError,'native step failure'):
            create_repaired_rml_handle(path(),sim_backend=backend)
        self.assertEqual(backend.handles,{});self.assertEqual(backend.removed,[1])

    def test_original_endpoint_segment_condition_and_multijoint_path_are_preserved(self):
        backend=Backend();original_step=backend.simRMLStep
        def overshoot(handle,dt,dofs):
            state,pva=original_step(handle,dt,dofs)
            if state==1:pva[0]=np.nextafter(pva[0],np.inf)
            return state,pva
        backend.simRMLStep=overshoot;receipt={}
        create_repaired_rml_handle(path(),sim_backend=backend,receipt=receipt)
        self.assertTrue(receipt['complete'])
        self.assertTrue(all(len(row['original_segment_condition_unmatched_samples'])==1 for row in receipt['probes']))
        arm=Arm();arm.get_joint_upper_velocity_limits=lambda:[.5,1.]
        p=SimpleNamespace(_arm=arm,_num_joints=2,_path_points=np.array([0.,0.,1.,0.,1.,1.]),
            _get_path_point_lengths=lambda:[0.,1.,2.])
        backend=Backend();receipt={};handle=create_repaired_rml_handle(p,sim_backend=backend,receipt=receipt)
        self.assertLessEqual(receipt['probes'][-1]['maximum_joint_velocity_ratio'],1.001)
        self.assertEqual(receipt['joint_upper_velocity_limits'],[.5,1.])
        self.assertEqual(set(backend.handles),{handle})

    def test_invalid_native_state_and_cleanup_failure_are_infrastructure_not_accepted_paths(self):
        backend=Backend();backend.simRMLStep=lambda *args:(-1,[0.,0.,0.])
        with self.assertRaises(RMLProbeBackendFailure):create_repaired_rml_handle(path(),sim_backend=backend)
        self.assertEqual(backend.handles,{})
        backend=Backend();backend.remove_failure=True;receipt={}
        with self.assertRaises(RMLProbeCleanupFailure):
            create_repaired_rml_handle(path(),sim_backend=backend,receipt=receipt)
        self.assertFalse(receipt['complete']);self.assertEqual(len(receipt['cleanup_failures']),1)

    def test_factory_source_has_no_physics_steps_or_joint_target_setters(self):
        import our_mani.multitask.rml_path_repair as module
        tree=ast.parse(Path(module.__file__).read_text())
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertFalse(set(calls)&{'step','simStep','set_joint_positions','set_joint_target_positions',
            'set_joint_target_velocities','set_model_dynamic','set_simulation_timestep'})


if __name__=='__main__':unittest.main()
