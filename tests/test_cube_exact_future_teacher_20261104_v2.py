"""Pure Native-source reference simulation; no PyRep import or model forward."""
import ast
from copy import deepcopy
from pathlib import Path
import random
import tempfile
import importlib.util
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from our_mani.cube_exact_future_teacher_20261104_v2.oracle import (
    simulate_free_future,forecast_from_private_task,PrivateExactFutureExecutionBinding)
from our_mani.cube_exact_future_teacher_20261104_v2.teacher import ImmediateExpiryTeacherCandidates
from our_mani.cube_exact_future_teacher_20261104_v2.teacher import run_private_oracle_teacher
from our_mani.cube_exact_future_teacher_20261104_v2 import contracts as c,worker
from our_mani.local_mpc_control_20261103.forecast_control import BoundForecastExecutionModel
from our_mani.cube_exact_future_teacher_20261104_v2.observer import TeacherObservationRecorder
from our_mani.cube_privileged_contact_teacher_20261104_v1.observer import TeacherObservationRecorder as OriginalRecorder

NATIVE=Path('/home/ubuntu/桌面/phsimani/PhysMani/third_party/RLBench/rlbench/tasks')
PLAN=c.ROOT/'config/framework/cube_exact_future_teacher_TRAIN12_20261104_v2/plan.json'
AREA=[.05,-.45,.8,.35,.45,.8]


def original_pure_functions():
    # Compile ONLY three inspected pure function bodies. No Native imports,
    # package initialization, environment, object handle or simulator is used.
    helper=ast.parse((NATIVE/'reach_single_moving_target_on_the_table_high_speed.py').read_text())
    task=ast.parse((NATIVE/'pick_moving_target_on_the_table.py').read_text())
    cls=next(x for x in task.body if isinstance(x,ast.ClassDef))
    bodies=[x for x in helper.body if isinstance(x,ast.FunctionDef)and x.name in ('compute_target_position','cross_boundary')]
    bodies.append(next(x for x in cls.body if isinstance(x,ast.FunctionDef)and x.name=='step'))
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future,*bodies],type_ignores=[]))
    namespace={'np':np,'deepcopy':deepcopy}
    exec(compile(module,'<pure_Native_source_reference>','exec'),namespace)
    return namespace


def reference(segment,time,current,dt=.05):
    position=np.asarray(current,float).copy()
    class Target:
        def get_position(self):return position.tolist()
        def set_position(self,value):position[:]=value
    task=SimpleNamespace(target_state_list=[deepcopy(segment)],t=time,area=AREA,
        target_block=Target(),pyrep=SimpleNamespace(get_simulation_timestep=lambda:dt),
        check_grasp_success=lambda:False,_bool_expert=False,step_id=0)
    func=original_pure_functions()['step'];values=[position.copy()]
    for _ in range(16):func(task);values.append(position.copy())
    return np.asarray(values),task


class ExactTeacherContracts(unittest.TestCase):
    def test_first_tick_uses_current_task_clock_not_one_step_future(self):
        segment=dict(x=[.2,0.,.77],v=[.2,0.,0.],a=[0.,0.,0.],t0=0.)
        forecast,receipt=simulate_free_future(segment,task_time=0.,area=AREA,current_position=segment['x'],
            current_velocity=[0.,0.,0.],anchor_time=0.)
        np.testing.assert_array_equal(forecast.positions[0,0],forecast.positions[1,0])
        self.assertAlmostEqual(forecast.positions[2,0,0],.21)
        self.assertTrue(receipt['index1_evaluates_task_time_before_increment'])
        self.assertFalse(forecast.metadata['public_policy_prediction'])

    def test_full16steps_matches_original_source_reflection_and_keeps_overshoot(self):
        for segment,time,current in [
            (dict(x=[.34,.44,.77],v=[.8,.7,0.],a=[.1,-.2,0.],t0=0.),.05,[.34,.44,.77]),
            (dict(x=[.06,-.44,.77],v=[-.4,-.6,0.],a=[-.1,.2,0.],t0=0.),.05,[.06,-.44,.77]),
            (dict(x=[.35,0.,.77],v=[.2,0.,0.],a=[.3,0.,0.],t0=0.),0.,[.35,0.,.77])]:
            expected,_=reference(segment,time,current)
            forecast,receipt=simulate_free_future(segment,task_time=time,area=AREA,current_position=current,
                current_velocity=[.123,0.,0.],anchor_time=1.)
            np.testing.assert_array_equal(forecast.positions[:,0],expected)
            self.assertTrue(any(x['boundary_axes']for x in receipt['trajectory_trace']))
            self.assertTrue(all(x['clamped']is False for x in receipt['trajectory_trace']))
        self.assertGreater(reference(dict(x=[.34,0.,.77],v=[.8,0.,0.],a=[0.,0.,0.],t0=0.),.05,[.34,0.,.77])[0][1,0],AREA[3])

    def test_native_float_driver_dt_preserved_and_rng_and_inputs_unchanged(self):
        dt=float(np.float32(.05));segment=dict(x=[.3,.2,.77],v=[.7,-.6,0.],a=[.2,.3,0.],t0=0.)
        before=deepcopy(segment);np_state=np.random.get_state();python_state=random.getstate()
        expected,_=reference(segment,.25,[.32,.18,.77],dt)
        actual,receipt=simulate_free_future(segment,task_time=.25,area=AREA,current_position=[.32,.18,.77],
            current_velocity=[1.,2.,0.],anchor_time=.2,task_dt=dt)
        np.testing.assert_array_equal(actual.positions[:,0],expected)
        self.assertEqual(segment,before);self.assertEqual(random.getstate(),python_state)
        after=np.random.get_state();self.assertTrue(all(np.array_equal(a,b)for a,b in zip(np_state,after)))
        self.assertEqual(receipt['measured_task_driver_dt_seconds'],dt)

    def test_anchor_velocity_exact_and_private_binding_does_not_claim_public(self):
        segment=dict(x=[.2,0.,.77],v=[.2,0.,0.],a=[0.,0.,0.],t0=0.)
        forecast,receipt=simulate_free_future(segment,task_time=.1,area=AREA,current_position=[.21,0.,.77],
            current_velocity=[.11,.02,0.],anchor_time=.1)
        model=SimpleNamespace(initialize_batch=lambda *a:None,step_batch=lambda *a:None)
        bound=PrivateExactFutureExecutionBinding(model,forecast,anchor_time=.1,oracle_receipt=receipt)
        state=dict(time=.1,held=False,object_position=[.21,0.,.77],object_quaternion=[0.,0.,0.,1.],
            object_linear_velocity=[.11,.02,0.],object_angular_velocity=[0.,0.,0.])
        bound.validate_anchor(state)
        with self.assertRaises(ValueError):bound.validate_anchor(dict(state,object_linear_velocity=[0.,0.,0.]))
        self.assertFalse(bound.receipt()['public_bridge_contract_verified'])
        self.assertTrue(bound.receipt()['Native_schedule_inputs_used'])
        self.assertFalse(bound.receipt()['GRU_used'])
        self.assertIs(PrivateExactFutureExecutionBinding.step_batch,BoundForecastExecutionModel.step_batch)

    def test_geometry_deadline_generation_immediate_and_not_rolling(self):
        geometry=ImmediateExpiryTeacherCandidates()
        state=dict(time=0.,held=False,object_quaternion=[0.,0.,0.,1.])
        original=geometry.update(state)
        self.assertEqual(geometry.generation,0)
        self.assertEqual(geometry.update(dict(state,time=.7)),original)
        self.assertEqual(geometry.update(dict(state,time=.8)),original)
        new=geometry.update(dict(state,time=.9))
        self.assertEqual(geometry.generation,1);self.assertNotEqual(new[0]['id'],original[0]['id'])
        self.assertEqual(geometry.update(dict(state,time=2.,held=True)),new)

    def test_task_snapshot_reads_only_copy_and_refuses_attached_or_expert_motion(self):
        segment=dict(x=[.2,0.,.77],v=[.2,0.,0.],a=[0.,0.,0.],t0=0.)
        task=SimpleNamespace(target_state_list=[deepcopy(segment)],t=.1,area=list(AREA),_bool_expert=False,
            pyrep=SimpleNamespace(get_simulation_timestep=lambda:.05))
        state=dict(held=False,time=.1,task_time=.1,object_position=[.21,0.,.77],object_linear_velocity=[.2,0.,0.])
        forecast_from_private_task(task,state)
        self.assertEqual(task.target_state_list,[segment]);self.assertEqual(task.t,.1)
        with self.assertRaises(ValueError):forecast_from_private_task(task,dict(state,held=True))
        task._bool_expert=True
        with self.assertRaises(ValueError):forecast_from_private_task(task,state)

    def test_prepared_roster_and_source_closure_no_heldout_or_public_oracle(self):
        plan=c.validate_plan(PLAN)
        self.assertEqual(len(plan['parents']),12)
        self.assertTrue(all(p['role']!='heldout'and p['episode_id'].endswith('_teacherpass_v2')for p in plan['parents']))
        for parent in plan['parents']:self.assertFalse(worker.check(PLAN,parent['parent_id'])['ready'])
        s,n=c.source_map(PLAN)
        self.assertEqual(len(n),8);self.assertTrue(all(not Path(p).is_absolute()for p in s))
        with self.assertRaises(ValueError):worker.check(PLAN,'Cube_native_generated_ordinary_seed2611042006')
        self.assertIs(TeacherObservationRecorder,OriginalRecorder)

    def test_real_setup_import_paths_with_only_learned_factory_stubbed(self):
        # Execute the real copied setup, including both original planner
        # imports/constructors, through a terminal-before-planning fixture.
        # No learned checkpoint/model, NN forward, executor or Native is used.
        state=dict(time=0.,held=False,object_position=np.array([.2,0.,.77]),
            tcp_position=np.array([.3,0.,1.]),native_success=False)
        backend=SimpleNamespace(read_state=lambda:deepcopy(state),terminal=lambda:True,motion_events=[])
        executor=SimpleNamespace(limits='synthetic-unused',config=SimpleNamespace(reference_frequency=10.))
        config=c.read(c.REFERENCE)
        with tempfile.TemporaryDirectory()as directory:
            with patch('our_mani.local_mpc_control_20261103.response_factory.build_planner',
                       return_value=(SimpleNamespace(),SimpleNamespace(),{'synthetic_factory_stub_no_weights':True}))as factory:
                result=run_private_oracle_teacher(backend,executor,Path(directory),config,capture_rgbd=False)
            self.assertEqual(factory.call_count,1)
            self.assertEqual(result['cycles'],0)
            self.assertTrue(result['private_exact_future_oracle'])
            self.assertFalse(result['GRU_used'])
            self.assertTrue((Path(directory)/'controller_provenance.json').is_file())
        # Every statically copied relative import resolves in its NEW package.
        for file in (c.ROOT/'our_mani/cube_exact_future_teacher_20261104_v2').glob('*.py'):
            for node in ast.walk(ast.parse(file.read_text())):
                if isinstance(node,ast.ImportFrom)and node.level:
                    module=importlib.util.resolve_name('.'*node.level+(node.module or ''),
                        'our_mani.cube_exact_future_teacher_20261104_v2')
                    self.assertIsNotNone(importlib.util.find_spec(module),(file.name,module))


if __name__=='__main__':unittest.main()
