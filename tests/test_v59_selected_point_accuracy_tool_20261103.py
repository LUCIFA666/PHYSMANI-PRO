"""Synthetic posthoc gate/time contracts; no real case labels are read."""
from copy import deepcopy
import importlib.util,json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'artifacts/framework/local_mpc_20261102/P4_v59_actual_selected_point_accuracy_tool_v1/analyze.py'
spec=importlib.util.spec_from_file_location('v59_accuracy_tool',PATH)
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def fixture(root):
    jobs=[]
    for name in tool.COHORTS['pilot4']:
        case=root/'runs'/name;report=root/'evaluations'/name/'report.json';configfile=root/(name+'.json')
        cfg=dict(candidate_darkred_target_v59=True,candidate_goal_sphere_filter_v56=True,
            postconfirmed_position_maintenance_v56=True,physics_dt=.05,control_dt=.1)
        write(configfile,cfg);write(case/'run_inputs.json',dict(config=cfg));write(case/'result.json',dict(complete=True,source_unchanged=True))
        write(case.parent/(name+'.owner.actual_exit.json'),dict(actual_process_returncode=0,output=str(case)))
        write(case/'worker_exit.json',dict(exit_code=0,native_closed=True))
        sources={key:dict(path=str(case/key),sha256='0'*64) for key in (*tool.GT_NAMES,'result.json','run_inputs.json')}
        for key in ('result.json','run_inputs.json'):sources[key]['sha256']=tool.sha((case/key).read_bytes())
        write(report,dict(case=str(case),run=dict(result_complete=True,result_source_unchanged=True),
            scope=dict(GT_used_only_posthoc=True,TEST_sources_read=False),source_files=sources))
        jobs.append(dict(id=name,path=str(case),config=str(configfile),evaluation=str(report.parent)))
    plan=root/'plan.json';write(plan,dict(jobs=jobs));return plan,jobs


def state(index,opening=1.,object_x=0.,tcp_z=.005):
    return dict(time=index*.05,counted_tick=index,object_position=[object_x,0.,0.],
        object_quaternion=[0.,0.,0.,1.],tcp_position=[0.,0.,tcp_z],held=False,requested_opening=opening)


def control():
    action=dict(candidate_id='chosen',joint_goal=[0.]*7,opening=0.)
    return dict(time=0.,decision=dict(plan=dict(selected_candidate_id='chosen'),action=action,
        candidate_pool=dict(candidates=[dict(id='chosen',object_tcp_position=[0.,0.,0.],deadline=.8)])),
        action_rejected_reason=None,executed_action=dict(joint_goal=[0.]*7,opening=0.),
        executor=dict(ticks=[dict(index=0,clock_before=0.,clock_after=.05,gripper_target=0.,tick_completed=True)]))


class AccuracyToolTests(unittest.TestCase):
    def test_whole_cohort_gate_precedes_any_GT_and_even_evaluation_labels(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);plan,jobs=fixture(root);last=Path(jobs[-1]['path'])
            (last/'worker_exit.json').unlink();reader=tool.Reader()
            gate,_,_=tool.preflight(plan,'pilot4',root=root,reader=reader,enforce_canonical_plan=False)
            self.assertFalse(gate['ready']);self.assertEqual(reader.GT_opened,[])
            self.assertFalse(any(Path(path).name=='report.json' for path in reader.sha256))
            with self.assertRaisesRegex(ValueError,'whole-cohort'):reader.read(Path(jobs[0]['path'])/'native_ticks.jsonl')
            with self.assertRaisesRegex(ValueError,'entire fixed cohort'):
                tool.analyze(plan,'pilot4',root/'unused',root=root,enforce_canonical_plan=False)
            self.assertFalse((root/'unused').exists())

    def test_preflight_allclosed_remains_no_GT_and_rejects_forged_metadata(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);plan,jobs=fixture(root)
            gate,_,reader=tool.preflight(plan,'pilot4',root=root,enforce_canonical_plan=False)
            self.assertTrue(gate['ready']);self.assertEqual(reader.GT_opened,[])
            # No GT files exist; metadata-only preflight must still not open them.
            report=Path(jobs[0]['evaluation'])/'report.json';obj=json.loads(report.read_text())
            obj['source_files']['native_ticks.jsonl']['path']=str(root/'wrong/native_ticks.jsonl');write(report,obj)
            with self.assertRaisesRegex(ValueError,'pointer'):
                tool.preflight(plan,'pilot4',root=root,enforce_canonical_plan=False)
            obj=json.loads(plan.read_text());obj['jobs']=obj['jobs'][:-1];write(plan,obj)
            with self.assertRaisesRegex(ValueError,'entire fixed ordered cohort'):
                tool.preflight(plan,'pilot4',root=root,enforce_canonical_plan=False)

    def test_A_uses_first_completed50ms_not_dispatch_and_object_rotation(self):
        initial=state(0,tcp_z=.020);closed=state(1,opening=0.,tcp_z=.006)
        rows=tool.close_attempts('negative_kept',[initial,closed],[control()])
        self.assertEqual(len(rows),1);record=rows[0]
        self.assertTrue(record['A_available']);self.assertEqual(record['actual_first_closed_tick_time'],.05)
        self.assertEqual(record['A']['error_l2_m'],.006)
        self.assertEqual(record['A_dispatch_boundary_legacy']['error_l2_m'],.020)
        rotated=state(1,0.);rotated['object_position']=[1.,2.,3.]
        rotated['object_quaternion']=[0.,0.,np.sqrt(.5),np.sqrt(.5)]
        rotated['tcp_position']=[1.,2.01,3.]
        self.assertAlmostEqual(tool.alignment(rotated,np.array([.01,0.,0.]))['error_l2_m'],0.,places=14)

    def test_all_attempts_missing_failed_negative_retained_and_gate_failclosed(self):
        states=[state(0),state(1,0.),state(2,1.),state(3,0.)]
        rows=tool.close_attempts('static_no_proxy',states,[control()])
        self.assertEqual(len(rows),2);self.assertTrue(rows[0]['A_available']);self.assertFalse(rows[1]['A_available'])
        stats=tool.statistic([rows[0]['A']['error_l2_m']],len(rows))
        self.assertEqual(stats['missing'],1);self.assertFalse(stats['all_expected_samples_le_5cm'])
        self.assertFalse(stats['mean_le_3cm'])
        rejected=control();rejected['action_rejected_reason']='collision'
        self.assertFalse(tool.close_attempts('failed',states[:2],[rejected])[0]['A_available'])
        self.assertFalse(tool.statistic([],0)['all_expected_samples_le_5cm'])

    def test_optional_preclose_forecast_never_crosses_close_or_uses_deadline(self):
        states=[state(i) for i in range(21)]
        def public(anchor):
            times=[anchor+i*.05 for i in range(23)]
            return dict(result=dict(mode='FREE',timestamp=anchor,free_forecast_available=True,
                Native_inputs_used=False,future_observations_used=False,observations_fabricated=False,
                forecast=dict(object_ids=['target'],times=times,positions=[[[0.,0.,0.]] for _ in times])))
        rows,coverage=tool.preclose_forecasts('case',states,[public(0.),public(.2)],.6)
        self.assertEqual([(r['anchor_time'],r['horizon_seconds']) for r in rows],[(0.,.4)])
        self.assertEqual(coverage[0]['excluded_close_intervention'],1)
        self.assertEqual(coverage[1]['excluded_close_intervention'],2)
        states[3]['held']=True
        rows,coverage=tool.preclose_forecasts('case',states,[public(0.)],.6)
        self.assertEqual(rows,[]);self.assertEqual(coverage[0]['excluded_proxy_held'],1)
        # Missing exact endpoint is not interpolated into a successful label.
        with self.assertRaisesRegex(ValueError,'no interpolation'):tool.exact_state(states,.425)


if __name__=='__main__':unittest.main()
