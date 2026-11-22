"""Independent V59 opt-in/early-rejection/archive checks; no Native/fit/TEST."""
from copy import deepcopy
import importlib,json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from our_mani.local_mpc_20261102 import records
from our_mani.benchmark_perception import BoxPrior
from our_mani.local_mpc_control_20261103.public_grasp_v42 import GoalAwareActiveVisualPublicBridge
from tests import test_public_candidate_v56_worker_wiring_independent_20261103 as old_fixture

ROOT=Path(__file__).resolve().parents[1]
BASE_CONFIG=ROOT/'config/framework/P4_student_closedloop_devVAL_v56_v1/00_val_static.json'
FLAG='candidate_darkred_target_v59'
NEW_SOURCES={'our_mani/local_mpc_control_20261103/'+name for name in(
    'darkred_goal_perception_v59.py','goal_aware_binding_v59.py','public_candidate_darkred_v59.py')}


class DarkRedV59WorkerRecordsTests(unittest.TestCase):
    def fixture(self):
        return old_fixture.V56WorkerWiringTests('test_default_and_explicit_false_construct_and_dispatch_only_original_v5')
    def config(self,**overrides):
        return self.fixture().config(candidate_goal_sphere_filter_v56=True,
            postconfirmed_position_maintenance_v56=True,**overrides)
    def enabled(self):
        result=self.config();result[FLAG]=True;return result
    def invalid_cases(self):
        cases=[]
        for value in(0,1,None,'true',[],{}):
            cfg=self.enabled();cfg[FLAG]=value;cases.append((cfg,'p1'))
        for key in('public_grasp_candidate_transformer_v5','candidate_goal_sphere_filter_v56','postconfirmed_position_maintenance_v56'):
            for value in(False,1,None,'true'):
                cfg=self.enabled();cfg[key]=value;cases.append((cfg,'p1'))
            cfg=self.enabled();cfg.pop(key);cases.append((cfg,'p1'))
        for mode in('p0','fk','audit','gripper'):cases.append((self.enabled(),mode))
        return cases

    def test_default_and_false_keep_original_v5_or_v56_routes(self):
        for goal,maintenance,expected in((False,False,'v5'),(True,False,'v56'),(False,True,'v56'),(True,True,'v56')):
            for value in('missing',False):
                cfg=self.fixture().config(candidate_goal_sphere_filter_v56=goal,postconfirmed_position_maintenance_v56=maintenance)
                if value!='missing':cfg[FLAG]=value
                with self.subTest(goal=goal,maintenance=maintenance,value=value):
                    report=self.fixture().mocked_worker(cfg)
                    self.assertEqual(report['rc'],0)
                    self.assertEqual([r[0]for r in report['constructors']],[expected])
                    self.assertEqual([r[0]for r in report['runners']],[expected])
                    self.assertEqual(report['native_calls'],1);self.assertEqual(report['closes'],1)

    def test_worker_invalid_type_or_scope_rejected_before_backend(self):
        for cfg,mode in self.invalid_cases():
            with self.subTest(config=cfg,mode=mode):
                report=self.fixture().mocked_worker(cfg,mode)
                self.assertEqual(report['rc'],1);self.assertEqual(report['native_calls'],0)
                self.assertEqual(report['constructors'],[]);self.assertEqual(report['runners'],[])
                self.assertEqual(report['failure']['type'],'ValueError')

    def test_true_constructs_v59_once_with_same_flags_and_old_v56_runner_contract(self):
        module=importlib.import_module('our_mani.local_mpc_control_20261103.public_candidate_darkred_v59')
        constructed=[];ran=[]
        def observer(raw,*args,**kwargs):
            result=old_fixture.FakeObserver(raw);constructed.append((kwargs,result));return result
        def run(backend,executor,output,config,**kwargs):
            ran.append((backend,config,kwargs));return dict(complete=True,CPU_fake_no_Native=True)
        with patch.object(module,'ObservedPublicCandidateDarkRedV59Backend',side_effect=observer),\
             patch.object(module,'run_public_candidate_darkred_v59',side_effect=run):
            report=self.fixture().mocked_worker(self.enabled())
        self.assertEqual(report['rc'],0);self.assertEqual(report['native_calls'],1)
        self.assertEqual(report['constructors'],[]);self.assertEqual(report['runners'],[])
        self.assertEqual(len(constructed),1);self.assertEqual(len(ran),1)
        options,backend=constructed[0]
        self.assertIs(options['goal_sphere_filter'],True);self.assertIs(options['postconfirmed_maintenance'],True)
        self.assertIs(ran[0][0],backend);self.assertIs(ran[0][1][FLAG],True)
        self.assertEqual(backend.samples,1);self.assertEqual(report['closes'],1)
        self.assertEqual(report['old42_calls'],0)
        self.assertIn('v59',report['inputs']['information_setting'])

    def test_records_default_false_same_inventory_true_adds_exact_three(self):
        cfg=json.loads(BASE_CONFIG.read_text());cfg.update(candidate_goal_sphere_filter_v56=True,postconfirmed_position_maintenance_v56=True)
        with TemporaryDirectory(prefix='darkred_v59_archive_CPU_',dir=ROOT/'config/framework')as directory:
            path=Path(directory)/'config.json';path.write_text(json.dumps(cfg));default=records.freeze_sources(path,'p1')
            cfg[FLAG]=False;path.write_text(json.dumps(cfg));disabled=records.freeze_sources(path,'p1')
            own=str(path.relative_to(ROOT));self.assertEqual(set(default),set(disabled))
            self.assertEqual({k:v for k,v in default.items()if k!=own},{k:v for k,v in disabled.items()if k!=own})
            cfg[FLAG]=True;path.write_text(json.dumps(cfg));enabled=records.freeze_sources(path,'p1')
            self.assertEqual(set(enabled)-set(disabled),NEW_SOURCES);self.assertEqual(set(disabled)-set(enabled),set())
            self.assertEqual({k:v for k,v in disabled.items()if k!=own},{k:v for k,v in enabled.items()if k not in NEW_SOURCES and k!=own})
            for name in NEW_SOURCES:self.assertEqual(enabled[name],records.sha(ROOT/name))

    def test_records_reject_invalid_flag_or_scope_for_every_stage(self):
        cfg=json.loads(BASE_CONFIG.read_text());cfg.update(candidate_goal_sphere_filter_v56=True,postconfirmed_position_maintenance_v56=True)
        with TemporaryDirectory(prefix='darkred_v59_invalid_CPU_',dir=ROOT/'config/framework')as directory:
            path=Path(directory)/'config.json'
            for small,stage in self.invalid_cases():
                invalid=deepcopy(cfg)
                for key in(FLAG,'public_grasp_candidate_transformer_v5','candidate_goal_sphere_filter_v56','postconfirmed_position_maintenance_v56'):
                    if key in small:invalid[key]=small[key]
                    else:invalid.pop(key,None)
                path.write_text(json.dumps(invalid))
                with self.subTest(config=small,stage=stage),self.assertRaises(ValueError):records.freeze_sources(path,stage)

    def test_missing_required_v59_source_fails_true_but_not_false(self):
        cfg=json.loads(BASE_CONFIG.read_text());cfg.update(candidate_goal_sphere_filter_v56=True,postconfirmed_position_maintenance_v56=True)
        real_sha=records.sha
        with TemporaryDirectory(prefix='darkred_v59_missing_CPU_',dir=ROOT/'config/framework')as directory:
            path=Path(directory)/'config.json'
            for missing in NEW_SOURCES:
                def digest(p):
                    if Path(p).resolve()==(ROOT/missing).resolve():raise FileNotFoundError('explicit missing V59 source')
                    return real_sha(p)
                cfg[FLAG]=True;path.write_text(json.dumps(cfg))
                with patch.object(records,'sha',side_effect=digest),self.assertRaises(FileNotFoundError):records.freeze_sources(path,'p1')
                cfg[FLAG]=False;path.write_text(json.dumps(cfg))
                with patch.object(records,'sha',side_effect=digest):disabled=records.freeze_sources(path,'p1')
                self.assertFalse(set(disabled)&NEW_SOURCES)

    def test_old_v42_binding_still_rejects_same_exact_tracker_with_minimum_red100(self):
        bridge=GoalAwareActiveVisualPublicBridge()
        self.assertEqual(bridge.tracker.prior,BoxPrior())
        replacement=BoxPrior(minimum_red=100)
        bridge.tracker.tracker.prior=replacement;bridge.tracker.prior=replacement
        with self.assertRaisesRegex(ValueError,'unchanged declared box prior'):
            bridge.binding.validate_dependencies()


if __name__=='__main__':unittest.main()
