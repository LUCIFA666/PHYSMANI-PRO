from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json,unittest
from our_mani.multitask import interaction,training_geometry
from our_mani.multitask.normal_cup_native_v111 import binding,library_scope,ORIGINAL_LIBRARY
from our_mani.multitask.normal_cup_primary_fitting_v107 import TASK
from our_mani.multitask.normal_cup_primary_library_v109 import PrimarySequenceLibrary
from our_mani.multitask.task_actor_backend_v69 import persistent_reference_continuation_identity,create_persistent_reference_actor
from our_mani.multitask.interaction import CandidateConfig,MPCConfig
from our_mani.multitask.specs import TASK_SPECS
from our_mani.multitask.evaluation import file_sha
from test_multitask_interaction import CartesianRobot

ROOT=Path(__file__).resolve().parents[1];B=ROOT/'artifacts/framework/multitask_r10'


class MapPort:
    def __init__(self):self.registered={};self.noncolliding={};self.estimated_attachments={}
    def register_estimated_geometry(self,identity,*args,**kwargs):self.registered[identity]=kwargs
    def declare_noncolliding_geometry(self,identity,**kwargs):self.noncolliding[identity]=kwargs
    def bind_estimated_attachment(self,identity,pose,**kwargs):pass
    def check(self,*args,**kwargs):return False


class Tests(unittest.TestCase):
    def config(self):
        c=json.loads((B/'task_executor_smoke_v69_setup/configuration.json').read_text())
        p=B/'normal_cup_primary_policy_v110/report.json';report=json.loads(p.read_text())
        c['learned_artifacts'][TASK]['interaction']=report['model']
        c['normal_cup_primary_report']=dict(path=str(p.relative_to(ROOT)),sha256=file_sha(p))
        c['normal_cup_primary_v111']=binding(ROOT,c)
        library=PrimarySequenceLibrary.load(ROOT/report['model']['path'])
        identity=persistent_reference_continuation_identity(ROOT,c,TASK,'cv',library,
            CandidateConfig(**c['candidate_config']),MPCConfig(**c['mpc_config']))
        c['continuation_identities'].setdefault(TASK,{})['cv']=identity['sha256']
        return c

    def test_actual_factory_constructs_final_controller_and_all_six_supported_skills(self):
        c=self.config();asset=json.loads((ROOT/c['learned_artifacts'][TASK]['geometry']['path']).read_text())
        bindings={};handle=100
        for role,body in asset['bodies'].items():
            bindings[role]={}
            for component in body['components']:
                handle+=1;bindings[role][component['name']]=(handle,)
        port=MapPort();port.required_dynamic_role_ids=tuple(asset['bodies'])
        env=SimpleNamespace(spec=TASK_SPECS[TASK],map_service=port,robot_model=CartesianRobot())
        with patch.object(training_geometry,'declared_instance_bindings',return_value=bindings),library_scope(ROOT,c,TASK):
            actor,identity=create_persistent_reference_actor(ROOT,env,c,'cv_geometry',['place cup'],purpose='train_collection')
        self.assertIs(actor.continuous_reference_manager.controller,actor.controller)
        self.assertEqual(len(actor.controller.library.templates),6)
        self.assertIsInstance(actor.controller.library,PrimarySequenceLibrary)
        self.assertFalse(actor.controller.library.metadata['initial_sequence_policy_scope']['original_all_observed_partitions_gate_passed'])
        self.assertEqual(actor.controller.ranking,'geometry');self.assertIsNone(actor.controller.goal)
        self.assertIs(interaction.InteractionLibrary,ORIGINAL_LIBRARY)

    def test_consumer_scope_restores_after_failure(self):
        c=self.config()
        with self.assertRaises(RuntimeError):
            with library_scope(ROOT,c,TASK):
                self.assertIs(interaction.InteractionLibrary,PrimarySequenceLibrary)
                raise RuntimeError('fixture')
        self.assertIs(interaction.InteractionLibrary,ORIGINAL_LIBRARY)

    def test_scope_does_not_admit_another_task_or_changed_consumer_source_binding(self):
        c=self.config()
        with self.assertRaisesRegex(ValueError,'explicit initial-sequence'):
            with library_scope(ROOT,c,'place_cups_on_rotating_frame_high_speed'):pass
        c['normal_cup_primary_v111']['sources_sha256']['our_mani/multitask/normal_cup_native_v111.py']='0'*64
        with self.assertRaisesRegex(ValueError,'explicit initial-sequence'):
            with library_scope(ROOT,c,TASK):pass


if __name__=='__main__':unittest.main()
