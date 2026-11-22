from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from our_mani.multitask import continuous_reference_actor_backend_v3 as backend
from our_mani.multitask.interaction import InteractionLibrary, CandidateConfig, MPCConfig
from our_mani.multitask.protocol import MultitaskProtocolError
from our_mani.multitask.evaluation import file_sha
from scripts import run_geometry_continuous_reference_smoke_v3_r10 as smoke

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/framework/multitask_r10/geometry_continuous_reference_smoke_v2_setup'
TASK='moving_basketball_in_hoop'


def prepared(method):
    config=json.loads((BASE/'configuration.json').read_text())
    old=config['policy_overlay']['configurations']
    config['policy_overlay']=backend.persistent_reference_policy_overlay(ROOT,
        old['feedback'],{},old['causal_retarget'])
    library=InteractionLibrary.load(backend.factory.artifact_path(ROOT,config['learned_artifacts'][TASK]['interaction']))
    identity=backend.persistent_reference_continuation_identity(ROOT,config,TASK,method.split('_')[0],
        library,CandidateConfig(**config['candidate_config']),MPCConfig(**config['mpc_config']))
    config['continuation_identities'][TASK][method.split('_')[0]]=identity['sha256']
    files=json.loads((BASE/'frozen_inputs.json').read_text())['files']
    for value in config['policy_overlay']['sources'].values(): files[str(ROOT/value['path'])]=value['sha256']
    return dict(args=SimpleNamespace(method=method,task=TASK),config=config,files=files),identity


class PersistentBackendTests(unittest.TestCase):
    def test_actual_CPU_loaders_require_new_reference_and_dispatch_sources(self):
        for method in backend.METHODS:
            value,identity=prepared(method)
            result=backend.persistent_reference_actor_model_preflight(ROOT,value)
            self.assertEqual(result['continuation_identity'],identity)
            self.assertEqual(result['context_checkpoint_sha256'] is not None,method=='context_geometry')
            sources=value['config']['policy_overlay']['sources']
            self.assertEqual(set(sources),{'actor_backend','feedback_controller','persistent_reference',
                'direct_dispatch','causal_retarget'})
            for source in sources.values(): self.assertEqual(file_sha(ROOT/source['path']),source['sha256'])
            for key in ('persistent_reference','direct_dispatch'):
                altered=deepcopy(value);del altered['files'][str(ROOT/sources[key]['path'])]
                with self.assertRaises(MultitaskProtocolError): backend.persistent_reference_actor_model_preflight(ROOT,altered)
            altered=deepcopy(value)
            altered['config']['continuation_identities'][TASK][method.split('_')[0]]=identity['original_continuation']['sha256']
            with self.assertRaises(MultitaskProtocolError): backend.persistent_reference_actor_model_preflight(ROOT,altered)

    def test_unknown_overlay_and_extended_budget_are_rejected(self):
        value,_=prepared('cv_geometry')
        altered=deepcopy(value['config']);altered['policy_overlay']['configurations']['unknown']=True
        with self.assertRaises(MultitaskProtocolError): backend.verify_persistent_reference_policy_overlay(ROOT,altered)
        altered=deepcopy(value['config']);altered['simulation_seconds']=11.
        with self.assertRaises(MultitaskProtocolError): backend.verify_persistent_reference_policy_overlay(ROOT,altered)
        for method in ('cv_value','context_value','physmani'):
            value['args'].method=method
            with self.assertRaises(MultitaskProtocolError): backend.persistent_reference_actor_model_preflight(ROOT,value)

    def test_wrapper_scope_is_train_nonfitting_geometry_only(self):
        args=['--config','x','--manifest','x','--frozen-inputs','x','--task',TASK,
            '--raw-episode-id','episode104','--out','x','--split','train',
            '--method','cv_geometry','--development-demo']
        parsed=smoke.parser().parse_args(args)
        self.assertEqual(parsed.raw_episode_id,'episode104')
        self.assertTrue(parsed.development_demo)
        self.assertFalse(parsed.record_demo)
        self.assertFalse(parsed.demo_only)


if __name__=='__main__': unittest.main()
