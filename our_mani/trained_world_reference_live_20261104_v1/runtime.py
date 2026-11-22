"""Actual frozen e2 -> world-v3 -> typed demo bank -> generic reference ABI.

This bridge supplies an expert-command reference, not an actuator command or
body forecast. Gripper intent remains pending; no Native or GRU/contact state
is constructed. CPU compatibility preparation only, no all16 qualification.
"""
from pathlib import Path
import hashlib
import time
from . import VERSION
from .history import PublicHistoryV1

ROOT=Path(__file__).resolve().parents[2]
E2_SHA256='0d8fe15f459e521c4b04f5dda1757bf37d2816a376d3182af889f4e9c0d10924'
WORLD_SHA256='59afa7407f4332bb757981f575a83a93ea9e41a6e5b8b0d357d74b2ed4991ac7'
E2=ROOT/'artifacts/framework/local_mpc_20261102/semantic_task_e207M_v2_deferred_run_v3/artifacts/fit10/selected.pt'
WORLD=ROOT/'artifacts/framework/local_mpc_20261102/world_command_v3_deferred_run_v1/artifacts/fit5/selected.pt'
SPLIT=ROOT/'artifacts/framework/local_mpc_20261102/current_network16_data_manifest_v1/manifest.json'
CACHE=ROOT/'artifacts/framework/local_mpc_20261102/world_command_v3_deferred_run_v1/artifacts/joined_commands/joined_manifest.json'


def _stat(path):
    s=Path(path).stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)


def source_map():
    from ..world_tcp_command_warmstart_20261104_v3.common import source_map as world_sources
    result=world_sources()
    paths=[Path(__file__).with_name(name)for name in ('__init__.py','history.py','runtime.py')]
    paths += [ROOT/'scripts/check_trained_world_reference_live_20261104_v1.py',
              ROOT/'our_mani/generic_reference_mpc_20261104_v1/public_observation.py']
    for path in paths:
        result[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def check():
    return dict(version=VERSION,source_sha256=source_map(),CPU_only=True,
        expected_e2_sha256=E2_SHA256,expected_world_sha256=WORLD_SHA256,
        actual_artifacts_exist=all(p.is_file()for p in (E2,WORLD,SPLIT,CACHE)),
        model_constructed=False,model_forward_calls=0,GPU=False,Native=False,
        native_execution_ready=False,opening_intent_issued=False,physical_handoff_qualified=False,
        blockers=['Real arm-to-gripper timing/current-object alignment remains undefined',
            'Official reset/VAL/TEST and current-task controller admission remain separate',
            'No multirole GRU/contact model or real-time execution qualification is supplied'])


class TrainedWorldReferenceLiveV1:
    def __init__(self,*,encoder_checkpoint=E2,world_checkpoint=WORLD,split_manifest=SPLIT,cache_manifest=CACHE,device='cpu'):
        if str(device)!='cpu':raise ValueError('This compatibility bridge is CPU-only; no implicit GPU execution')
        import torch
        from ..world_tcp_command_warmstart_20261104_v3.features import FrozenFinePublicSceneMemoryV3
        from ..world_tcp_command_warmstart_20261104_v3.evaluation import load_selected
        from ..world_tcp_command_warmstart_20261104_v3.common import pointer,verify_sources,read
        from ..world_tcp_command_warmstart_20261104_v3.cache import sha_json
        torch.set_num_threads(2)
        self.sources=source_map();verify_sources(self.sources)
        self.encoder=FrozenFinePublicSceneMemoryV3(encoder_checkpoint,E2_SHA256,split_manifest,device='cpu')
        self.model,self.metadata=load_selected(world_checkpoint,WORLD_SHA256,cache_manifest,device='cpu')
        if (self.metadata['encoder_checkpoint']!=pointer(encoder_checkpoint)
                or self.metadata['encoder_spec_sha256']!=sha_json(self.encoder.model.spec())):
            raise ValueError('Trained candidate must bind this exact e2 checkpoint and specification')
        joined=read(cache_manifest)
        if (joined['encoder_checkpoint']!=self.metadata['encoder_checkpoint']
                or joined['encoder_spec_sha256']!=self.metadata['encoder_spec_sha256']
                or joined['cache_dtypes']!=self.metadata['cache_dtypes']):
            raise ValueError('Original training feature/precision identity required, no compact encoder substitution')
        self._encoder=self.encoder;self._model=self.model
        self._files={str(Path(p).resolve()):_stat(p)for p in (encoder_checkpoint,world_checkpoint,split_manifest,cache_manifest)}
        self.history=PublicHistoryV1();self.last_private_diagnostic=None;self._fault=None
        self.encoder_forward_calls=0;self.world_forward_calls=0

    def clear_history(self):
        if self._fault:raise RuntimeError('Faulted bridge must be reconstructed, not silently retried')
        self.history.clear()

    def _validate(self):
        from ..world_tcp_command_warmstart_20261104_v3.model import WorldTCPCommandTransformerV3
        from ..world_tcp_command_warmstart_20261104_v3.common import verify_sources
        if self._fault:raise RuntimeError('Bridge faulted: '+self._fault)
        if (self.encoder is not self._encoder or self.model is not self._model
                or type(self.model)is not WorldTCPCommandTransformerV3
                or getattr(self.model.forward,'__func__',None)is not WorldTCPCommandTransformerV3.forward
                or self.model.training or any(p.requires_grad for p in self.model.parameters())):
            raise ValueError('Exact original frozen world model and feature adapter required')
        if any(_stat(path)!=stat for path,stat in self._files.items()):raise ValueError('Bound model/data artifact changed')
        verify_sources(self.sources)

    def predict_reference(self,packet,*,frame_index):
        import torch
        from ..world_tcp_command_warmstart_20261104_v3.cache import INPUT_KEYS,cache_inputs
        from ..world_tcp_command_warmstart_20261104_v3.runtime_adapter import bank_from_v3_model_output,select_generic_reference
        from ..generic_reference_mpc_20261104_v1.contracts import WorldTCPReference
        self._validate();self.history.append(packet,frame_index=frame_index)
        arrays=self.history.arrays();inputs={k:torch.from_numpy(v)for k,v in arrays.items()}
        started=time.perf_counter()
        try:
            with torch.autocast('cpu',dtype=torch.bfloat16):encoded=self.encoder.encode(**inputs)
            self.encoder_forward_calls+=1;encoder_seconds=time.perf_counter()-started
            # Match the declared BF16 stored memories and FP32 public geometry,
            # then the original training tensor_batch promotion back to FP32.
            materialized=cache_inputs(encoded)
            model_inputs={k:materialized[k].float()if materialized[k].is_floating_point()else materialized[k]for k in INPUT_KEYS}
            world_started=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):raw=self.model(**model_inputs)
            self.world_forward_calls+=1
            bank=bank_from_v3_model_output(raw)
            reference=select_generic_reference(bank,timestamp=packet['timestamp'],policy_sha256=WORLD_SHA256,
                e_sha256=E2_SHA256,selection_rule='highest_demo_likelihood')
            if type(reference)is not WorldTCPReference:raise TypeError('Exact existing generic MPC reference ABI required')
            self._validate()
            self.last_private_diagnostic=dict(version=VERSION,history=self.history.trace(),selected_index=reference.selected_index,
                encoder_checkpoint_sha256=E2_SHA256,world_checkpoint_sha256=WORLD_SHA256,
                encoder_forward_seconds=encoder_seconds,world_forward_seconds=time.perf_counter()-world_started,
                encoder_forward_calls=self.encoder_forward_calls,world_forward_calls=self.world_forward_calls,
                CPU_BF16=True,cross_device_bitwise_equivalence_claim=False,
                opening_intent_issued=False,physical_handoff_tau=None,body_forecast_supplied=False,
                actuator_h_supplied=False,Native_contact_supplied=False,qualified=False)
            return reference
        except BaseException as error:
            self._fault=type(error).__name__+': '+str(error);raise
