from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

import torch

from our_mani.world_tcp_command_warmstart_20261104_v3 import cache,common


class CacheContracts(unittest.TestCase):
    def test_target_stream_decodes_only_requested_objects(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'targets.json'
            p.write_text(json.dumps(dict(labels_only=True,targets=[dict(value=i,text='brace} and escaped"')for i in range(10)])))
            original=json.loads;decoded=[]
            def decode(text):decoded.append(text);return original(text)
            with mock.patch.object(cache.json,'loads',side_effect=decode):result=cache.selected_targets(p,[1,4])
            self.assertEqual(set(result),{1,4});self.assertEqual(len(decoded),2)
            self.assertEqual(result[4]['value'],4)

    def test_inference_outputs_become_ordinary_before_trainable_projection(self):
        with torch.inference_mode():value=torch.ones(2,3)
        self.assertTrue(torch.is_inference(value))
        ordinary=cache.ordinary_cpu(value,torch.float32)
        self.assertFalse(torch.is_inference(ordinary))
        layer=torch.nn.Linear(3,1);layer(ordinary).sum().backward()
        self.assertIsNotNone(layer.weight.grad)

    def test_first_load_and_LRU_reload_under_selection_context_remain_ordinary(self):
        # Exercise the real loader: metadata-only fixture, serialized full-parent
        # tensors, first selection access, LRU eviction/reload, then backward.
        from collections import OrderedDict
        with tempfile.TemporaryDirectory()as tmp:
            shapes=dict(visual_memory=(16,86,1024),visual_memory_mask=(16,86),current_robot=(16,14),
                camera_calibration=(16,4,23),task_e=(16,16),fine_visual=(16,512,512),
                fine_geometry=(16,512,6),fine_token_mask=(16,512))
            encoded={k:torch.zeros(shape,dtype=torch.bool if k.endswith('_mask')else torch.float32)
                     for k,shape in shapes.items()}
            tensors=cache.cache_inputs(encoded)
            path=Path(tmp)/'parent.pt';identities=[dict(global_record_index=i)for i in range(16)]
            torch.save(dict(inputs=tensors,identities=identities),path)
            dataset=cache.CachedCommandDataset.__new__(cache.CachedCommandDataset)
            dataset.master_indices=list(range(3))
            dataset.public=dict(chunks=[dict(file=cache.pointer(path))for _ in range(3)],
                records=[dict(chunk_index=i,row_in_chunk=0,identity=identities[0])for i in range(3)])
            dataset.labels=dict(target=dict(position=torch.zeros(3,3),rotation=torch.eye(3).repeat(3,1,1),
                opening=torch.ones(3)),expert_stage=['labelside']*3)
            dataset._lru=OrderedDict()
            with torch.inference_mode():
                for index in (0,1,2,0):
                    item=dataset[index]
                    self.assertTrue(all(not torch.is_inference(v)for v in item['inputs'].values()))
                    self.assertTrue(all(not torch.is_inference(v)for v in item['target'].values()))
                self.assertEqual(list(dataset._lru),[2,0])
            projection=torch.nn.Linear(14,1)
            projection(item['inputs']['current_robot']).sum().backward()
            self.assertIsNotNone(projection.weight.grad)

    def test_partial_public_cache_cannot_decode_command_labels(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'cache_manifest.json';p.write_text(json.dumps(dict(schema=cache.CACHE_SCHEMA,complete=True,
                public_features_frozen=True,decoded_command_targets=False,role_counts=dict(fit=1,selection=0),records=[],chunks=[])))
            out=Path(tmp)/'join'
            with mock.patch.object(cache,'selected_targets',side_effect=AssertionError('labels')):
                with self.assertRaises(ValueError):cache.join_command_labels(p,out)
            self.assertFalse(out.exists())

    def test_worst_history_canary_indices_are_metadata_only(self):
        class Fixture:
            global_indices=list(range(8))
            full=dict(records=[dict(current_frame_index=i)for i in (0,2,5,10,0,6,7,1)])
        self.assertEqual(cache.first_full_windows(Fixture()),[2,3,5,6])

    def test_cache_geometry_remains_FP32_and_has_no_private_fields(self):
        shapes=dict(visual_memory=(1,86,1024),visual_memory_mask=(1,86),current_robot=(1,14),camera_calibration=(1,4,23),
                    task_e=(1,16),fine_visual=(1,512,512),fine_geometry=(1,512,6),fine_token_mask=(1,512))
        encoded={k:torch.zeros(shape,dtype=torch.bool if k.endswith('_mask')else torch.float32)for k,shape in shapes.items()}
        tensors=cache.cache_inputs(encoded);cache.validate_cached_inputs(tensors,1)
        self.assertEqual(tensors['fine_geometry'].dtype,torch.float32)
        self.assertEqual(tensors['fine_visual'].dtype,torch.bfloat16)
        with self.assertRaises(ValueError):cache.validate_cached_inputs(dict(tensors,GT=False),1)

    def test_cached_collation_preserves_targets_separate_from_inputs(self):
        row=dict(inputs={key:torch.tensor([1.])for key in cache.INPUT_KEYS},
            target=dict(position=torch.zeros(3),rotation=torch.eye(3),opening=torch.tensor(1.)),
            identity=dict(global_record_index=1,task='synthetic'),expert_stage='labelside')
        batch=cache.collate_cached([row,row])
        self.assertEqual(set(batch['inputs']),set(cache.INPUT_KEYS));self.assertNotIn('expert_stage',batch['inputs'])
        self.assertEqual(batch['target']['position'].shape,(2,3))

    def test_fixed_budget_and_microbatch_canary_measurement_gate(self):
        cfg=common.fixed_config()
        self.assertEqual(cfg['epochs'],5);self.assertEqual(cfg['effective_batch_size'],16)
        self.assertEqual(cfg['cache_dtypes']['fine_geometry'],'float32')
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'fake.json';p.write_text('{}')
            with self.assertRaises(ValueError):common.checked_canary(p,common.sha(p),sources={'s':'h'},
                subset_pointer={'s':'h'},encoder_pointer={'e':'h'},cache_dtypes=cfg['cache_dtypes'])


if __name__=='__main__':unittest.main()
