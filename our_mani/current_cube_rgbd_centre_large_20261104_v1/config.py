"""New fixed large-model protocol; no mutation of the frozen small pipeline."""
from copy import deepcopy
import json
from pathlib import Path
from ..current_cube_rgbd_centre_20261103_v1.data import ROOT, sha256

PROTOCOL_PATH = ROOT / 'artifacts/framework/local_mpc_20261102/current_cube_dual_crop_centre_large_training_protocol_v1/protocol.json'
PROTOCOL_SHA256 = 'd157b5f282686e3532046c580658a3039ab290c30c814f1f22e1789c5522a4c1'
ENCODER_SHA256 = 'dbf998242ad13fea26ede9c4b17a0e2bfd0daeb581331d21342af9a6b6d06bfa'


def fixed_config():
    if sha256(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise ValueError('Immutable large-centre training protocol changed')
    value = json.loads(PROTOCOL_PATH.read_text())
    config = value['config']
    if (config['logical_batch_size'] != config['micro_batch_size'] * config['gradient_accumulation']
            or config['epochs'] * config['updates_per_epoch'] != config['planned_optimizer_steps']
            or config['encoder_initialization_sha256'] != ENCODER_SHA256):
        raise ValueError('Original fixed40/520/micro2x8 protocol required')
    return deepcopy(config)
