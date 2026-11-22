"""Explicit feedback protocol and source identity; old frozen family unchanged."""
from pathlib import Path
import hashlib,json
from ..cube_corrected_anchor_live_20261104_v2.sources import anchor_consistent_corrected_source_map
ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'config/framework/cube_native_success_protocol_20261104_v1/protocol.json'
PROTOCOL_SHA256='cfa854179a187d70d4d5c5b002e71146f1573c406607c9b01ec60fc13fff9db1'

def validate_native_success_protocol():
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()!=PROTOCOL_SHA256:raise ValueError('Explicit user Native-success-only protocol changed')
    value=json.loads(PROTOCOL.read_text())
    if value['visual_hold_probe_or_lift_metric_is_success_gate']is not False or value['Native_contact_feedback_is_explicit_policy_input']is not True:
        raise ValueError('Original Native success only and explicit contact feedback required')
    return value

def native_contact_source_map():
    validate_native_success_protocol();sources=anchor_consistent_corrected_source_map()
    for name in ('__init__.py','backend.py','observer.py','carry.py','features.py','candidate_runtime.py','planner.py','coordinator.py','producer.py','sources.py'):
        p=Path(__file__).with_name(name);sources[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    sources[str(PROTOCOL)]=PROTOCOL_SHA256
    return sources
