"""CPU-only real text/real fitted Cube-weight wiring check; no fitting/Native.

Reads one TRAIN candidate batch, not VAL/TEST. Writes a new report only.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from our_mani.cube_corrected_live_20261104_v1.matched_candidate_runtime import MatchedCorrectedCandidateInference
from our_mani.cube_matched_candidate_20261104_v1.losses import candidate_loss
from our_mani.language_conditioned_cube_20261112_v1.model import (
    InstructionConditionedCubeCandidate, LanguageConditionedCubePolicy)
from our_mani.language_conditioned_cube_20261112_v1.text import FrozenCLIPTextEncoder, file_sha256


# These are actual red-Cube init_episode templates, not task-name inputs.
INSTRUCTIONS = [
    'pick up the red block and lift it up to the target',
    'grasp the red block to the target',
    'lift the red block up to the target',
    'pick up the red block and lift it up to the target (high speed)',
    'grasp the red block to the target  (high speed)',
    'lift the red block up to the target (high speed)',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Fresh report path required')
    torch.set_num_threads(2)
    torch.manual_seed(20261112)
    fit = ROOT / 'artifacts/framework/cube_full_reuse_20261112_v1/candidate_fit'
    status = json.loads((fit / 'fit_status.json').read_text())
    initial = fit / 'selected.pt'
    dataset = fit.parent / 'candidate_data/fit/dataset.npz'
    motion = fit.parent / 'motion_fit/selected.pt'
    inputs = {str(p): file_sha256(p) for p in (initial, dataset, motion)}
    expected = status['checkpoint']['sha256']
    if inputs[str(initial)] != expected:
        raise ValueError('Completed baseline checkpoint changed')
    source = MatchedCorrectedCandidateInference(initial, expected_sha256=expected,
                                               forecast_sha256=inputs[str(motion)])
    network = InstructionConditionedCubeCandidate.from_matched_checkpoint(
        initial, expected_sha256=expected, forecast_sha256=inputs[str(motion)]).eval()
    started = time.perf_counter()
    text_encoder = FrozenCLIPTextEncoder(args.clip_checkpoint)
    embedding = text_encoder(INSTRUCTIONS)
    uncached_seconds = time.perf_counter() - started
    started = time.perf_counter()
    repeated = text_encoder(INSTRUCTIONS)
    cached_seconds = time.perf_counter() - started
    assert torch.equal(embedding, repeated)
    assert not any(name.startswith('visual') for name, _ in text_encoder.named_parameters())

    # Check extracted text tower against the original official implementation.
    import clip
    reference, _ = clip.load(str(args.clip_checkpoint), device='cpu', jit=False)
    with torch.no_grad():
        expected_text = F.normalize(reference.encode_text(clip.tokenize(INSTRUCTIONS)), dim=-1)
    text_error = (embedding - expected_text).abs().max().item()
    assert text_error <= 1e-6
    del reference
    assert not torch.equal(embedding[0], embedding[3])
    try:
        text_encoder(['word ' * 100])
    except RuntimeError:
        long_instruction_rejected = True
    else:
        raise AssertionError('Instruction silently truncated')

    policy = LanguageConditionedCubePolicy(text_encoder, network).eval()
    with np.load(dataset, allow_pickle=False) as data:
        # Same public inputs twice isolates a text change; no motion from words.
        features = {k: torch.from_numpy(np.repeat(data[k][:1], 2, axis=0).copy())
                    for k in ('future_tokens', 'robot', 'geometry', 'context_c')}
        targets = [torch.from_numpy(np.repeat(data[k][:1], 2, axis=0).copy())
                   for k in ('target_position', 'target_rotation', 'target_dt', 'target_time_valid')]
    before = {k: v.clone() for k, v in features.items()}
    paired_text = [INSTRUCTIONS[0], INSTRUCTIONS[3]]
    with torch.no_grad():
        old = source._model(**features, task_e=torch.full((2, 16), .125))
        new = policy(instructions=paired_text, **features)
    keys = [k for k, v in old.items() if isinstance(v, torch.Tensor)]
    parity = all(torch.equal(old[k], new[k]) for k in keys)
    assert parity
    assert torch.equal(new['task_e'], torch.full((2, 16), .125))

    # Backward only: no optimizer step, selection or new learned-weight artifact.
    policy.train()
    assert not text_encoder.training
    policy.candidate.eval()  # deterministic real-weight gradient check
    prediction = policy(instructions=paired_text, **features)
    loss = candidate_loss(prediction, *targets)['per_example'].mean()
    loss.backward()
    grad = network.language.to_task_e.weight.grad
    grad_norm = grad.norm().item()
    assert grad_norm > 0 and bool(torch.isfinite(grad).all())
    assert all(p.grad is None for p in text_encoder.parameters())

    # An explicit synthetic parameter intervention checks the conditional path.
    # Restore it immediately. This does NOT train or qualify language behavior.
    with torch.no_grad():
        saved = network.language.to_task_e.weight.clone()
        network.language.to_task_e.weight.normal_(std=.05)
        conditioned = policy(instructions=paired_text, **features)
        network.language.to_task_e.weight.copy_(saved)
    action_delta = (conditioned['positions'][0] - conditioned['positions'][1]).abs().max().item()
    assert action_delta > 0
    assert all(torch.equal(v, before[k]) for k, v in features.items())
    assert all(file_sha256(p) == h for p, h in inputs.items())
    source._assert_frozen()
    report = dict(
        complete=True, check='CPU_real_text_real_fitted_Cube_weight_wiring_only',
        instructions=INSTRUCTIONS, text_encoder=text_encoder.provenance(),
        official_CLIP_text_max_abs_error=text_error,
        normal_high_text_cosine=F.cosine_similarity(embedding[:3], embedding[3:]).tolist(),
        uncached_load_and_encode_seconds=uncached_seconds, cached_six_text_seconds=cached_seconds,
        text_cache_exact=True, overlong_instruction_rejected=long_instruction_rejected,
        baseline_candidate_outputs_exactly_equal_before_language_training=parity,
        action_loss_gradient_to_language_adapter_l2=grad_norm,
        text_tower_frozen_and_no_gradient=True,
        synthetic_adapter_intervention_action_max_delta_m=action_delta,
        synthetic_intervention_restored=True, input_arrays_unchanged=True,
        inputs_sha256=inputs, old_artifacts_unchanged=True,
        new_model_spec=network.spec(), language_optimizer_steps=0,
        operation_and_speed_heads_trained=False, language_conditioned_success_rate_measured=False,
        TRAIN_batch_only=True, VAL_read=False, TEST_read=False,
        Native_launched=False, GPU_used=False, default_runtime_changed=False,
        source_sha256={str(p): file_sha256(p) for p in [Path(__file__), *sorted(
            (ROOT / 'our_mani/language_conditioned_cube_20261112_v1').glob('*.py'))]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('complete', 'official_CLIP_text_max_abs_error',
        'baseline_candidate_outputs_exactly_equal_before_language_training',
        'action_loss_gradient_to_language_adapter_l2', 'language_optimizer_steps')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
