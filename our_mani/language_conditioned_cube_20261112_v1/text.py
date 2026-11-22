"""Local, frozen OpenAI CLIP RN50 text tower, with bounded instruction caching.

RN50 names the released checkpoint; its visual tower is discarded. No images,
task IDs, scene names, Native state, network downloads or remote API are used.
"""
from collections import OrderedDict
from pathlib import Path
import hashlib

import torch
from torch import nn
from torch.nn import functional as F


RN50_SHA256 = 'afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_instructions(instructions):
    if (type(instructions) not in (list, tuple) or not 1 <= len(instructions) <= 64
            or any(type(s) is not str or not s.strip() or len(s) > 1024 for s in instructions)):
        raise ValueError('1..64 nonempty instruction strings, at most1024 characters each, required')
    return tuple(instructions)


class FrozenCLIPTextEncoder(nn.Module):
    """English instruction -> L2-normalized FP32[1024]; always frozen/eval.

    Tokenization rejects >77 tokens instead of silently losing a goal/suffix.
    The cache contains only frozen text embeddings, not trainable task_e.
    """
    output_dim = 1024

    def __init__(self, checkpoint, *, expected_sha256=RN50_SHA256):
        super().__init__()
        path = Path(checkpoint).expanduser().resolve(strict=True)
        if expected_sha256 != RN50_SHA256 or file_sha256(path) != RN50_SHA256:
            raise ValueError('Exact original local CLIP RN50 checkpoint required')
        import clip
        from clip import clip as clip_api
        from clip import model as clip_model
        from clip import simple_tokenizer

        # A local file path bypasses clip.load's model-name download route.
        original, _ = clip.load(str(path), device='cpu', jit=False)
        self.token_embedding = original.token_embedding
        self.positional_embedding = original.positional_embedding
        self.transformer = original.transformer
        self.ln_final = original.ln_final
        self.text_projection = original.text_projection
        if self.text_projection.shape != (512, self.output_dim):
            raise ValueError('Original RN50 text projection shape required')
        self._tokenize = clip.tokenize
        self._cache = OrderedDict()
        self._provenance = dict(
            encoder='OpenAI_CLIP_RN50_text_only', checkpoint_path=str(path),
            checkpoint_sha256=RN50_SHA256, pretrained=True, frozen=True,
            output_dimension=self.output_dim, l2_normalized=True,
            context_length=77, silent_truncation=False, image_tower_retained=False,
            inference_input='instruction_text_only', chinese_instruction_quality_verified=False,
            source_sha256={str(Path(m.__file__).resolve()): file_sha256(m.__file__)
                           for m in (clip_api, clip_model, simple_tokenizer)},
            tokenizer_bpe_sha256=file_sha256(simple_tokenizer.default_bpe()))
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        # Calling parent_policy.train() must not unfreeze/change this tower.
        super().train(False)
        return self

    def provenance(self):
        from copy import deepcopy
        return deepcopy(self._provenance)

    @torch.no_grad()
    def forward(self, instructions):
        instructions = validate_instructions(instructions)
        if any(p.requires_grad for p in self.parameters()):
            raise ValueError('Text tower must remain frozen')
        if self.text_projection.dtype != torch.float32:
            raise ValueError('Reference text tower uses FP32')
        missing = list(dict.fromkeys(s for s in instructions if s not in self._cache))
        device = self.text_projection.device
        if missing:
            # Official CLIP encode_text math; end-of-text is the largest token ID.
            tokens = self._tokenize(missing, context_length=77, truncate=False).to(device)
            x = self.token_embedding(tokens) + self.positional_embedding
            x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = self.ln_final(x)
            x = x[torch.arange(len(tokens), device=device), tokens.argmax(-1)] @ self.text_projection
            if not bool(torch.isfinite(x).all()) or bool((x.norm(dim=-1) < 1e-8).any()):
                raise ValueError('Finite nonzero pretrained text features required')
            x = F.normalize(x, dim=-1)
            for text, feature in zip(missing, x):
                self._cache[text] = feature.detach().cpu().clone()
        # Resolve the whole batch before eviction (a batch may exceed cache size).
        output = torch.stack([self._cache[s] for s in instructions]).to(device)
        for s in instructions:
            self._cache.move_to_end(s)
        while len(self._cache) > 32:
            self._cache.popitem(last=False)
        return output
