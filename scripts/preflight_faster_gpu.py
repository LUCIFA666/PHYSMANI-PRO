"""Small numerical CUDA check before attempting a VLA training smoke."""
from datetime import datetime, timezone
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def main():
    devices = jax.devices()
    if not any(d.platform == 'gpu' for d in devices):
        raise RuntimeError(f'FASTER GPU unavailable: {devices}')
    rng = np.random.default_rng(9)
    a = rng.normal(size=(64, 64)).astype(np.float32)
    b = rng.normal(size=(64, 64)).astype(np.float32)
    operation = jax.jit(lambda x, y: jnp.tanh(jnp.dot(x, y, precision=jax.lax.Precision.HIGHEST)))
    output = np.asarray(operation(jnp.asarray(a), jnp.asarray(b)))
    expected = np.tanh(a@b)
    np.testing.assert_allclose(output, expected, atol=2e-5, rtol=2e-4)
    # The vision patch embedding uses a convolution; a working CUDA matmul
    # alone does not establish cuDNN support on a Blackwell GPU.
    pixels = rng.normal(size=(1, 28, 28, 3)).astype(np.float32)
    kernel = rng.normal(size=(14, 14, 3, 32)).astype(np.float32)
    convolution = jax.jit(lambda x, w: jax.lax.conv_general_dilated(
        x, w, (14, 14), 'VALID', dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=jax.lax.Precision.HIGHEST))
    conv_output = np.asarray(convolution(jnp.asarray(pixels), jnp.asarray(kernel)))
    conv_expected = np.empty((1, 2, 2, 32), dtype=np.float32)
    for i in range(2):
        for j in range(2):
            conv_expected[0, i, j] = np.tensordot(pixels[0, i*14:(i+1)*14, j*14:(j+1)*14], kernel, axes=([0, 1, 2], [0, 1, 2]))
    np.testing.assert_allclose(conv_output, conv_expected, atol=2e-4, rtol=2e-4)
    bf16_output = np.asarray(convolution(jnp.asarray(pixels, dtype=jnp.bfloat16), jnp.asarray(kernel, dtype=jnp.bfloat16)), dtype=np.float32)
    if not np.isfinite(bf16_output).all():
        raise RuntimeError('non-finite BF16 vision convolution')
    root = Path(__file__).resolve().parents[1]
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
              'jax_version': jax.__version__, 'devices': [str(d) for d in devices],
              'maximum_absolute_error': float(np.max(np.abs(output-expected))),
              'vision_convolution_max_absolute_error': float(np.max(np.abs(conv_output-conv_expected))),
              'bf16_vision_convolution_finite': True,
              'kind': 'cuda_numerical_preflight_not_model_training'}
    out = root/'artifacts/framework/smoke/faster_gpu_preflight.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
