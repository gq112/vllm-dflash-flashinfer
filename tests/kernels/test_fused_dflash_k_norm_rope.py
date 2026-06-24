# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fused DFlash K RMSNorm + RoPE kernel."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.models.qwen3_dflash import _dflash_k_norm_rope_triton


def _op_available() -> bool:
    return hasattr(torch.ops._C, "dflash_k_norm_rope")


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _op_available(),
    reason="CUDA not available or fused DFlash K norm + RoPE op not built in",
)


def make_cos_sin_cache(
    max_pos: int,
    rotary_dim: int,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype=dtype)


def reference_dflash_k_norm_rope(
    all_k: torch.Tensor,
    k_norm_weights: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    eps: float,
) -> torch.Tensor:
    num_layers, num_ctx, num_kv_heads, head_dim = all_k.shape
    kv_size = num_kv_heads * head_dim
    out = torch.empty_like(all_k)
    for i in range(num_layers):
        ops.rms_norm(out[i], all_k[i], k_norm_weights[i], eps)

    flat = out.view(num_layers * num_ctx, kv_size)
    ops.rotary_embedding(
        positions.repeat(num_layers),
        flat,
        None,
        head_dim,
        cos_sin_cache,
        is_neox,
    )
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("num_layers", [1, 3])
@pytest.mark.parametrize("num_ctx", [1, 7, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("head_dim", [64, 128])
@torch.inference_mode()
def test_dflash_k_norm_rope_matches_reference(
    dtype: torch.dtype,
    is_neox: bool,
    num_layers: int,
    num_ctx: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    torch.manual_seed(0)
    device = "cuda"
    eps = 1e-6
    max_pos = 4096

    all_k = torch.randn(
        num_layers,
        num_ctx,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    k_norm_weights = torch.randn(
        num_layers,
        head_dim,
        dtype=dtype,
        device=device,
    )
    positions = torch.randint(
        0,
        max_pos,
        (num_ctx,),
        dtype=torch.int64,
        device=device,
    )
    cos_sin_cache = make_cos_sin_cache(max_pos, head_dim, dtype, device)

    expected = reference_dflash_k_norm_rope(
        all_k,
        k_norm_weights,
        positions,
        cos_sin_cache,
        is_neox,
        eps,
    )
    actual = torch.empty_like(all_k)
    ops.dflash_k_norm_rope(
        all_k,
        actual,
        k_norm_weights,
        positions,
        cos_sin_cache,
        head_dim,
        is_neox,
        eps,
    )
    actual_triton = torch.empty_like(all_k)
    _dflash_k_norm_rope_triton(
        all_k,
        actual_triton,
        k_norm_weights,
        positions,
        cos_sin_cache,
        eps,
        is_neox,
    )

    if dtype == torch.float16:
        atol, rtol = 2e-3, 2e-3
    else:
        atol, rtol = 1e-2, 1e-2
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_triton, expected, atol=atol, rtol=rtol)
