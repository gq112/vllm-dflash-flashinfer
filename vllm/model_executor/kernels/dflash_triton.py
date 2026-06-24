# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import Tensor

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_ctx"])
def _dflash_k_norm_rope_multi_cache_update_kernel(
    all_k_ptr,
    all_v_ptr,
    key_cache_ptr,
    value_cache_ptr,
    k_norm_weights_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    slot_mapping_ptr,
    num_ctx,
    all_k_stride_l,
    all_k_stride_t,
    all_k_stride_h,
    all_v_stride_l,
    all_v_stride_t,
    all_v_stride_h,
    key_cache_stride_l,
    key_cache_stride_b,
    key_cache_stride_o,
    key_cache_stride_h,
    value_cache_stride_l,
    value_cache_stride_b,
    value_cache_stride_o,
    value_cache_stride_h,
    block_size,
    eps,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    IS_NEOX: tl.constexpr,
    ROUND_TO_BF16: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    heads_per_layer = num_ctx * NUM_KV_HEADS
    layer_idx = row // heads_per_layer
    rem = row - layer_idx * heads_per_layer
    token_idx = rem // NUM_KV_HEADS
    head_idx = rem - token_idx * NUM_KV_HEADS

    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    block_idx = slot_idx // block_size
    block_offset = slot_idx - block_idx * block_size

    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM

    k_base = (
        all_k_ptr
        + layer_idx * all_k_stride_l
        + token_idx * all_k_stride_t
        + head_idx * all_k_stride_h
    )
    weight_base = k_norm_weights_ptr + layer_idx * HEAD_DIM

    k = tl.load(k_base + offs, mask=mask, other=0.0).to(tl.float32)
    sum_squares = tl.sum(k * k, axis=0)
    rms_rcp = tl.rsqrt(sum_squares / HEAD_DIM + eps)

    weight = tl.load(weight_base + offs, mask=mask, other=0.0).to(tl.float32)
    k_norm = k * rms_rcp * weight
    if ROUND_TO_BF16:
        k_norm = k_norm.to(tl.bfloat16).to(tl.float32)
    else:
        k_norm = k_norm.to(tl.float16).to(tl.float32)

    rope_mask = offs < ROTARY_DIM
    half_rotary_dim = ROTARY_DIM // 2
    if IS_NEOX:
        pair_offs = tl.where(offs < half_rotary_dim,
                             offs + half_rotary_dim,
                             offs - half_rotary_dim)
        sign = tl.where(offs < half_rotary_dim, -1.0, 1.0)
        cos_idx = offs % half_rotary_dim
    else:
        pair_offs = offs ^ 1
        sign = tl.where((offs & 1) == 0, -1.0, 1.0)
        cos_idx = offs // 2

    safe_pair_offs = tl.where(rope_mask, pair_offs, offs)
    pair_k = tl.load(k_base + safe_pair_offs, mask=mask, other=0.0).to(tl.float32)
    pair_weight = tl.load(
        weight_base + safe_pair_offs,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    pair_norm = pair_k * rms_rcp * pair_weight
    if ROUND_TO_BF16:
        pair_norm = pair_norm.to(tl.bfloat16).to(tl.float32)
    else:
        pair_norm = pair_norm.to(tl.float16).to(tl.float32)

    pos = tl.load(positions_ptr + token_idx)
    rope_base = cos_sin_cache_ptr + pos * ROTARY_DIM
    cos = tl.load(rope_base + cos_idx, mask=rope_mask, other=1.0).to(tl.float32)
    sin = tl.load(
        rope_base + half_rotary_dim + cos_idx,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    k_rope = k_norm * cos + sign * pair_norm * sin
    k_out = tl.where(rope_mask, k_rope, k_norm)

    key_base = (
        key_cache_ptr
        + layer_idx * key_cache_stride_l
        + block_idx * key_cache_stride_b
        + block_offset * key_cache_stride_o
        + head_idx * key_cache_stride_h
    )
    store_mask = mask & (slot_idx >= 0)
    tl.store(
        key_base + offs,
        k_out.to(key_cache_ptr.dtype.element_ty),
        mask=store_mask,
    )

    v_base = (
        all_v_ptr
        + layer_idx * all_v_stride_l
        + token_idx * all_v_stride_t
        + head_idx * all_v_stride_h
    )
    value_base = (
        value_cache_ptr
        + layer_idx * value_cache_stride_l
        + block_idx * value_cache_stride_b
        + block_offset * value_cache_stride_o
        + head_idx * value_cache_stride_h
    )
    v = tl.load(v_base + offs, mask=mask, other=0.0)
    tl.store(
        value_base + offs,
        v.to(value_cache_ptr.dtype.element_ty),
        mask=store_mask,
    )


def dflash_k_norm_rope_multi_cache_update(
    all_k: Tensor,
    all_v: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    k_norm_weights: Tensor,
    positions: Tensor,
    cos_sin_cache: Tensor,
    slot_mapping: Tensor,
    rope_head_size: int,
    is_neox: bool,
    eps: float,
) -> None:
    num_layers, num_ctx, num_kv_heads, head_dim = all_k.shape
    rotary_dim = cos_sin_cache.shape[1]

    if head_dim != rope_head_size:
        raise ValueError("DFlash Triton cache update expects rope_head_size=head_dim")
    if all_k.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("DFlash Triton cache update only supports fp16/bf16 K/V")
    if all_k.dtype != all_v.dtype or all_k.dtype != key_cache.dtype:
        raise ValueError("DFlash Triton cache update requires matching K/cache dtypes")
    if key_cache.dtype != value_cache.dtype:
        raise ValueError("DFlash Triton cache update requires matching cache dtypes")
    if key_cache.dim() != 5 or value_cache.dim() != 5:
        raise ValueError("DFlash Triton cache update expects 5D cache tensors")
    if key_cache.shape[0] != num_layers or value_cache.shape[0] != num_layers:
        raise ValueError("DFlash Triton cache update cache layer count mismatch")
    if key_cache.shape[3] != num_kv_heads or value_cache.shape[3] != num_kv_heads:
        raise ValueError("DFlash Triton cache update cache head count mismatch")
    if key_cache.shape[4] != head_dim or value_cache.shape[4] != head_dim:
        raise ValueError("DFlash Triton cache update cache head_dim mismatch")
    if all_k.stride(-1) != 1 or all_v.stride(-1) != 1:
        raise ValueError("DFlash Triton cache update requires contiguous head_dim")
    if key_cache.stride(-1) != 1 or value_cache.stride(-1) != 1:
        raise ValueError(
            "DFlash Triton cache update requires contiguous cache head_dim"
        )

    block_size = key_cache.shape[2]
    block_d = triton.next_power_of_2(head_dim)
    grid = (num_layers * num_ctx * num_kv_heads,)

    _dflash_k_norm_rope_multi_cache_update_kernel[grid](
        all_k,
        all_v,
        key_cache,
        value_cache,
        k_norm_weights,
        positions,
        cos_sin_cache,
        slot_mapping,
        num_ctx,
        all_k.stride(0),
        all_k.stride(1),
        all_k.stride(2),
        all_v.stride(0),
        all_v.stride(1),
        all_v.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        block_size,
        eps,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_dim,
        IS_NEOX=is_neox,
        ROUND_TO_BF16=all_k.dtype == torch.bfloat16,
        BLOCK_D=block_d,
        num_warps=1,
    )
