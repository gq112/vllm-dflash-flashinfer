# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DFlash K RMSNorm + RoPE + KV cache write fusion.

Example:
    .venv/bin/python -m benchmarks.kernels.benchmark_dflash_k_norm_rope_cache
"""

import argparse

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.models.qwen3_dflash import (
    _dflash_k_norm_rope_cache_triton_qwen3_4kv,
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


def run_cuda_then_cache_path(
    all_k: torch.Tensor,
    all_v: torch.Tensor,
    key_caches: list[torch.Tensor],
    value_caches: list[torch.Tensor],
    k_norm_weights: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    eps: float,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    all_k_final = torch.empty_like(all_k)
    ops.dflash_k_norm_rope(
        all_k,
        all_k_final,
        k_norm_weights,
        positions,
        cos_sin_cache,
        all_k.shape[-1],
        is_neox,
        eps,
    )
    for layer_idx in range(all_k.shape[0]):
        ops.reshape_and_cache_flash(
            all_k_final[layer_idx],
            all_v[layer_idx],
            key_caches[layer_idx],
            value_caches[layer_idx],
            slot_mapping,
            "auto",
            k_scale,
            v_scale,
        )


def run_triton_cache_path(
    all_k: torch.Tensor,
    all_v: torch.Tensor,
    key_caches: list[torch.Tensor],
    value_caches: list[torch.Tensor],
    k_norm_weights: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    eps: float,
) -> None:
    for layer_idx in range(all_k.shape[0]):
        _dflash_k_norm_rope_cache_triton_qwen3_4kv(
            all_k[layer_idx],
            all_v[layer_idx],
            key_caches[layer_idx],
            value_caches[layer_idx],
            k_norm_weights[layer_idx],
            positions,
            cos_sin_cache,
            slot_mapping,
            eps,
            is_neox,
        )


def benchmark(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-ctx", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--max-pos", type=int, default=4096)
    parser.add_argument("--neox", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not hasattr(torch.ops._C, "dflash_k_norm_rope"):
        raise RuntimeError("dflash_k_norm_rope op is not available")
    if args.num_kv_heads != 4 or args.head_dim != 128:
        raise RuntimeError("triton_cache benchmark currently targets Qwen3 4x128 KV")

    device = "cuda"
    dtype = getattr(torch, args.dtype)
    eps = 1e-6
    torch.manual_seed(0)

    cos_sin_cache = make_cos_sin_cache(args.max_pos, args.head_dim, dtype, device)
    k_norm_weights = torch.randn(
        args.num_layers,
        args.head_dim,
        dtype=dtype,
        device=device,
    )
    k_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    v_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

    print("num_ctx,cuda_cache_ms,triton_cache_ms,triton_cache_speedup")
    for num_ctx in args.num_ctx:
        all_k = torch.randn(
            args.num_layers,
            num_ctx,
            args.num_kv_heads,
            args.head_dim,
            dtype=dtype,
            device=device,
        )
        all_v = torch.randn_like(all_k)
        positions = torch.randint(
            0,
            args.max_pos,
            (num_ctx,),
            dtype=torch.int64,
            device=device,
        )
        num_blocks = triton_cdiv(num_ctx, args.block_size)
        slot_mapping = torch.arange(num_ctx, dtype=torch.int64, device=device)
        key_caches = [
            torch.empty(
                num_blocks,
                args.block_size,
                args.num_kv_heads,
                args.head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(args.num_layers)
        ]
        value_caches = [torch.empty_like(cache) for cache in key_caches]

        cuda_cache_ms = benchmark(
            lambda: run_cuda_then_cache_path(
                all_k,
                all_v,
                key_caches,
                value_caches,
                k_norm_weights,
                positions,
                slot_mapping,
                cos_sin_cache,
                args.neox,
                eps,
                k_scale,
                v_scale,
            ),
            args.iters,
            args.warmup,
        )
        triton_cache_ms = benchmark(
            lambda: run_triton_cache_path(
                all_k,
                all_v,
                key_caches,
                value_caches,
                k_norm_weights,
                positions,
                slot_mapping,
                cos_sin_cache,
                args.neox,
                eps,
            ),
            args.iters,
            args.warmup,
        )
        print(
            f"{num_ctx},{cuda_cache_ms:.6f},{triton_cache_ms:.6f},"
            f"{cuda_cache_ms / triton_cache_ms:.3f}x"
        )


def triton_cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


if __name__ == "__main__":
    main()
