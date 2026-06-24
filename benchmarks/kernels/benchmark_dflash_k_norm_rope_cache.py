# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DFlash K RMSNorm + RoPE + KV cache update fusion.

Example:
    .venv/bin/python -m benchmarks.kernels.benchmark_dflash_k_norm_rope_cache
"""

import argparse

import torch

from vllm import _custom_ops as ops


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


def write_cache(
    all_k: torch.Tensor,
    all_v: torch.Tensor,
    kv_caches: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    for layer_idx, kv_cache in enumerate(kv_caches):
        ops.reshape_and_cache_flash(
            all_k[layer_idx],
            all_v[layer_idx],
            kv_cache[:, 0],
            kv_cache[:, 1],
            slot_mapping,
            "auto",
            k_scale,
            v_scale,
        )


def run_old_path(
    all_k: torch.Tensor,
    all_v: torch.Tensor,
    kv_caches: list[torch.Tensor],
    k_norm_weights: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    eps: float,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    num_layers, num_ctx, num_kv_heads, head_dim = all_k.shape
    kv_size = num_kv_heads * head_dim
    all_k_final = torch.empty_like(all_k)
    for layer_idx in range(num_layers):
        ops.rms_norm(
            all_k_final[layer_idx],
            all_k[layer_idx],
            k_norm_weights[layer_idx],
            eps,
        )
    ops.rotary_embedding(
        positions.repeat(num_layers),
        all_k_final.view(num_layers * num_ctx, kv_size),
        None,
        head_dim,
        cos_sin_cache,
        is_neox,
    )
    write_cache(all_k_final, all_v, kv_caches, slot_mapping, k_scale, v_scale)


def run_cuda_path(
    all_k: torch.Tensor,
    all_v: torch.Tensor,
    kv_caches: list[torch.Tensor],
    k_norm_weights: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    eps: float,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    all_k_final = all_k.new_empty(all_k.shape)
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
    write_cache(all_k_final, all_v, kv_caches, slot_mapping, k_scale, v_scale)


def run_cuda_cache_path(
    all_k: torch.Tensor,
    all_v: torch.Tensor,
    kv_caches: list[torch.Tensor],
    k_norm_weights: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    eps: float,
) -> None:
    for layer_idx, kv_cache in enumerate(kv_caches):
        ops.dflash_k_norm_rope_cache_update(
            all_k[layer_idx],
            all_v[layer_idx],
            kv_cache[:, 0],
            kv_cache[:, 1],
            k_norm_weights[layer_idx],
            positions,
            cos_sin_cache,
            slot_mapping,
            all_k.shape[-1],
            is_neox,
            eps,
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
    if not hasattr(torch.ops._C, "dflash_k_norm_rope_cache_update"):
        raise RuntimeError("dflash_k_norm_rope_cache_update op is not available")

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

    print(
        "num_ctx,old_ms,cuda_ms,cuda_cache_ms,"
        "cuda_speedup,cuda_cache_speedup,cuda_cache_vs_cuda"
    )
    for num_ctx in args.num_ctx:
        all_kv_flat = torch.randn(
            num_ctx,
            args.num_layers,
            2,
            args.num_kv_heads,
            args.head_dim,
            dtype=dtype,
            device=device,
        )
        all_k = all_kv_flat[:, :, 0].permute(1, 0, 2, 3)
        all_v = all_kv_flat[:, :, 1].permute(1, 0, 2, 3)
        positions = torch.randint(
            0,
            args.max_pos,
            (num_ctx,),
            dtype=torch.int64,
            device=device,
        )
        num_blocks = (num_ctx + args.block_size - 1) // args.block_size
        slot_mapping = torch.arange(num_ctx, dtype=torch.int64, device=device)
        kv_caches = [
            torch.empty(
                num_blocks,
                2,
                args.block_size,
                args.num_kv_heads,
                args.head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(args.num_layers)
        ]

        old_ms = benchmark(
            lambda: run_old_path(
                all_k.contiguous(),
                all_v.contiguous(),
                kv_caches,
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
        cuda_ms = benchmark(
            lambda: run_cuda_path(
                all_k,
                all_v,
                kv_caches,
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
        cuda_cache_ms = benchmark(
            lambda: run_cuda_cache_path(
                all_k,
                all_v,
                kv_caches,
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
            f"{num_ctx},{old_ms:.6f},{cuda_ms:.6f},{cuda_cache_ms:.6f},"
            f"{old_ms / cuda_ms:.3f}x,{old_ms / cuda_cache_ms:.3f}x,"
            f"{cuda_ms / cuda_cache_ms:.3f}x"
        )


if __name__ == "__main__":
    main()
