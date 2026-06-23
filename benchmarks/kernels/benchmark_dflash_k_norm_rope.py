# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark DFlash K RMSNorm + RoPE fusion.

Example:
    .venv/bin/python -m benchmarks.kernels.benchmark_dflash_k_norm_rope
"""

import argparse

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.models.qwen3_dflash import _dflash_k_norm_rope_triton


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


def run_old_path(
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
    ops.rotary_embedding(
        positions.repeat(num_layers),
        out.view(num_layers * num_ctx, kv_size),
        None,
        head_dim,
        cos_sin_cache,
        is_neox,
    )
    return out


def run_new_path(
    all_k: torch.Tensor,
    k_norm_weights: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    eps: float,
) -> torch.Tensor:
    out = torch.empty_like(all_k)
    ops.dflash_k_norm_rope(
        all_k,
        out,
        k_norm_weights,
        positions,
        cos_sin_cache,
        all_k.shape[-1],
        is_neox,
        eps,
    )
    return out


def run_triton_path(
    all_k: torch.Tensor,
    k_norm_weights: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    eps: float,
) -> torch.Tensor:
    out = torch.empty_like(all_k)
    _dflash_k_norm_rope_triton(
        all_k,
        out,
        k_norm_weights,
        positions,
        cos_sin_cache,
        eps,
        is_neox,
    )
    return out


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

    device = "cuda"
    dtype = getattr(torch, args.dtype)
    eps = 1e-6
    torch.manual_seed(0)

    cos_sin_cache = make_cos_sin_cache(
        args.max_pos,
        args.head_dim,
        dtype,
        device,
    )
    k_norm_weights = torch.randn(
        args.num_layers,
        args.head_dim,
        dtype=dtype,
        device=device,
    )

    print("num_ctx,old_ms,cuda_ms,triton_ms,cuda_speedup,triton_speedup")
    for num_ctx in args.num_ctx:
        all_k = torch.randn(
            args.num_layers,
            num_ctx,
            args.num_kv_heads,
            args.head_dim,
            dtype=dtype,
            device=device,
        )
        positions = torch.randint(
            0,
            args.max_pos,
            (num_ctx,),
            dtype=torch.int64,
            device=device,
        )

        old_ms = benchmark(
            lambda: run_old_path(
                all_k, k_norm_weights, positions, cos_sin_cache, args.neox, eps
            ),
            args.iters,
            args.warmup,
        )
        new_ms = benchmark(
            lambda: run_new_path(
                all_k, k_norm_weights, positions, cos_sin_cache, args.neox, eps
            ),
            args.iters,
            args.warmup,
        )
        triton_ms = benchmark(
            lambda: run_triton_path(
                all_k, k_norm_weights, positions, cos_sin_cache, args.neox, eps
            ),
            args.iters,
            args.warmup,
        )
        print(
            f"{num_ctx},{old_ms:.6f},{new_ms:.6f},{triton_ms:.6f},"
            f"{old_ms / new_ms:.3f}x,{old_ms / triton_ms:.3f}x"
        )


if __name__ == "__main__":
    main()
